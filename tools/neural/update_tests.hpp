#pragma once
#include "training/checkpoint.hpp"
#include <limits>
#include <sstream>

namespace blitz::neural::training {
inline void update_contracts() {
    auto check = [](bool value, const char* reason) {
        if (!value)
            throw std::runtime_error(reason);
    };
    auto gpu = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    struct Temporary {
        fs::path directory =
            fs::temp_directory_path() / ("blitz-update-scope-" + std::to_string(::getpid()));
        Temporary() {
            fs::create_directory(directory);
        }
        ~Temporary() {
            fs::remove_all(directory);
        }
    } temporary;
    torch::manual_seed(413);
    auto rejects = [&](auto&& call, const char* reason) {
        bool rejected = false;
        try {
            call();
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected, reason);
    };
    for (auto scope : {TrainableScope::RankingRow, TrainableScope::EndpointScorer})
        for (uint32_t width : {64u, 128u})
            for (bool captured : {false, true}) {
                ActionNetwork ranker(conditioned_placement_schema, width);
                ranker->to(torch::kCUDA);
                auto features = torch::randn({2, action_pool, placement_features}, gpu);
                auto labels = torch::zeros({2, action_pool}, gpu.dtype(torch::kUInt8));
                labels.select(1, 0).fill_(31);
                labels.select(1, 1).fill_(24);
                labels.select(1, 2).fill_(27);
                std::vector<std::array<std::vector<uint32_t>, 4>> bins(1);
                bins[0][0] = {0, 1};
                SamplingTables sampling({{0}}, bins, 2);
                UpdateSettings settings;
                settings.scope = scope;
                settings.decay = .2f;
                ActionUpdate update(ranker, features, labels, sampling, 7, 971, settings,
                                    torch::zeros({2, action_pool, 9}, gpu));
                const auto expected_use = scope == TrainableScope::EndpointScorer
                                              ? ModelUse::EndpointReuseOnly
                                              : ModelUse::Unrestricted;
                check(export_actions(ranker, {}).use == expected_use,
                      "training export lost its use scope");
                for (auto& moment : update.optimizer.mean)
                    moment.fill_(.13f);
                for (auto& moment : update.optimizer.variance)
                    moment.fill_(.17f);
                auto before = update.optimizer.snapshot();
                auto original = ranker->forward(features).detach().clone();
                if (captured)
                    update.capture();
                update.run(9);
                auto after = update.optimizer.snapshot();
                check(update.optimizer.state().step == 9, "rank-only update failed");
                check(!torch::equal(before[4].select(0, 0), after[4].select(0, 0)),
                      "ranking weight row did not change");
                for (size_t i = 0; i + 1 < before.size(); ++i) {
                    if (scope == TrainableScope::EndpointScorer && i % 6 < 4)
                        continue;
                    auto a = before[i], b = after[i];
                    if (i % 6 >= 4) {
                        a = a.slice(0, 1);
                        b = b.slice(0, 1);
                    }
                    check(torch::equal(a, b), "rank training changed frozen parameters or moments");
                }
                check(torch::equal(original.slice(2, 1), ranker->forward(features).slice(2, 1)) ==
                          (scope == TrainableScope::RankingRow),
                      "shared-trunk output compatibility was misrepresented");
                uint32_t expected = width + 1;
                if (scope == TrainableScope::EndpointScorer)
                    for (size_t i = 0; i < 4; ++i)
                        expected += uint32_t(ranker->parameters()[i].numel());
                check(update.optimizer.total == expected,
                      "optimizer trained the wrong parameter ranges");
                check(torch::equal(before[0], after[0]) == (scope == TrainableScope::RankingRow),
                      "endpoint feature extractor did not train or ranking-row trunk changed");
                std::stringstream stream;
                torch::serialize::OutputArchive output;
                update.optimizer.save(output);
                output.save_to(stream);
                torch::serialize::InputArchive input;
                input.load_from(stream);
                ActionNetwork other(conditioned_placement_schema, width);
                other->to(torch::kCUDA);
                {
                    torch::NoGradGuard guard;
                    auto p = ranker->parameters(), q = other->parameters();
                    for (size_t i = 0; i < p.size(); ++i)
                        q[i].copy_(p[i]);
                }
                DeviceAdam wrong(other->parameters());
                rejects([&] { wrong.load(input); }, "rank checkpoint resumed full-policy Adam");
                auto mismatch = settings;
                mismatch.scope = scope == TrainableScope::RankingRow
                                     ? TrainableScope::EndpointScorer
                                     : TrainableScope::RankingRow;
                DeviceAdam wrong_scope(other->parameters(), mismatch);
                rejects([&] { wrong_scope.load(input); },
                        "row-only and endpoint optimizer checkpoints mixed");
                std::stringstream legacy_bytes;
                torch::serialize::OutputArchive legacy;
                legacy.write("version", torch::tensor(int64_t(update_checkpoint_version)));
                legacy.write("sampler", torch::tensor(int64_t(sampler_version)));
                legacy.write("ranking_only", torch::tensor(int64_t(1)));
                legacy.save_to(legacy_bytes);
                torch::serialize::InputArchive legacy_input;
                legacy_input.load_from(legacy_bytes);
                DeviceAdam new_objective(other->parameters(), settings);
                rejects([&] { new_objective.load(legacy_input); },
                        "old pair-weighted optimizer silently resumed equal-state loss");
                ActionUpdate restored(other, features, labels, sampling, 7, 971, settings,
                                      torch::zeros({2, action_pool, 9}, gpu));
                restored.optimizer.load(input);
                if (captured)
                    restored.capture();
                update.run(2);
                restored.run(2);
                auto saved = update.optimizer.snapshot(), resumed = restored.optimizer.snapshot();
                for (size_t i = 0; i < saved.size(); ++i)
                    check(torch::equal(saved[i], resumed[i]),
                          "rank-only checkpoint restore differs");
                const auto file = temporary.directory / "checkpoint.pt";
                save_state(file, ranker, update.optimizer, update.optimizer.state().step);
                ActionNetwork loaded(conditioned_placement_schema, width);
                loaded->to(torch::kCUDA);
                load_checkpoint_model(file, loaded, torch::kCUDA);
                check(loaded->use == expected_use && export_actions(loaded, {}).use == expected_use,
                      "model-only checkpoint restore stripped endpoint scope");
                if (scope == TrainableScope::EndpointScorer) {
                    UpdateSettings row;
                    row.scope = TrainableScope::RankingRow;
                    ActionUpdate row_update(loaded, features, labels, sampling, 7, 971, row,
                                            torch::zeros({2, action_pool, 9}, gpu));
                    CheckpointWriter writer;
                    writer.submit(temporary.directory / "row", loaded, row_update.optimizer, {});
                    writer.join();
                    check(load_weights(temporary.directory / "row/model.blzn").use == expected_use,
                          "ranking-row writer widened an inherited endpoint model");
                    fs::remove_all(temporary.directory / "row");
                    rejects([&] { trained_model_use(loaded, TrainableScope::Joint); },
                            "joint updates widened endpoint use");
                    ActionNetwork legacy(placement_schema, width);
                    legacy->to(torch::kCUDA);
                    rejects(
                        [&] {
                            ActionUpdate invalid(legacy, features, labels, sampling, 7, 971,
                                                 settings, torch::zeros({2, action_pool, 9}, gpu));
                        },
                        "endpoint update admitted legacy v3 feature semantics");
                }
            }
    // Each public host entry must reject invalid scalar settings before any
    // device access. Null buffers deliberately make a missed launch guard fail.
    auto parameter = torch::ones({2}, gpu).requires_grad_();
    auto rejects_settings = [&](UpdateSettings settings) {
        rejects([&] { DeviceAdam optimizer({parameter}, settings); },
                "optimizer accepted invalid update settings");
        rejects(
            [&] {
                loss_update(nullptr, nullptr, nullptr, nullptr, nullptr, 1, 2, settings,
                            c10::cuda::getCurrentCUDAStream());
            },
            "action loss accepted invalid update settings");
        rejects(
            [&] {
                placement_loss_update(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                      1, 2, settings, c10::cuda::getCurrentCUDAStream());
            },
            "placement loss accepted invalid update settings");
        rejects(
            [&] {
                adam_update(nullptr, 1, 2, nullptr, nullptr, settings,
                            c10::cuda::getCurrentCUDAStream());
            },
            "Adam launch accepted invalid update settings");
    };
    for (auto field :
         {&UpdateSettings::margin, &UpdateSettings::auxiliary, &UpdateSettings::penalty,
          &UpdateSettings::lr, &UpdateSettings::decay, &UpdateSettings::max_norm})
        for (float value :
             {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
              -std::numeric_limits<float>::infinity(), -.25f}) {
            UpdateSettings settings;
            settings.*field = value;
            rejects_settings(settings);
        }
    for (auto field : {&UpdateSettings::lr, &UpdateSettings::max_norm}) {
        UpdateSettings settings;
        settings.*field = 0;
        rejects_settings(settings);
    }
    for (auto field : {&UpdateSettings::lr, &UpdateSettings::decay}) {
        UpdateSettings settings;
        settings.*field = 1.25f;
        rejects_settings(settings);
    }
    check(!parameter.grad().defined(), "rejected settings mutated optimizer parameters");
    for (auto settings :
         {UpdateSettings{0, 0, 0, 1, 0, .25f}, UpdateSettings{2, 3, 4, .125f, 1, 2}}) {
        DeviceAdam valid({parameter}, settings);
        valid.update();
        check(valid.state().step == 1 && torch::isfinite(parameter).all().item<bool>(),
              "valid scalar boundary settings failed");
    }
    // A finite negative second moment is corrupt too: sqrt(v) would silently
    // write NaN parameters while the update counter advances. Zero remains valid.
    for (float value :
         {-1.f, -std::numeric_limits<float>::min(), std::numeric_limits<float>::quiet_NaN(),
          std::numeric_limits<float>::infinity(), 0.f, .125f}) {
        DeviceAdam source(
            {torch::ones({3}, gpu).requires_grad_(), torch::ones({2}, gpu).requires_grad_()});
        source.variance.back()[1].fill_(value);
        torch::serialize::OutputArchive saved;
        source.save(saved);
        std::stringstream bytes;
        saved.save_to(bytes);
        torch::serialize::InputArchive loaded;
        loaded.load_from(bytes, torch::Device(torch::kCUDA));
        DeviceAdam restored(
            {torch::ones({3}, gpu).requires_grad_(), torch::ones({2}, gpu).requires_grad_()});
        if (!std::isfinite(value) || value < 0) {
            rejects([&] { restored.load(loaded); }, "corrupt saved Adam variance was accepted");
            check(restored.state().step == 0 && restored.parameters.back().eq(1).all().item<bool>(),
                  "corrupt checkpoint changed weights or advanced an update");
        } else {
            restored.load(loaded);
            check(restored.variance.back()[1].item<float>() == value,
                  "valid saved Adam variance changed on load");
            restored.update();
            check(restored.state().step == 1 &&
                      torch::isfinite(restored.parameters.back()).all().item<bool>(),
                  "valid saved Adam variance failed its first update");
        }
    }
    // Compare the fused loss with autograd at several test-owned weights,
    // including no preferred choices and entirely unknown/padded states.
    for (auto config :
         {UpdateSettings{.75f, .3f, .002f}, UpdateSettings{2.25f, 0, 0}, UpdateSettings{0, 0, 0}}) {
        auto labels = torch::tensor({15, 15, 11, 8, 8, 11, 0, 0, 0, 0, 0, 0}, torch::kUInt8)
                          .view({3, 4})
                          .to(torch::kCUDA);
        auto prediction = torch::randn({3, 4, 3}, gpu).requires_grad_();
        auto loss = action_loss(prediction, labels, labels.bitwise_and(Queried).ne(0),
                                config.margin, config.auxiliary, config.penalty);
        loss.backward();
        auto control = torch::zeros({int64_t(sizeof(UpdateState))}, gpu.dtype(torch::kUInt8));
        auto gradient = torch::zeros_like(prediction), losses = torch::empty({3}, gpu);
        loss_update(prediction.data_ptr<float>(), labels.data_ptr<uint8_t>(),
                    gradient.data_ptr<float>(), losses.data_ptr<float>(),
                    records<UpdateState>(control), 3, 4, config, c10::cuda::getCurrentCUDAStream());
        cuda_check(cudaGetLastError());
        auto host = control.cpu();
        UpdateState state;
        std::memcpy(&state, host.data_ptr(), sizeof(state));
        check(std::abs(state.loss - loss.item<float>()) < 2e-6,
              "fused action loss differs from reference");
        check(torch::allclose(gradient, prediction.grad(), 2e-5, 2e-6),
              "fused action loss gradient differs from reference");
        check(gradient[2].abs().sum().item<float>() == 0, "unknown actions acquired a gradient");
    }
    for (auto config :
         {UpdateSettings{.8f, .2f, .001f}, UpdateSettings{1.7f, .4f, 0}, UpdateSettings{0, 0, 0}}) {
        auto labels = torch::tensor({255, 59, 8, 16, 9, 25, 24, 0}, torch::kUInt8)
                          .view({2, 4})
                          .to(torch::kCUDA);
        auto y = torch::randn({2, 4, 12}, gpu).requires_grad_(),
             targets = torch::randn({2, 4, 9}, gpu);
        auto ids = torch::arange(2, gpu.dtype(torch::kInt32));
        auto joint = labels.bitwise_and(24).eq(24),
             preferred = labels.bitwise_and(4).ne(0).logical_and(joint),
             other = preferred.logical_not().logical_and(joint),
             pairs = preferred.unsqueeze(2).logical_and(other.unsqueeze(1));
        auto scores = y.select(2, 0);
        auto expected =
            (torch::relu(config.margin + scores.unsqueeze(1) - scores.unsqueeze(2)) * pairs).sum() /
            pairs.sum().clamp_min(1);
        for (unsigned h = 0; h < 3; ++h) {
            auto mask = h ? labels.bitwise_and(h == 1 ? 8 : 16).ne(0) : joint;
            auto z = y.select(2, h);
            expected = expected +
                       config.penalty * (z.square() * mask).sum() / (mask.sum().clamp_min(1) * 3);
            if (h) {
                auto t = labels.bitwise_and(h == 1 ? 1 : 2).ne(0).to(torch::kFloat32);
                expected = expected + config.auxiliary *
                                          (torch::binary_cross_entropy_with_logits(
                                               z, t, {}, {}, torch::Reduction::None) *
                                           mask)
                                              .sum() /
                                          mask.sum().clamp_min(1);
            }
        }
        for (unsigned g = 0; g < 3; ++g) {
            auto mask = labels.bitwise_and(32u << g).ne(0);
            auto d = y.slice(2, 3 + g * 3, 6 + g * 3) - targets.slice(2, g * 3, g * 3 + 3);
            auto loss = torch::where(d.abs() < 1, .5 * d.square(), d.abs() - .5);
            expected = expected + (loss * mask.unsqueeze(2)).sum() / (mask.sum().clamp_min(1) * 3);
        }
        expected.backward();
        auto control = torch::zeros({int64_t(sizeof(UpdateState))}, gpu.dtype(torch::kUInt8)),
             gradient = torch::empty_like(y), losses = torch::empty({2}, gpu);
        placement_loss_update(
            y.data_ptr<float>(), labels.data_ptr<uint8_t>(), targets.data_ptr<float>(),
            reinterpret_cast<uint32_t*>(ids.data_ptr<int32_t>()), gradient.data_ptr<float>(),
            losses.data_ptr<float>(), records<UpdateState>(control), 2, 4, config,
            c10::cuda::getCurrentCUDAStream());
        auto host = control.cpu();
        UpdateState state;
        std::memcpy(&state, host.data_ptr(), sizeof(state));
        check(std::abs(state.loss - expected.item<float>()) < 3e-6,
              "placement loss differs from independent autograd reference");
        check(torch::allclose(gradient, y.grad(), 2e-5, 2e-6), "placement loss gradient differs");
        check(gradient[0][2][2].item<float>() == 0 && gradient[0][3][1].item<float>() == 0 &&
                  gradient[1][3].abs().sum().item<float>() == 0,
              "unknown source/adjacent verdict became a negative");
    }
    // Repeating alternatives or changing the number of tied positives cannot
    // alter a sampled state's total ranking weight.
    for (unsigned positives : {1u, 3u}) {
        auto scores = torch::zeros({2, 16, 12}, gpu);
        auto labels = torch::zeros({2, 16}, gpu.dtype(torch::kUInt8));
        labels[0].slice(0, 0, 4).fill_(24);
        labels[0][0].fill_(28);
        labels[1].fill_(24);
        labels[1].slice(0, 0, positives).fill_(28);
        auto targets = torch::zeros({2, 16, 9}, gpu);
        auto control = torch::zeros({int64_t(sizeof(UpdateState))}, gpu.dtype(torch::kUInt8));
        auto gradient = torch::empty_like(scores), losses = torch::empty({2}, gpu);
        UpdateSettings config;
        config.scope = TrainableScope::RankingRow;
        config.penalty = 0;
        config.margin = positives == 1 ? .7f : 1.3f;
        placement_loss_update(scores.data_ptr<float>(), labels.data_ptr<uint8_t>(),
                              targets.data_ptr<float>(), nullptr, gradient.data_ptr<float>(),
                              losses.data_ptr<float>(), records<UpdateState>(control), 2, 16,
                              config, c10::cuda::getCurrentCUDAStream());
        check(std::abs(losses.sum().item<float>() - config.margin) < 2e-6,
              "pair multiplicity changed equal-state ranking loss");
        check(std::abs(gradient[0][0][0].item<float>() + .5f) < 2e-6 &&
                  std::abs(gradient[1].slice(0, 0, positives).select(1, 0).sum().item<float>() +
                           .5f) < 2e-6,
              "state with more preference pairs dominated ranking gradient");
        check(gradient.slice(2, 1).abs().sum().item<float>() == 0,
              "ranking loss trained frozen policy channels");
    }
    // Test actual evolving bias correction, clipping and decoupled weight decay.
    for (float max_norm : {.4f, 1.7f}) {
        std::vector<torch::Tensor> a{torch::randn({13}, gpu).requires_grad_(),
                                     torch::randn({259}, gpu).requires_grad_()},
            b;
        for (auto& p : a)
            b.push_back(p.detach().clone().requires_grad_());
        UpdateSettings settings;
        settings.lr = .003f;
        settings.decay = .02f;
        settings.max_norm = max_norm;
        DeviceAdam actual(a, settings);
        torch::optim::AdamW expected(
            b, torch::optim::AdamWOptions(settings.lr).weight_decay(settings.decay));
        for (unsigned step = 0; step < 12; ++step) {
            for (size_t i = 0; i < a.size(); ++i) {
                auto g = step == 0 ? torch::zeros_like(a[i])
                                   : torch::randn_like(a[i]) * (step % 2 ? .001f : 4.f);
                a[i].grad().copy_(g);
                b[i].mutable_grad() = g.clone();
            }
            torch::nn::utils::clip_grad_norm_(b, max_norm);
            expected.step();
            actual.update();
            check(actual.state().step == step + 1, "device AdamW step did not advance");
            for (size_t i = 0; i < a.size(); ++i) {
                auto& reference = static_cast<torch::optim::AdamWParamState&>(
                    *expected.state().at(b[i].unsafeGetTensorImpl()));
                check(torch::allclose(a[i], b[i], 2e-5, 2e-6),
                      "device AdamW parameters differ from stock reference");
                check(torch::allclose(actual.mean[i], reference.exp_avg(), 2e-5, 2e-7),
                      "device AdamW first moment differs");
                check(torch::allclose(actual.variance[i], reference.exp_avg_sq(), 2e-5, 2e-8),
                      "device AdamW second moment differs");
            }
        }
        auto before = actual.snapshot();
        a[1].mutable_grad().index_put_({7}, NAN);
        actual.update();
        actual.update();
        check(actual.state().failure == 2 && actual.state().step == 12,
              "bad gradients were not latched");
        auto after = actual.snapshot();
        for (size_t i = 0; i + 1 < before.size(); ++i)
            check(torch::equal(before[i], after[i]), "nonfinite gradient changed optimizer state");
    }
    // A nontrivial hierarchy exercises both uniform levels and variable pool masks.
    std::vector<std::vector<uint32_t>> categories{{0, 1}, {2}};
    std::vector<std::array<std::vector<uint32_t>, 4>> bins(3);
    bins[0][0] = {0, 1};
    bins[0][3] = {2};
    bins[1][1] = {3};
    bins[2][2] = {4, 5};
    SamplingTables tables(categories, bins, 6);
    auto x = torch::randn({6, action_pool, action_features}, gpu);
    auto labels = torch::full({6, action_pool}, 8, gpu.dtype(torch::kUInt8));
    labels.select(1, 0).fill_(15);
    labels.select(1, 1).fill_(11);
    labels.select(1, 15).zero_();
    ActionNetwork model;
    model->to(torch::kCUDA);
    ActionUpdate update(model, x, labels, tables, 17, 819);
    auto initial = update.optimizer.snapshot();
    update.capture();
    auto captured = update.optimizer.snapshot();
    for (size_t i = 0; i < initial.size(); ++i)
        check(torch::equal(initial[i], captured[i]), "capture changed initial training state");
    update.run(3);
    check(update.optimizer.state().step == 3, "graph counter frozen");
    auto first_ids = update.ids.clone();
    auto checkpoint = update.optimizer.snapshot();
    update.run(4);
    auto expected = update.optimizer.snapshot();
    check(!torch::equal(first_ids, update.ids), "graph replay repeats fixed sampled IDs");
    update.optimizer.restore(checkpoint);
    update.run(4);
    auto resumed = update.optimizer.snapshot();
    for (size_t i = 0; i < expected.size(); ++i)
        check(torch::equal(expected[i], resumed[i]),
              "captured checkpoint continuation is not exact");
    // Serialize optimizer state and restore into fresh allocations before capture.
    std::stringstream bytes;
    torch::serialize::OutputArchive saved;
    update.optimizer.save(saved);
    saved.save_to(bytes);
    ActionNetwork fresh;
    fresh->to(torch::kCUDA);
    {
        torch::NoGradGuard guard;
        auto p = fresh->parameters(), q = model->parameters();
        for (size_t i = 0; i < p.size(); ++i)
            p[i].copy_(q[i]);
    }
    ActionUpdate restored(fresh, x, labels, tables, 17, 819);
    torch::serialize::InputArchive loaded;
    loaded.load_from(bytes, torch::Device(torch::kCUDA));
    restored.optimizer.load(loaded);
    restored.capture();
    update.run(2);
    restored.run(2);
    auto full = update.optimizer.snapshot(), continued = restored.optimizer.snapshot();
    for (size_t i = 0; i < full.size(); ++i)
        check(torch::equal(full[i], continued[i]), "serialized graph continuation differs");
    // The first failed loss/input remains available even when a replay window
    // contains further launches. No parameter/moment/step write may follow it.
    auto before = update.optimizer.snapshot();
    x.fill_(NAN);
    update.run(3);
    auto failed = update.optimizer.state();
    check(failed.failure == 1 && failed.step == 9, "nonfinite loss did not freeze update counter");
    auto after = update.optimizer.snapshot();
    for (size_t i = 0; i + 1 < before.size(); ++i)
        check(torch::equal(before[i], after[i]), "nonfinite loss changed parameters/moments");
    for (uint32_t architecture : {action_schema, placement_schema}) {
        auto width = policy_inputs(architecture);
        auto source = torch::randn({6, action_pool, width}, torch::kFloat32);
        auto data = source.accessor<float, 3>();
        std::vector<uint8_t> labels_host(6 * action_pool,
                                         architecture == action_schema ? uint8_t(8) : uint8_t(24));
        const std::vector<unsigned> flags =
            architecture == action_schema
                ? std::vector<unsigned>{15, 16, 17, 18, 19, 20, 21, 39, 40,
                                        41, 42, 43, 44, 45, 67, 68, 75, 76}
                : std::vector<unsigned>{15, 16,  17,  18,  19,  20,  21,  39,  40,  41, 42,
                                        43, 44,  45,  67,  68,  75,  76,  95,  96,  97, 98,
                                        99, 100, 101, 119, 120, 121, 122, 123, 124, 125};
        for (unsigned s = 0; s < 6; ++s)
            for (unsigned row = 0; row < action_pool; ++row) {
                for (auto channel : flags)
                    data[s][row][channel] = float((s + row + channel) % 2);
                for (unsigned c = 48; c < 56; ++c)
                    data[s][row][c] = float(s + c) / 100;
                data[s][row][78] = data[s][row][23];
                data[s][row][79] = data[s][row][47];
            }
        for (unsigned s = 0; s < 6; ++s) {
            labels_host[s * action_pool] = architecture == action_schema ? 15 : 255;
            labels_host[s * action_pool + 15] = 0;
            source[s][15].zero_();
        }
        auto packed = pack_actions({source.data_ptr<float>(), size_t(source.numel())}, labels_host,
                                   6, action_pool, width);
        auto values = torch::from_blob(packed.values.data(), {6, action_pool, packed_width(width)},
                                       torch::kFloat32)
                          .to(torch::kCUDA),
             bits = torch::from_blob(packed.flags.data(), {6, action_pool}, torch::kInt32)
                        .to(torch::kCUDA),
             conditions = torch::from_blob(packed.conditions.data(), {6, 8}, torch::kFloat32)
                              .to(torch::kCUDA);
        auto mask =
            torch::from_blob(labels_host.data(), {6, action_pool}, torch::kUInt8).to(torch::kCUDA);
        torch::Tensor targets;
        if (architecture == placement_schema)
            targets = torch::randn({6, action_pool, 9}, gpu);
        ActionNetwork a(architecture), b(architecture);
        a->to(torch::kCUDA);
        b->to(torch::kCUDA);
        {
            torch::NoGradGuard guard;
            auto p = a->parameters(), q = b->parameters();
            for (size_t i = 0; i < p.size(); ++i)
                q[i].copy_(p[i]);
        }
        auto dense = source.to(torch::kCUDA);
        ActionUpdate raw(a, dense, mask, tables, 11, 782, {}, targets),
            compact(b, values, mask, tables, 11, 782, {}, targets, bits, conditions);
        check(torch::equal(compact.inputs(0, 6), dense), "packed dataset changed feature bits");
        raw.capture();
        compact.capture();
        raw.run(5);
        compact.run(5);
        auto r = raw.optimizer.snapshot(), c = compact.optimizer.snapshot();
        for (size_t i = 0; i < r.size(); ++i)
            check(torch::equal(r[i], c[i]), "packing changed captured updates");
        auto weights = export_actions(b, {{"test", true}});
        ActionCuda native(weights, {}, 7);
        auto probe = source.flatten(0, 1).contiguous();
        auto predicted = native.predict({probe.data_ptr<float>(), size_t(probe.numel())});
        std::vector<double> actual(predicted.begin(), predicted.end());
        check(
            legacy_numeric_pass(numeric_difference(
                actual, action_oracle({probe.data_ptr<float>(), size_t(probe.numel())}, weights))),
            "placement native/FP64 export parity failed");
    }
    std::cout << json({{"update_contracts", true},
                       {"sampler", sampler_version},
                       {"checkpoint_schema", update_checkpoint_version},
                       {"captured_resume_exact", true}})
                     .dump()
              << '\n';
}
} // namespace blitz::neural::training
