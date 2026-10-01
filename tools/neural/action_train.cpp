#include "tools/neural/update_tests.hpp"
#include "training/checkpoint.hpp"
#include <c10/cuda/CUDACachingAllocator.h>
#include <csignal>
#include <iostream>
#include <random>
using namespace blitz;
using namespace blitz::neural;
using namespace blitz::neural::training;
static volatile std::sig_atomic_t stopped = 0;
static void stop(int) {
    stopped = 1;
}
struct Dataset {
    std::vector<float> x, targets;
    std::vector<uint8_t> labels;
    std::vector<std::array<std::vector<uint32_t>, 4>> asset_bins;
    std::vector<std::vector<uint32_t>> categories;
    json provenance = json::array();
    uint32_t states{}, architecture{};
};
Dataset dataset(const fs::path& directory) {
    Dataset out;
    auto index = read_json(directory / "index.json");
    std::vector<fs::path> paths{directory};
    if (index.contains("datasets")) {
        paths.clear();
        for (auto& p : index.at("datasets")) {
            fs::path relative = p.get<std::string>();
            if (relative.is_absolute() || relative.string().find("..") != std::string::npos)
                throw std::invalid_argument("invalid dataset path");
            paths.push_back(directory / relative);
        }
    }
    std::vector<std::string> names, asset_names, asset_categories;
    for (auto& path : paths) {
        auto j = read_json(path / "index.json");
        if (!j.at("complete").get<bool>() ||
            (j.at("schema") != action_schema &&
             !is_placement_schema(j.at("schema").get<uint32_t>())) ||
            j.at("path") != "actions.bin" || j.at("sha256") != file_sha256(path / "actions.bin") ||
            j.at("contract_sha256") != file_sha256(path / "contract.json"))
            throw std::invalid_argument("incomplete or changed action dataset");
        auto data = load_actions(path / "actions.bin");
        if (j.at("schema") != data.architecture ||
            (out.architecture && out.architecture != data.architecture))
            throw std::invalid_argument("mixed action policy datasets");
        out.architecture = data.architecture;
        auto width = policy_inputs(data.architecture);
        if (!data.states())
            throw std::invalid_argument("action dataset has no states");
        out.provenance.push_back({{"asset", j.at("asset")},
                                  {"sha256", j.at("sha256")},
                                  {"contract_sha256", j.at("contract_sha256")}});
        if (uint64_t(out.states) + data.states() >
            (8ull << 30) / (action_pool * width * sizeof(float)))
            throw std::length_error("resident action dataset exceeds 8 GiB");
        const auto category = j.at("category").get<std::string>();
        auto found = std::find(names.begin(), names.end(), category);
        if (found == names.end()) {
            names.push_back(category);
            out.categories.emplace_back();
            found = std::prev(names.end());
        }
        auto asset = j.at("asset").get<std::string>();
        auto asset_it = std::find(asset_names.begin(), asset_names.end(), asset);
        size_t asset_id = size_t(asset_it - asset_names.begin());
        if (asset_it == asset_names.end()) {
            asset_names.push_back(asset);
            asset_categories.push_back(category);
            out.categories[size_t(found - names.begin())].push_back(uint32_t(asset_id));
            out.asset_bins.emplace_back();
        } else if (asset_categories[asset_id] != category)
            throw std::invalid_argument("asset appears in multiple categories");
        for (uint32_t s = 0; s < data.states(); ++s) {
            auto count = data.offsets[s + 1] - data.offsets[s];
            out.asset_bins[asset_id][std::min(3u, uint32_t(data.progress[s] * 4))].push_back(
                out.states++);
            out.x.insert(out.x.end(), data.x.begin() + size_t(data.offsets[s]) * width,
                         data.x.begin() + size_t(data.offsets[s + 1]) * width);
            out.x.resize(size_t(out.states) * action_pool * width);
            out.labels.insert(out.labels.end(), data.labels.begin() + data.offsets[s],
                              data.labels.begin() + data.offsets[s + 1]);
            out.labels.resize(size_t(out.states) * action_pool);
            if (is_placement_schema(data.architecture)) {
                out.targets.insert(out.targets.end(),
                                   data.targets.begin() + size_t(data.offsets[s]) * 9,
                                   data.targets.begin() + size_t(data.offsets[s + 1]) * 9);
                out.targets.resize(size_t(out.states) * action_pool * 9);
            }
        }
    }
    if (!out.states)
        throw std::invalid_argument("action dataset collection is empty");
    if (out.x.size() * sizeof(float) > 8ull * 1024 * 1024 * 1024)
        throw std::length_error("resident action dataset exceeds 8 GiB");
    return out;
}
void check_contracts(const fs::path& output = {}) {
    torch::manual_seed(771);
    torch::Device device(torch::kCUDA);
    ActionNetwork model;
    model->to(device);
    auto input = torch::randn({35, action_features}, torch::TensorOptions().device(device));
    auto expected = model->forward(input);
    auto w = export_actions(model, {{"test", true}});
    ActionCuda native(w, {}, 7);
    auto host = input.cpu().contiguous();
    std::span<const float> x{host.data_ptr<float>(), size_t(host.numel())};
    auto result = native.predict(x);
    std::vector<double> actual(result.begin(), result.end());
    auto d = numeric_difference(doubles(expected), actual);
    if (!legacy_numeric_pass(d) ||
        !legacy_numeric_pass(numeric_difference(actual, action_oracle(x, w))))
        throw std::runtime_error("action native/FP64 parity failed");
    auto prediction =
        torch::zeros({2, 4, 3}, torch::TensorOptions().device(device).requires_grad(true));
    auto labels =
             torch::tensor({15, 11, 8, 0, 15, 15, 11, 0}, torch::kUInt8).view({2, 4}).to(device),
         valid = labels.bitwise_and(Queried).ne(0);
    auto loss = action_loss(prediction, labels, valid);
    loss.backward();
    auto grad = prediction.grad().cpu();
    if (grad[0][0][0].item<float>() >= 0 || grad[0][1][0].item<float>() <= 0 ||
        grad.select(1, 3).abs().sum().item<float>() != 0)
        throw std::runtime_error("action preferred/unknown loss contract");
    auto permutation = torch::tensor({1, 0, 2, 3}, torch::kInt64).to(device);
    if (std::abs(action_loss(prediction.index_select(1, permutation),
                             labels.index_select(1, permutation),
                             valid.index_select(1, permutation))
                     .item<double>() -
                 loss.item<double>()) > 1e-6)
        throw std::runtime_error("action loss depends on candidate ordering");
    if (!output.empty())
        save_weights(output, w);
    std::cout << json({{"training_started", false},
                       {"native_difference", difference_json(d)},
                       {"loss_contracts", true}})
                     .dump()
              << '\n';
}
int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--check-update") {
            ieee_fp32();
            torch::set_num_threads(4);
            if (!torch::cuda::is_available())
                return 77;
            update_contracts();
            return 0;
        }
        ieee_fp32();
        torch::set_num_threads(4);
        if ((argc == 2 || argc == 3) && std::string_view(argv[1]) == "--check") {
            if (!torch::cuda::is_available())
                return 77;
            check_contracts(argc == 3 ? fs::path(argv[2]) : fs::path{});
            return 0;
        }
        if (argc == 4 && std::string_view(argv[1]) == "--replay") {
            fs::path bundle = argv[2];
            auto w = load_weights(bundle / "model.blzn");
            torch::Tensor input, expected;
            torch::load(input, (bundle / "input.pt").string());
            torch::load(expected, (bundle / "expected.pt").string());
            input = input.to(torch::kCPU, torch::kFloat32).contiguous();
            if (input.numel() > 32768 * policy_inputs(w.architecture) || input.dim() != 2 ||
                input.size(1) != policy_inputs(w.architecture))
                throw std::invalid_argument("invalid replay tensor shape/cap");
            std::span<const float> features{input.data_ptr<float>(), size_t(input.numel())};
            ActionCuda native(w, {});
            auto values = native.predict(features);
            std::vector<double> actual(values.begin(), values.end());
            auto difference = numeric_difference(doubles(expected), actual),
                 oracle = numeric_difference(actual, action_oracle(features, w));
            auto passed = legacy_numeric_pass(difference) && legacy_numeric_pass(oracle);
            json report = {{"training_started", false},
                           {"model_sha256", file_sha256(bundle / "model.blzn")},
                           {"input_sha256", file_sha256(bundle / "input.pt")},
                           {"expected_sha256", file_sha256(bundle / "expected.pt")},
                           {"native", difference_json(difference)},
                           {"fp64", difference_json(oracle)},
                           {"passed", passed}};
            write_json(argv[3], report);
            std::cout << report.dump(2) << '\n';
            return passed ? 0 : 1;
        }
        if (argc < 3)
            throw std::invalid_argument("blitz-neural-action-train DATASET RUN [--steps N] "
                                        "[--minutes N] [--batch N] [--seed N]");
        uint64_t steps = 10000, seed = 0xB1172026;
        uint32_t batch = 512;
        double minutes = 5;
        std::string backend = "captured";
        fs::path initialize, warmstart;
        for (int i = 3; i < argc; i += 2) {
            if (i + 1 == argc)
                throw std::invalid_argument("missing action trainer option");
            std::string k = argv[i];
            if (k == "--steps")
                steps = std::stoull(argv[i + 1]);
            else if (k == "--minutes")
                minutes = std::stod(argv[i + 1]);
            else if (k == "--batch")
                batch = uint32_t(std::stoul(argv[i + 1]));
            else if (k == "--seed")
                seed = std::stoull(argv[i + 1]);
            else if (k == "--backend")
                backend = argv[i + 1];
            else if (k == "--initialize")
                initialize = argv[i + 1];
            else if (k == "--warmstart")
                warmstart = argv[i + 1];
            else
                throw std::invalid_argument("unknown action trainer option");
        }
        if (!initialize.empty() && !warmstart.empty())
            throw std::invalid_argument("choose model initialization or optimizer continuation");
        if (backend != "captured" && backend != "eager")
            throw std::invalid_argument("update backend must be captured or eager");
        if (!steps || steps > 1000000 || !batch || batch > 4096 || !std::isfinite(minutes) ||
            minutes <= 0 || minutes > 50 || seed > UINT32_MAX)
            throw std::invalid_argument("action trainer bounds");
        if (!torch::cuda::is_available())
            throw NeuralUnavailable("action training requires CUDA");
        std::signal(SIGTERM, stop);
        std::signal(SIGINT, stop);
        torch::manual_seed(seed);
        torch::Device device(torch::kCUDA);
        size_t free_bytes = 0, total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess)
            throw std::runtime_error("CUDA memory query failed");
        c10::cuda::CUDACachingAllocator::setMemoryFraction(
            std::min(.5, double(12ull << 30) / total_bytes), 0);
        fs::path directory = argv[1], run = argv[2];
        fs::create_directories(run);
        auto data = dataset(directory);
        auto width = policy_inputs(data.architecture), outputs = policy_outputs(data.architecture);
        json contract = {{"schema", data.architecture},
                         {"initialize_sha256", initialize.empty() ? "" : file_sha256(initialize)},
                         {"warmstart_sha256", warmstart.empty() ? "" : file_sha256(warmstart)},
                         {"optimizer_schema", update_checkpoint_version},
                         {"sampler", sampler_version},
                         {"backend", backend},
                         {"data", data.provenance},
                         {"seed", seed},
                         {"batch", batch},
                         {"precision", "IEEE FP32"},
                         {"margin", 1},
                         {"auxiliary", .25},
                         {"logit_penalty", 1e-4},
                         {"lr", .001},
                         {"weight_decay", .0001},
                         {"libtorch", TORCH_VERSION},
                         {"binary_sha256", file_sha256("/proc/self/exe")}};
        if (fs::exists(run / "contract.json") && read_json(run / "contract.json") != contract)
            throw std::invalid_argument("action training contract changed");
        write_json(run / "contract.json", contract);
        auto packed = pack_actions(data.x, data.labels, data.states, action_pool, width,
                                   action_condition_width(data.architecture));
        auto x = torch::from_blob(packed.values.data(),
                                  {data.states, action_pool, packed_width(width)}, torch::kFloat32)
                     .to(device);
        auto flags =
                 torch::from_blob(packed.flags.data(), {data.states, action_pool}, torch::kInt32)
                     .to(device),
             conditions = torch::from_blob(packed.conditions.data(),
                                           {data.states, action_condition_width(data.architecture)},
                                           torch::kFloat32)
                              .to(device);
        packed = {};
        data.x = {};
        auto labels =
            torch::from_blob(data.labels.data(), {data.states, action_pool}, torch::kUInt8)
                .to(device);
        torch::Tensor targets;
        if (is_placement_schema(data.architecture))
            targets = torch::from_blob(data.targets.data(), {data.states, action_pool, 9},
                                       torch::kFloat32)
                          .clone()
                          .to(device);
        ActionNetwork model(data.architecture);
        model->to(device);
        if (!initialize.empty()) {
            auto w = load_weights(initialize);
            if (w.architecture != data.architecture || w.hidden_width != model->hidden_width)
                throw std::invalid_argument("initial policy architecture or width differs; use the "
                                            "resident cycle for wider policies");
            torch::NoGradGuard guard;
            size_t at = 0;
            for (auto& parameter : model->parameters()) {
                parameter.copy_(
                    torch::from_blob(w.values.data() + at, parameter.sizes(), torch::kFloat32));
                at += size_t(parameter.numel());
            }
        }
        SamplingTables tables(data.categories, data.asset_bins, data.states);
        ActionUpdate update(model, x, labels, tables, batch, uint32_t(seed), {}, targets, flags,
                            conditions);
        auto& optimizer = update.optimizer;
        uint64_t step = 0;
        if (!warmstart.empty())
            step = load_state(warmstart, model, optimizer, device);
        if (fs::exists(run / "latest.json")) {
            auto latest = read_json(run / "latest.json");
            auto file = run / latest.at("checkpoint").get<std::string>();
            if (latest.at("checkpoint_sha256") != file_sha256(file))
                throw std::invalid_argument("action checkpoint changed");
            step = load_state(file, model, optimizer, device);
            if (step != latest.at("step").get<uint64_t>())
                throw std::invalid_argument("action checkpoint step mismatch");
        }
        optimizer.begin_segment();
        auto setup = std::chrono::steady_clock::now();
        if (backend == "captured")
            update.capture();
        double capture_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - setup).count();
        auto start = std::chrono::steady_clock::now();
        auto elapsed = [&] {
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        };
        auto initial = model->parameters().front().detach().clone();
        uint64_t began = step;
        double last_loss = 0, first_loss = 0, gradient = 0, update_seconds = 0,
               checkpoint_seconds = 0;
        std::ofstream log(run / "metrics.jsonl", std::ios::app);
        auto checkpoint = [&] {
            auto checkpoint_start = elapsed();
            torch::NoGradGuard guard;
            auto name = "step-" + std::to_string(step);
            auto bundle = run / "forensic" / name;
            fs::create_directories(bundle);
            auto w = export_actions(model, {{"schema", data.architecture},
                                            {"step", step},
                                            {"contract", file_sha256(run / "contract.json")}});
            save_state(bundle / "checkpoint.pt", model, optimizer, step);
            save_weights(bundle / "model.blzn", w);
            auto probe_count = std::min<uint32_t>(data.states, 32);
            std::array<int64_t, 32> probe{};
            for (uint32_t i = 0; i < probe_count; ++i)
                probe[i] = uint64_t(i) * (data.states - 1) / std::max(1u, probe_count - 1);
            auto probe_ids =
                torch::from_blob(probe.data(), {probe_count}, torch::kInt64).clone().to(device);
            auto input = update.inputs(0, probe_count, probe_ids).flatten(0, 1);
            torch::save(input.cpu(), bundle / "input.pt");
            auto expected = model->forward(input);
            torch::save(expected.cpu(), bundle / "expected.pt");
            auto cpu = input.cpu().contiguous();
            std::span<const float> features{cpu.data_ptr<float>(), size_t(cpu.numel())};
            ActionCuda native(w, {});
            auto predicted = native.predict(features);
            std::vector<double> actual(predicted.begin(), predicted.end());
            torch::save(
                torch::from_blob(predicted.data(), {input.size(0), outputs}, torch::kFloat32)
                    .clone(),
                bundle / "native.pt");
            auto difference = numeric_difference(doubles(expected), actual),
                 oracle = numeric_difference(actual, action_oracle(features, w));
            write_json(bundle / "verification.json", {{"native", difference_json(difference)},
                                                      {"fp64", difference_json(oracle)},
                                                      {"threshold", 2e-4}});
            if (!legacy_numeric_pass(difference) || !legacy_numeric_pass(oracle))
                throw std::runtime_error("action export parity failed; forensic bundle retained");
            ActionNetwork restored(data.architecture);
            restored->to(device);
            DeviceAdam restored_optimizer(restored->parameters());
            if (load_state(bundle / "checkpoint.pt", restored, restored_optimizer, device) != step)
                throw std::runtime_error("action restore step mismatch");
            for (size_t i = 0; i < model->parameters().size(); ++i) {
                auto a = model->parameters()[i], b = restored->parameters()[i];
                if (!torch::equal(a, b))
                    throw std::runtime_error("action restore parameters mismatch");
                if (!torch::equal(optimizer.control, restored_optimizer.control) ||
                    !torch::equal(optimizer.mean[i], restored_optimizer.mean[i]) ||
                    !torch::equal(optimizer.variance[i], restored_optimizer.variance[i]))
                    throw std::runtime_error("action restore optimizer mismatch");
            }
            double changed = (initial - model->parameters().front()).abs().max().item<double>();
            if (!std::isfinite(changed) || changed <= 0)
                throw std::runtime_error("action parameters did not update");
            double correct = 0, eligible = 0;
            for (uint32_t s = 0; s < data.states; s += 1024) {
                auto end = std::min(data.states, s + 1024);
                auto y = labels.slice(0, s, end), mask = data.architecture == action_schema
                                                             ? y.bitwise_and(Queried).ne(0)
                                                             : y.bitwise_and(24).eq(24);
                double count =
                    y.bitwise_and(Preferred).ne(0).logical_and(mask).any(1).sum().item<double>();
                correct +=
                    action_membership(model->forward(update.inputs(s, end - s)), y, mask) * count;
                eligible += count;
            }
            auto path = run / (name + ".pt");
            fs::rename(bundle / "checkpoint.pt", path);
            fs::rename(bundle / "model.blzn", run / (name + ".blzn"));
            // Keep the latest successful probe for cross-device validation. Failed
            // checkpoints retain their original forensic directory independently.
            fs::copy_file(run / (name + ".blzn"), bundle / "model.blzn");
            fs::remove_all(run / "replay");
            fs::rename(bundle, run / "replay");
            json health = {{"step", step},
                           {"checkpoint", name + ".pt"},
                           {"checkpoint_sha256", file_sha256(path)},
                           {"model", name + ".blzn"},
                           {"model_sha256", file_sha256(run / (name + ".blzn"))},
                           {"native_max_abs", difference.maximum},
                           {"fp64_max_abs", oracle.maximum},
                           {"restored", true},
                           {"optimizer_restored", true},
                           {"finite", true},
                           {"parameter_change", changed},
                           {"preferred_membership", eligible ? correct / eligible : 0},
                           {"preferred_states", eligible},
                           {"states", data.states},
                           {"first_loss", first_loss},
                           {"last_loss", last_loss},
                           {"gradient_norm", gradient},
                           {"seconds", elapsed()},
                           {"steps_this_segment", step - began},
                           {"complete", step >= steps},
                           {"proof_passed", false}};
            checkpoint_seconds += elapsed() - checkpoint_start;
            health["backend"] = backend;
            health["sampler"] = sampler_version;
            health["capture_seconds"] = capture_seconds;
            health["update_seconds"] = update_seconds;
            health["checkpoint_seconds"] = checkpoint_seconds;
            health["updates_per_second"] = update_seconds > 0 ? (step - began) / update_seconds : 0;
            health["resident_data_bytes"] = update.resident_bytes();
            health["architecture"] = data.architecture;
            write_json(run / "latest.json", health);
            std::cout << health.dump() << std::endl;
        };
        while (step < steps && elapsed() < minutes * 60 && !stopped) {
            auto count = uint32_t(std::min<uint64_t>(128 - step % 128, steps - step));
            auto before = elapsed();
            update.run(count);
            auto state = optimizer.state();
            update_seconds += elapsed() - before;
            if (state.failure) {
                auto bundle = run / "forensic" / ("update-" + std::to_string(state.step));
                fs::create_directories(bundle);
                auto reason =
                    state.failure == 1 ? "nonfinite action loss" : "nonfinite action gradient";
                write_json(bundle / "failure.json", {{"complete", false},
                                                     {"reason", reason},
                                                     {"step_before_update", state.step},
                                                     {"sampler", sampler_version}});
                save_state(bundle / "checkpoint.pt", model, optimizer, state.step);
                torch::save(update.ids.cpu(), bundle / "sample_ids.pt");
                torch::save(update.input.flatten(0, 1).cpu(), bundle / "input.pt");
                torch::save(update.prediction.flatten(0, 1).cpu(), bundle / "expected.pt");
                torch::save(update.target.cpu(), bundle / "labels.pt");
                if (is_placement_schema(data.architecture)) {
                    torch::save(update.target.bitwise_and(SourceKnown).ne(0).cpu(),
                                bundle / "source_known.pt");
                    torch::save(update.target.bitwise_and(AdjacentKnown).ne(0).cpu(),
                                bundle / "adjacent_known.pt");
                } else
                    torch::save(update.target.bitwise_and(Queried).ne(0).cpu(),
                                bundle / "valid.pt");
                save_weights(bundle / "model.blzn",
                             export_actions(model, {{"failure", reason}, {"step", state.step}}));
                write_json(bundle / "failure.json", {{"complete", true},
                                                     {"reason", reason},
                                                     {"step_before_update", state.step},
                                                     {"sampler", sampler_version}});
                throw std::runtime_error(std::string(reason) + "; forensic bundle retained");
            }
            if (state.step != step + count)
                throw std::runtime_error("device update counter mismatch");
            step = state.step;
            last_loss = state.loss;
            first_loss = state.first_loss;
            gradient = state.gradient;
            log << json({{"step", step},
                         {"loss", last_loss},
                         {"gradient", gradient},
                         {"seconds", elapsed()},
                         {"update_seconds", update_seconds},
                         {"backend", backend}})
                       .dump()
                << '\n';
            log.flush();
            if (step % 512 == 0)
                checkpoint();
        }
        if (step > began && step % 512)
            checkpoint();
        return step >= steps ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "action training: " << e.what() << '\n';
        return 1;
    }
}
