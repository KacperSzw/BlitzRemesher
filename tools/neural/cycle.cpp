#include "tools/neural/audit_io.hpp"
#include "training/checkpoint.hpp"
#include "training/curriculum.hpp"
#include "training/cycle_history.hpp"
#include "training/cycle_time.hpp"
#include "training/resident.hpp"
#include "training/workers.hpp"
#include <c10/cuda/CUDACachingAllocator.h>
#include <csignal>
using namespace blitz;
using namespace blitz::neural;
using namespace blitz::neural::training;
static volatile std::sig_atomic_t stopped = 0;
static void stop(int) {
    stopped = 1;
}
static double seconds(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}
int main(int argc, char** argv) {
    try {
        if (!allow_debugger_attach())
            throw std::runtime_error("explicit debugger attach request was rejected");
        ieee_fp32();
        torch::set_num_threads(1);
        if (!torch::cuda::is_available())
            return 77;
        if (argc < 2)
            throw std::invalid_argument(
                "blitz-neural-cycle RUN [--passes N] [--states N] [--updates N] [--batch N] "
                "[--seed N] [--minutes N] [--initialize MODEL] [--warmstart CHECKPOINT] "
                "[--raster-backend cuda|vulkan] [--vertex-storage fp32|position16|packed] "
                "[--data-storage compact|fp32] [--dataset-mib N] [--curriculum JSON] [--corpus "
                "JSON] [--training-selection JSON] [--mesh-cache DIR] [--checkpoint-scratch DIR] "
                "[--view-batch 1..4] [--direct-targets on|off] [--candidate-batch 1|2|4|8] "
                "[--update-backend reference|fused] [--duration-minutes 1..120] "
                "[--finalize-minutes N] [--architecture 3|4] [--teacher-selection "
                "geometric-random|policy-mixed] [--policy-rollout-trials 0..4096] "
                "[--teacher-strategy exhaustive|coverage-core-first]");
        fs::path run = argv[1], initialize, warmstart,
                 curriculum = "research/neural/resident-curriculum.json", checkpoint_scratch,
                 corpus = "research/corpus.json",
                 selection = "research/neural/training-manifest.json", mesh_cache;
        bool compact_data = true;
        uint32_t dataset_mib = 0;
        uint32_t passes = 1, states = 8, updates = 2048, batch = 512, seed = 101;
        double minutes = 5;
        bool quality = true, sparse = true;
        Profile profile = Profile::Attributes;
        NeuralOptions options;
        options.raster_backend = NeuralRasterBackend::Vulkan;
        options.memory_mib = 2048;
        options.candidate_batch = 2;
        UpdateBackend update_backend = UpdateBackend::Reference;
        uint32_t duration_minutes = 0, finalize_minutes = 10, workers = 1, hidden_width = 64,
                 checkpoint_seconds = 60, host_cache_mib = 2048, architecture = placement_schema,
                 policy_rollout_trials = 0;
        bool episode_seeds = false, simplifier_seeds = true, frozen_teacher = false,
             policy_candidates = false;
        TeacherStrategy teacher_strategy = TeacherStrategy::Exhaustive;
        for (int i = 2; i < argc; i += 2) {
            if (i + 1 == argc)
                throw std::invalid_argument("missing cycle option");
            std::string_view key = argv[i];
            auto* value = argv[i + 1];
            if (key == "--teacher-strategy") {
                teacher_strategy = teacher_strategy_option(value);
                continue;
            }
            if (key == "--architecture") {
                architecture = neural_unsigned(value);
                if (!is_placement_schema(architecture))
                    throw std::invalid_argument("cycle requires a placement architecture");
                continue;
            }
            if (key == "--policy-rollout-trials") {
                policy_rollout_trials = neural_unsigned(value);
                if (policy_rollout_trials > 4096)
                    throw std::invalid_argument("policy rollout trial cap");
                continue;
            }
            if (key == "--teacher-selection") {
                std::string_view v = value;
                if (v != "geometric-random" && v != "policy-mixed")
                    throw std::invalid_argument("invalid teacher action selection");
                policy_candidates = v == "policy-mixed";
                continue;
            }
            if (key == "--workers") {
                workers = neural_unsigned(value);
                continue;
            }
            if (key == "--hidden-width") {
                hidden_width = neural_unsigned(value);
                continue;
            }
            if (key == "--checkpoint-seconds") {
                checkpoint_seconds = neural_unsigned(value);
                continue;
            }
            if (key == "--host-cache-mib") {
                host_cache_mib = neural_unsigned(value);
                continue;
            }
            if (key == "--frozen-teacher") {
                std::string_view v = value;
                if (v != "on" && v != "off")
                    throw std::invalid_argument("frozen teacher must be on or off");
                frozen_teacher = v == "on";
                continue;
            }
            if (key == "--episode-seeds" || key == "--simplifier-seeds") {
                std::string_view v = value;
                if (v != "on" && v != "off")
                    throw std::invalid_argument("seed mode must be on or off");
                (key == "--episode-seeds" ? episode_seeds : simplifier_seeds) = v == "on";
                continue;
            }
            if (key == "--training-profile") {
                profile = training_profile_option(value);
                continue;
            }
            if (key == "--mask-only-coverage") {
                std::string_view v = value;
                if (v != "on" && v != "off")
                    throw std::invalid_argument("mask-only coverage must be on or off");
                options.mask_only_coverage = v == "on";
                continue;
            }
            if (key == "--update-backend") {
                std::string_view v = value;
                if (v != "reference" && v != "fused")
                    throw std::invalid_argument("update backend must be reference or fused");
                update_backend = v == "fused" ? UpdateBackend::Fused : UpdateBackend::Reference;
                continue;
            }
            if (key == "--duration-minutes") {
                duration_minutes = neural_unsigned(value);
                continue;
            }
            if (key == "--finalize-minutes") {
                finalize_minutes = neural_unsigned(value);
                continue;
            }
            if (key == "--candidate-batch") {
                auto n = neural_unsigned(value);
                if (n != 1 && n != 2 && n != 4 && n != 8)
                    throw std::invalid_argument("candidate batch must be 1, 2, 4 or 8");
                options.candidate_batch = uint8_t(n);
                continue;
            }
            if (key == "--passes")
                passes = neural_unsigned(value);
            else if (key == "--states")
                states = neural_unsigned(value);
            else if (key == "--updates")
                updates = neural_unsigned(value);
            else if (key == "--batch")
                batch = neural_unsigned(value);
            else if (key == "--seed")
                seed = neural_unsigned(value);
            else if (key == "--minutes")
                minutes = std::stod(value);
            else if (key == "--initialize")
                initialize = value;
            else if (key == "--warmstart")
                warmstart = value;
            else if (key == "--corpus")
                corpus = value;
            else if (key == "--training-selection")
                selection = value;
            else if (key == "--mesh-cache")
                mesh_cache = value;
            else if (key == "--checkpoint-scratch")
                checkpoint_scratch = value;
            else if (key == "--curriculum")
                curriculum = value;
            else if (key == "--dataset-mib")
                dataset_mib = neural_unsigned(value);
            else if (key == "--data-storage") {
                if (std::string_view(value) != "compact" && std::string_view(value) != "fp32")
                    throw std::invalid_argument("data storage must be compact or fp32");
                compact_data = std::string_view(value) == "compact";
            } else if (key == "--view-batch") {
                auto n = neural_unsigned(value);
                if (n < 1 || n > 4)
                    throw std::invalid_argument("view batch outside 1..4");
                options.view_batch = uint8_t(n);
            } else if (key == "--direct-targets") {
                if (std::string_view(value) != "on" && std::string_view(value) != "off")
                    throw std::invalid_argument("direct targets must be on or off");
                options.direct_targets = std::string_view(value) == "on";
            } else if (key == "--raster-backend")
                options.raster_backend = raster_option(value);
            else if (key == "--vertex-storage")
                options.vertex_storage = storage_option(value);
            else if (key == "--gpu-memory-mib")
                options.memory_mib = neural_unsigned(value);
            else if (key == "--quality") {
                if (std::string_view(value) != "on" && std::string_view(value) != "off")
                    throw std::invalid_argument("quality must be on or off");
                quality = std::string_view(value) == "on";
            } else if (key == "--audit-mode") {
                if (std::string_view(value) != "sparse" && std::string_view(value) != "exact")
                    throw std::invalid_argument("audit mode must be sparse or exact");
                sparse = std::string_view(value) == "sparse";
            } else
                throw std::invalid_argument("unknown cycle option");
        }
        if (!passes || passes > 10000 || !states || states > 4096 || !updates || updates > 100000 ||
            !batch || batch > 4096 || !std::isfinite(minutes) || minutes <= 0 || minutes > 180)
            throw std::invalid_argument("bounded cycle settings");
        if (workers < 1 || workers > 3 || !policy_weights(architecture, hidden_width) ||
            !checkpoint_seconds || checkpoint_seconds > 3600 || host_cache_mib < 128 ||
            host_cache_mib > 8192)
            throw std::invalid_argument("pipeline settings outside bounded contract");
        if (teacher_strategy == TeacherStrategy::CoverageCoreFirst &&
            (architecture != conditioned_placement_schema || profile != Profile::Coverage))
            throw std::invalid_argument("coverage-core-first requires v4 coverage teaching");
        if (duration_minutes > 120 ||
            (duration_minutes && (!finalize_minutes || finalize_minutes > 30)))
            throw std::invalid_argument("invalid learning/finalization budgets");
        if (duration_minutes)
            minutes = std::max(minutes, double(duration_minutes + finalize_minutes));
        if (profile == Profile::Coverage && !warmstart.empty())
            throw std::invalid_argument("coverage pretraining requires fresh Adam; initialize "
                                        "weights and resume only an identical cycle contract");
        if (mesh_cache.empty())
            mesh_cache = run / "mesh-cache";
        std::signal(SIGINT, stop);
        std::signal(SIGTERM, stop);
        fs::create_directories(run / "data");
        auto begin = std::chrono::steady_clock::now();
        CycleTime time;
        auto now_ms = [] {
            return std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                .count();
        };
        auto cancelled = [&] {
            return bool(stopped) || (!duration_minutes && seconds(begin) >= minutes * 60) ||
                   time.expired(now_ms());
        };
        auto work_stopped = [&] { return cancelled() || time.finishing(now_ms()); };
        const auto conditions = training_conditions(read_json(curriculum), architecture);
        json contract = {
            {"version", 9},
            {"corpus_sha256", file_sha256(corpus)},
            {"curriculum_sha256", file_sha256(curriculum)},
            {"data_storage",
             compact_data
                 ? (architecture == conditioned_placement_schema ? "compact-v2" : "compact-v1")
                 : "fp32"},
            {"dataset_mib", dataset_mib},
            {"view_batch", options.view_batch},
            {"direct_targets", options.direct_targets},
            {"states", states},
            {"updates", updates},
            {"batch", batch},
            {"seed", seed},
            {"gpu_memory_mib", options.memory_mib},
            {"raster", raster_name(options.raster_backend)},
            {"storage", storage_name(options.draw_storage())},
            {"sparse", sparse},
            {"quality", quality},
            {"initialize", initialize.empty() ? "" : file_sha256(initialize)},
            {"warmstart", warmstart.empty() ? "" : file_sha256(warmstart)},
            {"training_selection_sha256", file_sha256(selection)},
            {"diagnostic_sha256", file_sha256("research/neural/action-diagnostic.json")},
            {"quality_settings_sha256", file_sha256("research/neural/refactor-smoke.json")},
            {"binary_sha256", file_sha256("/proc/self/exe")}};
        contract["architecture"] = architecture;
        contract["policy_rollout_trials"] = policy_rollout_trials;
        contract["policy_action_candidates"] = policy_candidates;
        contract["teacher_strategy"] = teacher_strategy_name(teacher_strategy);
        contract["workers"] = workers;
        contract["queue_per_worker"] = 2;
        contract["hidden_width"] = hidden_width;
        contract["checkpoint_seconds"] = checkpoint_seconds;
        contract["host_cache_mib"] = host_cache_mib;
        contract["frozen_teacher"] = frozen_teacher;
        contract["episode_seeds"] = episode_seeds;
        contract["simplifier_seeds"] = simplifier_seeds;
        contract["training_profile"] = training_profile_name(profile);
        contract["checkpoint_kind"] =
            profile == Profile::Coverage ? "shape_pretraining" : "shading_training";
        contract["mask_only_coverage"] = options.mask_only_coverage;
        contract["exact_position_bps"] = options.exact_position_bps;
        contract["candidate_batch"] = options.candidate_batch;
        contract["duration_minutes"] = duration_minutes;
        contract["finalize_minutes"] = duration_minutes ? finalize_minutes : 0;
        contract["update_backend"] =
            update_backend == UpdateBackend::Fused ? "fp32-compensated-v2" : "reference";
        cudaDeviceProp properties{};
        cuda_check(cudaGetDeviceProperties(&properties, 0));
        int driver = 0, runtime = 0;
        cuda_check(cudaDriverGetVersion(&driver));
        cuda_check(cudaRuntimeGetVersion(&runtime));
        contract["update_device"] = {{"name", properties.name},
                                     {"major", properties.major},
                                     {"minor", properties.minor},
                                     {"driver", driver},
                                     {"runtime", runtime}};
        if (fs::exists(run / "contract.json") && read_json(run / "contract.json") != contract)
            throw std::invalid_argument("persistent cycle contract changed");
        write_json(run / "contract.json", contract);
        TrainingCache cache(corpus, selection, mesh_cache, size_t(host_cache_mib) << 20);
        for (const auto& c : conditions)
            (void)cache.get(c.asset);
        gpu::StreamScope native_stream;
        c10::cuda::CUDAStreamGuard learner(
            c10::cuda::getStreamFromExternal(native_stream.value, options.device));
        torch::manual_seed(seed);
        torch::Device device(torch::kCUDA);
        size_t free_bytes = 0, total_bytes = 0;
        cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes));
        const size_t training_bytes =
            std::min(size_t(512) << 20, (size_t(options.memory_mib) << 20) / 4);
        c10::cuda::CUDACachingAllocator::setMemoryFraction(double(training_bytes) / total_bytes, 0);
        MemoryScope memory(options);
        memory.budget.limit -= training_bytes;
        AuditSession gpu_session(options);
        ActionNetwork model(architecture, hidden_width);
        model->to(device);
        if (!initialize.empty()) {
            auto w = load_weights(initialize);
            if (w.architecture != architecture || w.hidden_width != hidden_width)
                throw std::invalid_argument("cycle requires placement policy");
            torch::NoGradGuard guard;
            size_t offset = 0;
            for (auto& p : model->parameters()) {
                p.copy_(torch::from_blob(w.values.data() + offset, p.sizes(), torch::kFloat32));
                offset += p.numel();
            }
        }
        if (!warmstart.empty())
            load_checkpoint_model(warmstart, model, device);
        auto initial_policy = run / "initial-model.blzn";
        if (!fs::exists(run / "latest.json"))
            save_weights(
                initial_policy,
                export_actions(model, {{"seed", seed}, {"warmstart", contract.at("warmstart")}}));
        const size_t dataset_limit = dataset_mib ? size_t(dataset_mib) << 20 : training_bytes / 2;
        if (dataset_limit > training_bytes / 2)
            throw std::invalid_argument("dataset budget exceeds half of training allocation");
        ResidentDataset data(device, seed, compact_data, dataset_limit);
        std::unique_ptr<ResidentUpdate> update;
        if (checkpoint_scratch.empty()) {
            auto name = fs::absolute(run).string();
            checkpoint_scratch = fs::temp_directory_path() / "blitz-checkpoints" /
                                 sha256(std::as_bytes(std::span(name))).substr(0, 20);
        }
        CheckpointWriter writer(checkpoint_scratch);
        json report = {{"quality_failed", false},
                       {"coverage_quality_failed", false},
                       {"failed_conditions_count", 0},
                       {"empty_conditions_count", 0},
                       {"resident_start", 0},
                       {"complete", false},
                       {"score", nullptr},
                       {"release_quality_proven", false},
                       {"datasets", json::array()},
                       {"next_condition", 0},
                       {"last_audit_iteration", 0},
                       {"active_training", nullptr}};
        uint32_t iteration = 0;
        fs::path checkpoint;
        if (fs::exists(run / "latest.json")) {
            report = read_json(run / "latest.json");
            if (report.contains("replay_order"))
                data.restore_order(report.at("replay_order"));
            iteration = report.at("next_condition");
            checkpoint = run / report.at("checkpoint").get<std::string>();
            if (file_sha256(checkpoint / "checkpoint.pt") !=
                report.at("checkpoint_sha256").get<std::string>())
                throw std::invalid_argument("cycle checkpoint changed");
            for (size_t shard = report.value("resident_start", size_t(0));
                 shard < report.at("datasets").size(); ++shard) {
                auto& name = report.at("datasets")[shard];
                auto dir = run / "data" / name.get<std::string>();
                auto index = read_json(dir / "index.json");
                if (!index.at("complete").get<bool>() ||
                    !index.at("reference_confirmed").get<bool>() ||
                    index.at("contract_sha256") != file_sha256(dir / "contract.json") ||
                    index.at("sha256") != file_sha256(dir / "actions.bin"))
                    throw std::invalid_argument("resident recovery shard changed");
                if (compact_data)
                    data.append(load_compact_actions(dir / "actions.bin"), index.at("asset"),
                                index.at("category"));
                else
                    data.append(load_actions(dir / "actions.bin"), index.at("asset"),
                                index.at("category"));
            }
            data.maximum_rows =
                std::max(data.maximum_rows, report.value("replay_maximum_rows", 1u));
            update = std::make_unique<ResidentUpdate>(model, data, batch, UpdateSettings{},
                                                      update_backend);
            load_state(checkpoint / "checkpoint.pt", model, update->optimizer, device);
            report["complete"] = false;
        }
        CycleHistory history(run / "history.jsonl", report.value("history_bytes", uint64_t(0)));
        int64_t completed_learning = report.value("learning_elapsed_ms", int64_t(0));
        std::string policy_hash;
        double refresh_seconds = 0;
        torch::Tensor frozen_policy;
        gpu::Event policy_ready;
        TeacherWorkers teachers(workers, options, memory.budget, hidden_width, architecture);
        uint32_t dispatched_end = iteration;
        bool dispatched = false;
        CheckpointCadence cadence(now_ms(), checkpoint_seconds);
        auto dispatch = [&] {
            auto start = std::chrono::steady_clock::now();
            torch::NoGradGuard guard;
            if (!report.contains("pending_wave") || report["pending_wave"].is_null()) {
                auto end = uint32_t(std::min<uint64_t>(
                    uint64_t(iteration) + workers * 2,
                    duration_minutes ? UINT32_MAX : uint64_t(passes) * conditions.size()));
                auto name = "wave-" + std::to_string(iteration) + ".blzn";
                fs::create_directories(run / "waves");
                save_weights(run / "waves" / name,
                             frozen_teacher ? load_weights(initial_policy)
                                            : export_actions(model, {{"wave", iteration}}));
                report["pending_wave"] = {{"start", iteration},
                                          {"end", end},
                                          {"model", name},
                                          {"sha256", file_sha256(run / "waves" / name)},
                                          {"jobs", json::array()}};
                for (uint32_t id = iteration; id < end; ++id) {
                    const auto& c = conditions[id % conditions.size()];
                    auto visit = id / conditions.size(), role = (id + visit) % 4;
                    std::string parent;
                    if (episode_seeds && role == 2 && !policy_rollout_trials &&
                        report.contains("episodes") && report["episodes"].contains(c.asset))
                        parent = report["episodes"][c.asset].at("path").get<std::string>();
                    report["pending_wave"]["jobs"].push_back(
                        {{"id", id},
                         {"asset", c.asset},
                         {"pixels", c.pixels},
                         {"previous", c.previous},
                         {"previous_pixels", c.previous_pixels},
                         {"source_limit", c.source_limit},
                         {"adjacent_limit", c.adjacent_limit},
                         {"preserve_uv", c.preserve_uv},
                         {"rollout_trials", episode_seeds && role == 2 ? policy_rollout_trials : 0},
                         {"parent", parent},
                         {"simplifier", episode_seeds && simplifier_seeds && role == 3},
                         {"retained", std::array{1., .75, .5, .25}[(id / 4 + visit / 4) % 4]}});
                }
            }
            auto& wave = report["pending_wave"];
            auto path = run / "waves" / wave.at("model").get<std::string>();
            if (wave.at("sha256") != file_sha256(path))
                throw std::invalid_argument("frozen wave policy changed");
            auto w = load_weights(path);
            policy_hash = sha256(std::as_bytes(std::span(w.values)));
            frozen_policy =
                torch::from_blob(w.values.data(), {int64_t(w.values.size())}, torch::kFloat32)
                    .to(device);
            cuda_check(cudaEventRecord(policy_ready.value, learner.current_stream()));
            teachers.wave(frozen_policy.data_ptr<float>(), w.values.size(), policy_ready.value,
                          w.use);
            for (const auto& descriptor : wave.at("jobs")) {
                auto id = descriptor.at("id").get<uint32_t>();
                if (id < iteration || (id == iteration && !report["active_training"].is_null()))
                    continue;
                TeacherJob job;
                job.id = id;
                job.asset = descriptor.at("asset");
                job.name = "shard-" + std::to_string(id);
                job.directory = run / "data" / job.name;
                auto& request = job.request;
                request.architecture = architecture;
                request.strategy = teacher_strategy;
                request.policy_candidates = policy_candidates;
                request.source_limit = descriptor.at("source_limit");
                request.adjacent_limit = descriptor.at("adjacent_limit");
                request.previous_pixels = descriptor.at("previous_pixels");
                request.preserve_uv = descriptor.at("preserve_uv");
                request.policy_rollout_trials = descriptor.at("rollout_trials");
                request.corpus = corpus;
                request.selection = selection;
                request.mesh_cache = mesh_cache;
                request.cache = &cache;
                request.compact_data = compact_data;
                request.states = states;
                request.pool = 4;
                request.pixels = descriptor.at("pixels");
                request.previous_steps = descriptor.at("previous");
                request.seed = seed + id;
                request.sparse = sparse;
                request.minutes = std::min(3., std::max(.001, minutes - seconds(begin) / 60));
                request.cancelled = work_stopped;
                request.profile = profile;
                request.policy_sha256 = policy_hash;
                auto parent = descriptor.at("parent").get<std::string>();
                if (!parent.empty()) {
                    request.episode = run / parent;
                    auto index = read_json(request.episode.parent_path() / "index.json");
                    if (index.at("episode_sha256") != file_sha256(request.episode))
                        throw std::invalid_argument("episode seed changed");
                }
                request.simplifier = descriptor.at("simplifier");
                request.retained = descriptor.at("retained");
                teachers.submit(std::move(job));
            }
            dispatched_end = wave.at("end");
            dispatched = true;
            refresh_seconds += seconds(start);
        };
        if (duration_minutes)
            time = CycleTime::resume(now_ms(), duration_minutes * 60, finalize_minutes * 60,
                                     completed_learning);
        int64_t next_audit = report.value("next_audit_ms", now_ms() + 1800000);
        auto quality_audit = [&] {
            if (!quality)
                return;
            auto local = writer.local_checkpoint();
            if (local.empty())
                local = checkpoint;
            auto manifest = read_json("research/neural/action-diagnostic.json");
            auto settings = settings_json(read_json("research/neural/refactor-smoke.json"));
            settings.cancelled = cancelled;
            for (auto audit_profile :
                 profile == Profile::Coverage
                     ? std::vector<Profile>{Profile::Coverage, Profile::Attributes}
                     : std::vector<Profile>{Profile::Attributes})
                for (auto ranking : {NeuralRanking::Constant, NeuralRanking::Learned}) {
                    settings.profile = audit_profile;
                    auto failed_key = audit_profile == Profile::Coverage ? "coverage_quality_failed"
                                                                         : "quality_failed";
                    auto audit_start = std::chrono::steady_clock::now();
                    auto controls = options;
                    controls.action_trials = 8;
                    controls.action_batch = 16;
                    controls.ranking = ranking;
                    controls.ranking_seed = 101;
                    controls.capture_confirmation_failure = true;
                    NeuralModel audit_model((local / "model.blzn").c_str(), controls);
                    json rows = json::array();
                    for (auto& asset : manifest.at("assets")) {
                        for (auto& f : asset.at("files"))
                            if (file_sha256(f.at("path").get<std::string>()) !=
                                f.at("sha256").get<std::string>())
                                throw std::invalid_argument("diagnostic asset changed");
                        auto mesh = load_mesh(asset.at("path").get<std::string>());
                        NeuralStats stats;
                        json row;
                        try {
                            auto r = generate_neural(mesh.view(), settings, audit_model, &stats);
                            row = result_json(r);
                            if (r.status != Status::Complete)
                                report[failed_key] = true;
                        } catch (const std::logic_error& error) {
                            if (std::string_view(error.what()) !=
                                    "exact source fallback failed reference audit" &&
                                std::string_view(error.what()) !=
                                    "packed baseline failed reference audit")
                                throw;
                            row = {{"status", "no_confirmed_chain"}, {"error", error.what()}};
                            report[failed_key] = true;
                        }
                        if (stats.resource_failures || stats.confirmation_nonfinite ||
                            stats.confirmation_cancelled)
                            throw std::runtime_error(
                                "quality diagnostic interrupted or resource/nonfinite failure");
                        save_audit_failure(stats,
                                           run / "audit-failures" /
                                               (std::to_string(iteration) + "-" +
                                                training_profile_name(audit_profile) + "-" +
                                                ranking_name(ranking) + "-" +
                                                asset.at("id").get<std::string>() + ".json"));
                        row["asset"] = asset.at("id");
                        row["neural"] = neural_json(stats);
                        rows.push_back(row);
                    }
                    history.append("phases", {{"phase", "quality_audit"},
                                              {"profile", training_profile_name(audit_profile)},
                                              {"blocking", audit_profile == Profile::Coverage ||
                                                               profile != Profile::Coverage},
                                              {"ranking", ranking_name(ranking)},
                                              {"seconds", seconds(audit_start)},
                                              {"rows", rows}});
                }
            report["last_audit_iteration"] = iteration;
            report["last_audit_step"] = update ? update->optimizer.state().step : 0;
            next_audit = now_ms() + 1800000;
            report["next_audit_ms"] = next_audit;
        };
        std::set<std::string> durable_data;
        auto publish_checkpoint = [&] {
            Timeline range("checkpoint");
            auto s = update->optimizer.state();
            checkpoint = run / "checkpoints" / ("step-" + std::to_string(s.step));
            auto probe = update->input.slice(0, 0, std::min<int64_t>(8, batch)).flatten(0, 1);
            torch::NoGradGuard guard;
            auto expected = model->forward(probe);
            auto native_input = probe.cpu().contiguous();
            auto w = export_actions(model, {{"step", s.step}});
            ActionCuda check_model(w, options, 32);
            auto values =
                check_model.predict({native_input.data_ptr<float>(), size_t(native_input.numel())});
            auto native =
                torch::from_blob(values.data(), expected.sizes(), torch::kFloat32).clone();
            for (const auto& item : report.at("datasets")) {
                auto name = item.get<std::string>();
                if (durable_data.insert(name).second) {
                    auto dir = run / "data" / name;
                    for (const auto* file :
                         {"actions.bin", "index.json", "contract.json", "episode.bin"})
                        sync_checkpoint_file(dir / file);
                    sync_checkpoint_file(dir);
                }
            }
            sync_checkpoint_file(run / "data");
            if (report.contains("episodes"))
                for (const auto& item : report.at("episodes")) {
                    auto episode = run / item.at("path").get<std::string>();
                    sync_checkpoint_file(episode);
                    sync_checkpoint_file(episode.parent_path() / "index.json");
                    sync_checkpoint_file(episode.parent_path());
                }
            if (report.contains("pending_wave") && !report["pending_wave"].is_null()) {
                sync_checkpoint_file(run / "waves" /
                                     report["pending_wave"].at("model").get<std::string>());
                sync_checkpoint_file(run / "waves");
            }
            report["history_bytes"] = history.sync();
            if (duration_minutes)
                report["learning_elapsed_ms"] =
                    time.completed(now_ms(), completed_learning, duration_minutes * 60);
            report["checkpoint_kind"] = contract.at("checkpoint_kind");
            report["states"] = data.states();
            report["replay_order"] = data.order();
            report["replay_maximum_rows"] = data.maximum_rows;
            writer.submit(checkpoint, model, update->optimizer,
                          {{"step", s.step},
                           {"contract", file_sha256(run / "contract.json")},
                           {"training_profile", training_profile_name(profile)},
                           {"checkpoint_kind", contract.at("checkpoint_kind")}},
                          probe, expected, run / "latest.json", report, native);
            cadence.submitted(now_ms());
        };
        uint64_t initial_step = update ? update->optimizer.state().step : 0,
                 initial_fresh_states = report.value("fresh_states", uint64_t(0));
        while ((duration_minutes || iteration < uint64_t(passes) * conditions.size()) &&
               !work_stopped()) {
            if (!dispatched && update && report["active_training"].is_null() &&
                now_ms() >= next_audit) {
                publish_checkpoint();
                writer.join();
                quality_audit();
            }
            if (!dispatched)
                dispatch();
            auto phase_start = std::chrono::steady_clock::now();
            if (report["active_training"].is_null()) {
                const auto& condition = conditions[iteration % conditions.size()];
                auto waited = std::chrono::steady_clock::now();
                auto done = [&] {
                    Timeline range("wait-teacher");
                    return teachers.take(iteration);
                }();
                auto& teacher = done.result;
                auto name = done.job.name;
                auto directory = done.job.directory;
                history.append("phases", {{"phase", "teacher"},
                                          {"job", iteration},
                                          {"seconds", done.seconds},
                                          {"stream_span_seconds", done.device_seconds},
                                          {"learner_wait_seconds", seconds(waited)},
                                          {"index", teacher.index}});
                if (!teacher.complete) {
                    if (time.finishing(now_ms()) && !cancelled()) {
                        report["duration_interrupted_teacher"] = name;
                        break;
                    }
                    if (teacher.index.value("status", std::string{}) == "representation_failure") {
                        report["failed_conditions_count"] =
                            report.at("failed_conditions_count").get<uint64_t>() + 1;
                        history.append(
                            "failed_conditions",
                            {{"condition", iteration}, {"shard", name}, {"index", teacher.index}});
                        ++iteration;
                        report["next_condition"] = iteration;
                        if (iteration == dispatched_end) {
                            report["pending_wave"] = nullptr;
                            dispatched = false;
                        }
                        continue;
                    }
                    throw std::runtime_error("incomplete teacher shard retained; training skipped");
                }
                const bool available =
                    teacher.index.at("requested_condition_available").get<bool>();
                auto availability_key =
                    available ? "available_conditions_count" : "unavailable_conditions_count";
                report[availability_key] = report.value(availability_key, uint64_t(0)) + 1;
                if (!available)
                    history.append(
                        "unavailable_conditions",
                        {{"condition", iteration}, {"shard", name}, {"index", teacher.index}});
                const auto requested_states =
                    available ? teacher.data.states() -
                                    std::min<size_t>(teacher.data.states(), condition.previous)
                              : 0;
                report["requested_condition_states"] =
                    report.value("requested_condition_states", uint64_t(0)) + requested_states;
                // A valid unchanged mesh can have no legal action. Preserve this
                // outcome without publishing an empty page to the training graph.
                if (!teacher.data.states()) {
                    report["empty_conditions_count"] =
                        report.at("empty_conditions_count").get<uint64_t>() + 1;
                    history.append(
                        "empty_conditions",
                        {{"condition", iteration}, {"shard", name}, {"index", teacher.index}});
                    ++iteration;
                    report["next_condition"] = iteration;
                    if (iteration == dispatched_end) {
                        report["pending_wave"] = nullptr;
                        dispatched = false;
                    }
                    continue;
                }
                phase_start = std::chrono::steady_clock::now();
                auto before_evictions = data.evicted();
                if (compact_data)
                    data.append(teacher.compact, condition.asset, teacher.index.at("category"));
                else
                    data.append(teacher.data, condition.asset, teacher.index.at("category"));
                for (uint64_t i = before_evictions; i < data.evicted(); ++i)
                    report["datasets"].erase(report["datasets"].begin());
                report["datasets"].push_back(name);
                history.append("all_datasets", name);
                report["fresh_states"] =
                    report.value("fresh_states", uint64_t(0)) + teacher.data.states();
                if (!report["episodes"].contains(condition.asset) ||
                    teacher.index.at("teacher_triangles") <
                        report["episodes"][condition.asset].at("faces"))
                    report["episodes"][condition.asset] = {
                        {"path", fs::relative(directory / "episode.bin", run).string()},
                        {"faces", teacher.index.at("teacher_triangles")}};
                if (!update) {
                    update = std::make_unique<ResidentUpdate>(model, data, batch, UpdateSettings{},
                                                              update_backend);
                    if (!warmstart.empty()) {
                        load_state(warmstart, model, update->optimizer, device);
                        initial_step = update->optimizer.state().step;
                    }
                }
                update->optimizer.begin_segment();
                uint32_t target_step = update_target(update->optimizer.state().step, updates);
                report["active_training"] = {
                    {"target_step", target_step}, {"append_seconds", seconds(phase_start)},
                    {"capture_seconds", 0.},      {"update_seconds", 0.},
                    {"checkpoint_seconds", 0.},   {"seconds", 0.}};
            }
            auto& progress = report["active_training"];
            auto capture_start = std::chrono::steady_clock::now();
            update->capture();
            progress["capture_seconds"] =
                progress.at("capture_seconds").get<double>() + seconds(capture_start);
            uint32_t target_step = progress.at("target_step");
            while (update->optimizer.state().step < target_step && !work_stopped()) {
                Timeline range("optimizer-window");
                auto s = update->optimizer.state();
                auto count = std::min(128u, target_step - s.step);
                auto t = std::chrono::steady_clock::now();
                update->run(count);
                s = update->optimizer.state();
                progress["update_seconds"] =
                    progress.at("update_seconds").get<double>() + seconds(t);
                if (s.failure)
                    throw std::runtime_error("nonfinite resident optimizer state");
                if (cadence.due(now_ms()) && !writer.busy()) {
                    t = std::chrono::steady_clock::now();
                    progress["seconds"] =
                        progress.at("seconds").get<double>() + seconds(phase_start);
                    phase_start = std::chrono::steady_clock::now();
                    publish_checkpoint();
                    progress["checkpoint_seconds"] =
                        progress.at("checkpoint_seconds").get<double>() + seconds(t);
                }
            }
            auto s = update->optimizer.state();
            if (s.step != target_step)
                break;
            ++iteration;
            if (iteration == dispatched_end) {
                report["pending_wave"] = nullptr;
                dispatched = false;
            }
            auto training = progress;
            training["seconds"] = training.at("seconds").get<double>() + seconds(phase_start);
            training["phase"] = "training";
            training["step"] = s.step;
            training["loss"] = s.loss;
            training["gradient"] = s.gradient;
            training["pool"] = update->pool();
            training["fused_epilogues"] = update->fused_epilogues();
            training["fused_algorithms"] = update->fused_algorithms();
            training["captures"] = update->captures;
            training["resident_data_bytes"] = data.bytes();
            history.append("phases", training);
            report["active_training"] = nullptr;
            report["next_condition"] = iteration;
            report["step"] = s.step;
            if (!checkpoint.empty())
                report["checkpoint"] = fs::relative(checkpoint, run).string();
            report["states"] = data.states();
            report["seconds"] = seconds(begin);
        }
        teachers.shutdown();
        if (update && !time.expired(now_ms())) {
            auto step = update->optimizer.state().step;
            if (checkpoint.filename() != ("step-" + std::to_string(step)))
                publish_checkpoint();
            report["step"] = step;
            report["checkpoint"] = fs::relative(checkpoint, run).string();
        }
        writer.join();
        if (update && !cancelled() &&
            report.value("last_audit_step", uint32_t(UINT32_MAX)) != update->optimizer.state().step)
            quality_audit();
        bool duration_done = duration_minutes && time.finishing(now_ms());
        report["complete"] =
            bool(update) && (duration_done || iteration == uint64_t(passes) * conditions.size()) &&
            !cancelled() && report.at("failed_conditions_count").get<uint64_t>() == 0 &&
            !report["coverage_quality_failed"].get<bool>() &&
            (profile == Profile::Coverage || !report["quality_failed"].get<bool>());
        report["status"] = report["complete"].get<bool>()
                               ? (duration_done ? "duration_complete" : "work_complete")
                           : stopped                ? "cancelled"
                           : time.expired(now_ms()) ? "finalization_deadline_exceeded"
                                                    : "incomplete";
        report["curriculum_coverage_complete"] =
            report.value("unavailable_conditions_count", uint64_t(0)) == 0 &&
            report.value("failed_conditions_count", uint64_t(0)) == 0 &&
            report.value("empty_conditions_count", uint64_t(0)) == 0 &&
            report.value("available_conditions_count", uint64_t(0)) >= conditions.size();
        if (!checkpoint.empty())
            report["checkpoint_sha256"] = file_sha256(checkpoint / "checkpoint.pt");
        report["cache"] = cache.stats();
        report["replay_evicted_pages"] = data.evicted();
        report["gpu_peak_bytes"] = memory.budget.peak.load() + training_bytes;
        report["checkpoint_timing"] = writer.stats();
        report["policy_refresh_seconds"] = refresh_seconds;
        report["seconds"] = seconds(begin);
        report["fresh_states_this_invocation"] =
            report.value("fresh_states", uint64_t(0)) - initial_fresh_states;
        report["fresh_states_per_second"] =
            report["fresh_states_this_invocation"].get<uint64_t>() / seconds(begin);
        report["updates_this_invocation"] =
            update ? update->optimizer.state().step - initial_step : 0;

        report["history_bytes"] = history.sync();
        if (duration_minutes)
            report["learning_elapsed_ms"] =
                time.completed(now_ms(), completed_learning, duration_minutes * 60);
        report["training_profile"] = training_profile_name(profile);
        report["checkpoint_kind"] = contract.at("checkpoint_kind");
        if (report["complete"].get<bool>())
            write_json(run / "latest.json", report);
        history.assemble(report);
        write_json(run / "report.json", report);
        std::cout << json({{"complete", report["complete"]},
                           {"status", report["status"]},
                           {"step", report.value("step", 0)},
                           {"seconds", report["seconds"]}})
                         .dump(2)
                  << '\n';
        return report.at("complete").get<bool>() ? 0 : 2;
    } catch (const std::exception& e) {
        if (argc > 1 && !std::string_view(argv[1]).starts_with("--")) {
            fs::create_directories(argv[1]);
            write_json(fs::path(argv[1]) / "failure.json",
                       {{"complete", false}, {"error", e.what()}});
        }
        std::cerr << "resident cycle: " << e.what() << '\n';
        return 1;
    }
}
