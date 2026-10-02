#pragma once
#include "training/workers.hpp"
#include <set>
namespace blitz::neural::training {
inline uint32_t benchmark_uint(const json& j, const char* key, uint32_t low, uint32_t high) {
    const auto& value = j.at(key);
    if (!value.is_number_integer())
        throw std::invalid_argument(std::string("integer benchmark field required: ") + key);
    auto n = value.get<uint64_t>();
    if (n < low || n > high)
        throw std::invalid_argument(std::string("benchmark field outside bounds: ") + key);
    return uint32_t(n);
}
// This uses the production resident workers without constructing an optimizer
// or LibTorch model. Every job has fresh artifacts; recovery is forbidden.
inline int prepare_teacher_jobs(const fs::path& plan_path, const fs::path& output,
                                const std::function<bool()>& interrupted) {
    if (fs::exists(output))
        throw std::invalid_argument("choose a fresh resident teacher directory");
    fs::create_directories(output);
    const auto start = std::chrono::steady_clock::now();
    auto elapsed = [&] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    };
    json report = {{"version", 1},           {"complete", false}, {"training_started", false},
                   {"optimizer_updates", 0}, {"score", nullptr},  {"jobs", json::array()},
                   {"waves", json::array()}};
    try {
        if (!allow_debugger_attach())
            throw std::runtime_error("explicit debugger attach request was rejected");
        auto plan = read_json(plan_path);
        const std::set<std::string> fields = {"version",
                                              "architecture",
                                              "hidden_width",
                                              "profile",
                                              "mask_only_coverage",
                                              "teacher_selection",
                                              "teacher_strategy",
                                              "pool",
                                              "source_limit",
                                              "adjacent_limit",
                                              "seed",
                                              "raster_backend",
                                              "vertex_storage",
                                              "candidate_batch",
                                              "workers",
                                              "gpu_memory_mib",
                                              "host_cache_mib",
                                              "measured_passes",
                                              "max_seconds",
                                              "job_seconds",
                                              "model",
                                              "model_sha256",
                                              "corpus",
                                              "training_selection",
                                              "conditions"};
        for (const auto& [key, value] : plan.items())
            if (!fields.contains(key))
                throw std::invalid_argument("unknown resident benchmark field: " + key);
        if (benchmark_uint(plan, "version", 1, 1) != 1 ||
            benchmark_uint(plan, "architecture", 4, 4) != conditioned_placement_schema ||
            benchmark_uint(plan, "pool", 4, 4) != 4 || plan.at("profile") != "coverage" ||
            plan.at("mask_only_coverage") != true ||
            plan.at("teacher_selection") != "policy-mixed" || plan.at("source_limit") != 3 ||
            plan.at("adjacent_limit") != 2 || plan.at("raster_backend") != "vulkan" ||
            plan.at("vertex_storage") != "packed")
            throw std::invalid_argument(
                "resident benchmark requires the frozen v4 teacher contract");
        auto count = benchmark_uint(plan, "workers", 1, 2);
        const auto strategy =
            teacher_strategy_option(plan.value("teacher_strategy", std::string("exhaustive")));
        auto passes = benchmark_uint(plan, "measured_passes", 1, 8);
        auto maximum = benchmark_uint(plan, "max_seconds", 1, 1800);
        auto job_seconds = benchmark_uint(plan, "job_seconds", 1, 180);
        auto host_mib = benchmark_uint(plan, "host_cache_mib", 128, 8192);
        auto seed = benchmark_uint(plan, "seed", 0, UINT32_MAX);
        auto width = benchmark_uint(plan, "hidden_width", 64, 256);
        NeuralOptions options;
        options.memory_mib = benchmark_uint(plan, "gpu_memory_mib", 128, 65536);
        options.candidate_batch = uint8_t(benchmark_uint(plan, "candidate_batch", 1, 8));
        if (options.candidate_batch != 1 && options.candidate_batch != 2 &&
            options.candidate_batch != 4 && options.candidate_batch != 8)
            throw std::invalid_argument("candidate batch must be 1, 2, 4 or 8");
        options.raster_backend = NeuralRasterBackend::Vulkan;
        options.vertex_storage = NeuralVertexStorage::Packed;
        options.mask_only_coverage = true;
        const auto& conditions = plan.at("conditions");
        if (!conditions.is_array() || conditions.empty() || conditions.size() > 64)
            throw std::invalid_argument("resident benchmark condition count outside 1..64");
        const fs::path model = plan.at("model").get<std::string>(),
                       corpus = plan.at("corpus").get<std::string>(),
                       selection = plan.at("training_selection").get<std::string>();
        const auto model_hash = file_sha256(model);
        if (model_hash != plan.at("model_sha256").get<std::string>())
            throw std::invalid_argument("benchmark model checksum changed");
        auto weights = load_weights(model);
        if (weights.architecture != conditioned_placement_schema || weights.hidden_width != width)
            throw std::invalid_argument("benchmark policy schema/width mismatch");
        const auto policy_hash = sha256(std::as_bytes(std::span(weights.values)));
        auto cancelled = [&] { return interrupted() || elapsed() >= maximum; };
        std::vector<PlacementRequest> requests;
        std::vector<std::string> assets;
        std::set<std::string> names;
        for (const auto& c : conditions) {
            const std::set<std::string> fields = {"name",
                                                  "asset",
                                                  "pixels",
                                                  "previous_steps",
                                                  "previous_pixels",
                                                  "preserve_uv",
                                                  "policy_rollout_trials",
                                                  "simplifier",
                                                  "retained",
                                                  "states"};
            for (const auto& [key, value] : c.items())
                if (!fields.contains(key))
                    throw std::invalid_argument("unknown benchmark condition field: " + key);
            auto name = c.at("name").get<std::string>();
            if (name.empty() || !names.insert(name).second)
                throw std::invalid_argument("duplicate/empty benchmark condition name");
            assets.push_back(c.at("asset").get<std::string>());
            PlacementRequest r;
            r.strategy = strategy;
            r.architecture = conditioned_placement_schema;
            r.states = benchmark_uint(c, "states", 1, 64);
            r.pool = 4;
            r.seed = seed;
            r.pixels = benchmark_uint(c, "pixels", 16, 512);
            r.previous_steps = benchmark_uint(c, "previous_steps", 0, 64);
            r.previous_pixels = benchmark_uint(c, "previous_pixels", 0, 512);
            r.preserve_uv = c.at("preserve_uv").get<bool>();
            r.policy_candidates = true;
            r.policy_rollout_trials = benchmark_uint(c, "policy_rollout_trials", 0, 64);
            r.simplifier = c.at("simplifier").get<bool>();
            r.retained = c.at("retained").get<double>();
            if ((r.policy_rollout_trials && r.simplifier) || !std::isfinite(r.retained) ||
                r.retained <= 0 || r.retained > 1 ||
                (r.previous_pixels && (!r.previous_steps || r.previous_pixels < r.pixels)))
                throw std::invalid_argument("invalid benchmark episode condition");
            r.source_limit = 3;
            r.adjacent_limit = 2;
            r.profile = Profile::Coverage;
            r.sparse = true;
            r.compact_data = true;
            r.policy_sha256 = policy_hash;
            r.minutes = job_seconds / 60.;
            r.cancelled = cancelled;
            r.corpus = corpus;
            r.selection = selection;
            requests.push_back(std::move(r));
        }
        report["request"] = plan;
        report["request_sha256"] = file_sha256(plan_path);
        report["binary_sha256"] = file_sha256("/proc/self/exe");
        report["model_sha256"] = model_hash;
        report["policy_payload_sha256"] = policy_hash;
        report["telemetry_clock"] = "nanoseconds since resident worker pool construction";
        report["warmup_jobs"] = count * 2;
        report["measured_jobs"] = passes * conditions.size();
        report["wave_jobs"] = count * 2;
        report["host_cache_mib"] = host_mib;
        write_json(output / "report.json", report);
        TrainingCache cache(corpus, selection, output / "mesh-cache", size_t(host_mib) << 20);
        for (const auto& asset : assets) {
            if (cancelled())
                throw std::runtime_error("benchmark deadline during mesh preloading");
            (void)cache.get(asset);
        }
        report["cache_after_preload"] = cache.stats();
        {
            gpu::StreamScope stream;
            MemoryScope memory(options);
            gpu::Device device(options);
            gpu::Buffer<float> frozen(device, weights.values.size());
            frozen.upload(weights.values);
            gpu::Event ready;
            gpu::check(cudaEventRecord(ready.value, gpu::stream()));
            TeacherWorkers workers(count, options, memory.budget, width,
                                   conditioned_placement_schema);
            workers.wait_ready();
            report["setup_seconds"] = elapsed();
            uint32_t next_id = 0;
            double warmup_seconds = 0, measured_seconds = 0;
            std::set<uint8_t> warmed;
            auto wave = [&](bool warmup, uint32_t pass, uint32_t first, uint32_t size) {
                if (cancelled())
                    throw std::runtime_error("benchmark deadline before wave");
                auto wave_start = std::chrono::steady_clock::now();
                workers.wave(frozen.p, frozen.n, ready.value, weights.use);
                const auto begin_id = next_id;
                for (uint32_t j = 0; j < size; ++j) {
                    auto condition = warmup ? 0u : first + j;
                    TeacherJob job;
                    job.id = next_id++;
                    job.asset = assets[condition];
                    job.name = (warmup ? "warmup-" : "measured-") + std::to_string(job.id);
                    job.directory = output / job.name;
                    job.request = requests[condition];
                    job.request.cache = &cache;
                    job.allow_recovery = false;
                    workers.submit(std::move(job));
                }
                for (uint32_t j = 0; j < size; ++j) {
                    auto condition = warmup ? 0u : first + j;
                    auto done = workers.take(begin_id + j);
                    auto& index = done.result.index;
                    bool available = index.value("requested_condition_available", false);
                    const auto& r = requests[condition];
                    bool seeded = !(r.simplifier || r.policy_rollout_trials) ||
                                  index.value("seed", json::object()).value("accepted", false);
                    bool healthy = done.result.complete &&
                                   index.value("reference_confirmed", false) && available &&
                                   seeded && done.result.data.states() > 0 && !done.recovered &&
                                   !cancelled();
                    json row = {{"id", done.job.id},
                                {"condition", condition},
                                {"pass", pass},
                                {"phase", warmup ? "warmup" : "measured"},
                                {"name", conditions[condition].at("name")},
                                {"directory", done.job.name},
                                {"complete", healthy},
                                {"worker", done.worker},
                                {"wave", done.wave},
                                {"recovered", done.recovered},
                                {"submitted_ns", done.job.submitted_ns},
                                {"started_ns", done.started_ns},
                                {"completed_ns", done.completed_ns},
                                {"consumed_ns", done.consumed_ns},
                                {"seconds", done.seconds},
                                {"stream_span_seconds", done.device_seconds},
                                {"index", index}};
                    row["payload_sha256"] = file_sha256(done.job.directory / "actions.bin");
                    if (fs::exists(done.job.directory / "episode.bin"))
                        row["episode_sha256"] = file_sha256(done.job.directory / "episode.bin");
                    report["jobs"].push_back(std::move(row));
                    if (warmup)
                        warmed.insert(done.worker);
                    if (!healthy) {
                        write_json(output / "report.json", report);
                        throw std::runtime_error(
                            "incomplete or unavailable benchmark condition: " +
                            conditions[condition].at("name").get<std::string>());
                    }
                }
                double duration =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - wave_start)
                        .count();
                (warmup ? warmup_seconds : measured_seconds) += duration;
                report["waves"].push_back({{"phase", warmup ? "warmup" : "measured"},
                                           {"first_id", begin_id},
                                           {"jobs", size},
                                           {"seconds", duration}});
                report["warmup_seconds"] = warmup_seconds;
                report["measured_seconds"] = measured_seconds;
                write_json(output / "report.json", report);
            };
            wave(true, 0, 0, count * 2);
            if (warmed.size() != count)
                throw std::runtime_error("warmup did not exercise every resident worker");
            for (uint32_t pass = 0; pass < passes; ++pass)
                for (uint32_t first = 0; first < conditions.size(); first += count * 2)
                    wave(false, pass, first,
                         uint32_t(std::min<size_t>(count * 2, conditions.size() - first)));
            teardown_trace("benchmark.workers", &workers, "begin");
            workers.shutdown();
            teardown_trace("benchmark.workers", &workers, "end");
            teardown_trace("benchmark.stream_sync", &workers, "begin");
            gpu::check(cudaStreamSynchronize(gpu::stream()));
            teardown_trace("benchmark.stream_sync", &workers, "end");
            report["gpu_peak_bytes"] = memory.budget.peak.load();
        }
        teardown_trace("benchmark.resources", &report, "end");
        report["cache"] = cache.stats();
        report["complete"] = !cancelled();
        if (!report["complete"].get<bool>())
            report["error"] = "benchmark deadline during teardown";
    } catch (const std::exception& e) {
        report["error"] = e.what();
    }
    report["total_wall_seconds"] = elapsed();
    write_json(output / "report.json", report);
    return report["complete"].get<bool>() ? 0 : 2;
}
} // namespace blitz::neural::training
