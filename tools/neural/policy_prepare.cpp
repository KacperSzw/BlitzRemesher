#include "tools/neural/audit_settings.hpp"
#include "training/runtime_teacher.hpp"
#include <csignal>
#include <iostream>

using namespace blitz;
using namespace blitz::neural;
using namespace blitz::neural::training;
namespace {
volatile std::sig_atomic_t stopped = 0;
void stop(int) {
    stopped = 1;
}
} // namespace
int main(int argc, char** argv) {
    try {
        if (argc != 3)
            throw std::invalid_argument("blitz-neural-policy-prepare JOB_JSON FRESH_OUTPUT");
        const auto job = read_json(argv[1]);
        const fs::path output = argv[2], model = job.at("model").get<std::string>(),
                       corpus = job.at("corpus").get<std::string>(),
                       selection = job.at("training_selection").get<std::string>();
        if (fs::exists(output))
            throw std::invalid_argument("runtime teacher output exists");
        fs::create_directories(output);
        const auto metadata =
            training_metadata(job.at("asset").get<std::string>(), corpus, selection);
        auto config = read_json(job.at("settings").get<std::string>());
        config.erase("audit_rankings");
        config["preserve_uv"] = job.at("preserve_uv");
        config["gpu_memory_mib"] = job.at("gpu_memory_mib");
        NeuralOptions options;
        options.raster_backend = NeuralRasterBackend::Vulkan;
        options.ranking = ranking_option(job.value("rollout_ranking", "learned"));
        auto settings = model_audit_settings(config, options);
        const auto weights = load_weights(model);
        RuntimeTeachingSettings teaching;
        teaching.states_per_proposal = job.value("states_per_proposal", 16u);
        teaching.maximum_states = job.value("maximum_states", 256u);
        teaching.pool = job.value("pool", 16u);
        teaching.seed = job.value("seed", 101u);
        teaching.cache_rasters = job.value("teacher_cache_rasters", false);
        const double minutes = job.value("minutes", 3.);
        if (!std::isfinite(minutes) || minutes <= 0 || minutes > 50)
            throw std::invalid_argument("runtime teacher duration outside (0,50]");
        std::signal(SIGINT, stop);
        std::signal(SIGTERM, stop);
        const auto start = std::chrono::steady_clock::now();
        auto seconds = [&] {
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        };
        settings.cancelled =
            teaching.cancelled = [&] { return stopped || seconds() >= minutes * 60; };
        json contract = {
            {"teacher_version", 9},
            {"teacher_target", "runtime-endpoint-v2"},
            {"schema", weights.architecture},
            {"data_storage", "fp32"},
            {"training_profile", "coverage"},
            {"asset", metadata},
            {"job", job},
            {"visual_settings", settings_json(settings)},
            {"execution_settings", neural_json(options)},
            {"preserve_uv", options.preserve_uv},
            {"policy_payload_sha256", sha256(std::as_bytes(std::span(weights.values)))},
            {"model_sha256", file_sha256(model)},
            {"source_manifest_sha256", file_sha256(corpus)},
            {"training_selection_sha256", file_sha256(selection)},
            {"protocol_sha256", file_sha256("research/PROTOCOL.md")},
            {"binary_sha256", file_sha256("/proc/self/exe")},
            {"preference",
             "minimum triangles, then exact normalized source/adjacent search and audit error; all "
             "exact ties preferred"}};
        const auto parent = load_mesh(metadata.at("path").get<std::string>());
        auto source = parent.view();
        Mesh augmented;
        const uint32_t requested_triangles = job.value("source_triangles", 0u);
        const auto source_output = job.value("source_output", "rebuild");
        const bool merge_wedges = job.value("source_merge_wedges", false);
        if (source_output != "reuse" && source_output != "rebuild")
            throw std::invalid_argument("source_output must be reuse or rebuild");
        if (merge_wedges && source_output != "rebuild")
            throw std::invalid_argument("source wedge merging requires rebuilt output");
        const auto augmentation_start = seconds();
        if (requested_triangles) {
            if (requested_triangles < 4 || requested_triangles >= source.triangles())
                throw std::invalid_argument("augmentation must request [4,parent triangles)");
            ReduceSettings reduction;
            reduction.target_triangles = requested_triangles;
            reduction.output = source_output == "reuse" ? OutputMode::Reuse : OutputMode::Rebuild;
            reduction.coupled_wedges = true;
            reduction.merge_wedges = merge_wedges;
            reduction.cancelled = teaching.cancelled;
            const auto reduced = reduce(source, reduction);
            augmented = copy_mesh(reduced.view(source));
            compact(augmented);
            source = augmented.view();
            if (teaching.cancelled() || !source.triangles() ||
                source.triangles() > parent.view().triangles())
                throw std::runtime_error("source augmentation incomplete");
        }
        // A reduced source is a distinct training input, not an audited LOD of
        // its parent. Its exact FP32 geometry becomes this episode's LOD0.
        contract["source_augmentation"] = {
            {"algorithm", requested_triangles ? "quadric-coupled-" + source_output +
                                                    (merge_wedges ? "-merged-v1" : "-v1")
                                              : "none"},
            {"requested_triangles", requested_triangles},
            {"parent_triangles", parent.view().triangles()},
            {"actual_triangles", source.triangles()},
            {"reference", "source-reference.bin"}};
        const double augmentation_seconds = seconds() - augmentation_start;
        write_json(output / "contract.json", contract);
        auto result =
            prepare_runtime_rankings(source, settings, weights, options, teaching, output);
        if (fs::exists(output / "source-reference.bin"))
            contract["source_augmentation"]["reference_sha256"] =
                file_sha256(output / "source-reference.bin");
        write_json(output / "contract.json", contract);
        save_actions(output / "actions.bin", result.data, false);
        {
            std::ofstream stream(output / "geometry-rejected.bin", std::ios::binary);
            stream.write("BLZRANK1", 8);
            write_vector(stream, result.geometry);
            stream.close();
            if (!stream)
                throw std::runtime_error("runtime geometry bitmap write failed");
        }
        write_json(output / "requests.json", result.requests);
        write_json(output / "trajectory.json", result.observations);
        write_json(output / "verification.json", result.verification);
        auto labels = result.data.labels;
        apply_policy_rank_masks(labels, result.geometry);
        uint32_t informative = 0, pairs = 0;
        for (size_t i = 0; i < result.data.states(); ++i) {
            const auto count = ranking_pairs(std::span(labels).subspan(
                result.data.offsets[i], result.data.offsets[i + 1] - result.data.offsets[i]));
            informative += count != 0;
            pairs += count;
        }
        json index = {{"schema", weights.architecture},
                      {"complete", result.complete},
                      {"status", !result.complete       ? "incomplete_runtime_observation"
                                 : result.data.states() ? "complete"
                                                        : "complete_no_actions"},
                      {"ranking_eligible", informative != 0},
                      {"asset", job.at("asset")},
                      {"category", metadata.at("category")},
                      {"contract_sha256", file_sha256(output / "contract.json")},
                      {"path", "actions.bin"},
                      {"sha256", file_sha256(output / "actions.bin")},
                      {"geometry_rejected_sha256", file_sha256(output / "geometry-rejected.bin")},
                      {"requests_sha256", file_sha256(output / "requests.json")},
                      {"trajectory_sha256", file_sha256(output / "trajectory.json")},
                      {"verification", result.verification},
                      {"states", result.data.states()},
                      {"ranking_states", informative},
                      {"ranking_pairs", pairs},
                      {"source_triangles", source.triangles()},
                      {"source_augmentation", contract.at("source_augmentation")},
                      {"augmentation_seconds", augmentation_seconds},
                      {"teacher_stats", neural_json(result.teacher_stats)},
                      {"seconds", seconds()},
                      {"baseline_seconds", result.baseline_seconds},
                      {"rollout_seconds", result.rollout_seconds},
                      {"teacher_seconds_within_rollout", result.teacher_seconds}};
        write_json(output / "index.json", index);
        std::cout << index.dump() << '\n';
        return result.complete ? 0 : 2;
    } catch (const std::exception& error) {
        if (argc == 3 && fs::exists(argv[2]))
            write_json(fs::path(argv[2]) / "failure.json",
                       {{"complete", false}, {"error", error.what()}});
        std::cerr << error.what() << '\n';
        return 1;
    }
}
