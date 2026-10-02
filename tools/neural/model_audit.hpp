#pragma once
#include "neural/cuda.cuh"
#include "neural/generation.hpp"
#include "tools/neural/audit_settings.hpp"
#include "training/data.hpp"
#include "training/json.hpp"
#include "training/mesh_cache.hpp"
#include <set>
#include <sstream>
namespace blitz::neural::training {
inline void audit_model(const fs::path& manifest_path, const fs::path& model_path,
                        const fs::path& settings_path, const fs::path& output) {
    if (fs::exists(output))
        throw std::invalid_argument("model audit output exists");
    auto manifest = read_json(manifest_path), config = read_json(settings_path);
    const bool trace_execution = config.value("trace_execution", false);
    config.erase("trace_execution");
    const auto support_config = config.value("ranking_support", json());
    config.erase("ranking_support");
    fs::path collector_path;
    uint32_t support_seed = 101;
    if (!support_config.is_null()) {
        if (!support_config.is_object() || support_config.size() != 3 ||
            !support_config.contains("collector") || !support_config.at("collector").is_string() ||
            support_config.value("pool", 0) != 16 || !support_config.contains("seed") ||
            !support_config.at("seed").is_number_unsigned() ||
            support_config.at("seed").get<uint64_t>() > UINT32_MAX)
            throw std::invalid_argument("ranking support requires collector, pool 16 and u32 seed");
        collector_path = support_config.at("collector").get<std::string>();
        support_seed = support_config.at("seed").get<uint32_t>();
    }
    const auto preparation = config.value("source_preparation", json());
    config.erase("source_preparation");
    uint32_t source_limit = 0;
    if (!preparation.is_null()) {
        if (!preparation.is_object() || preparation.size() != 1 ||
            !preparation.contains("maximum_triangles") ||
            !preparation.at("maximum_triangles").is_number_unsigned())
            throw std::invalid_argument("source preparation requires maximum_triangles");
        const auto limit = preparation.at("maximum_triangles").get<uint64_t>();
        if (limit < 4 || limit > UINT32_MAX)
            throw std::invalid_argument("source preparation triangle limit outside [4,u32]");
        source_limit = uint32_t(limit);
    }
    std::vector<NeuralRanking> rankings{NeuralRanking::Constant, NeuralRanking::Learned};
    if (auto it = config.find("audit_rankings"); it != config.end()) {
        if (!it->is_array() || it->empty())
            throw std::invalid_argument("audit rankings must be nonempty");
        rankings.clear();
        for (const auto& name : *it) {
            auto value = ranking_option(name.get<std::string>());
            if (std::find(rankings.begin(), rankings.end(), value) != rankings.end())
                throw std::invalid_argument("duplicate audit ranking");
            rankings.push_back(value);
        }
        config.erase(it);
    }
    // These historical defaults keep existing preflight/final comparisons intact.
    NeuralOptions options;
    options.raster_backend = NeuralRasterBackend::Vulkan;
    options.memory_mib = 4096;
    options.action_trials = 8;
    options.action_batch = 16;
    auto settings = model_audit_settings(config, options);
    auto visual = settings_json(settings);
    auto execution = neural_json(options);
    execution.erase("ranking");
    execution["audit_rankings"] = json::array();
    execution["trace_execution"] = trace_execution;
    if (!collector_path.empty())
        execution["ranking_support"] = {{"collector_sha256", file_sha256(collector_path)},
                                        {"pool", 16},
                                        {"seed", support_seed},
                                        {"selection", "frozen collector PolicyMixed"}};
    if (source_limit)
        execution["source_preparation"] = {
            {"algorithm", "quadric-coupled-rebuild-merged-v1"},
            {"maximum_triangles", source_limit},
            {"reference", "derived mesh is the new LOD0; no parent fidelity claim"}};
    for (auto ranking : rankings)
        execution["audit_rankings"].push_back(ranking_name(ranking));
    auto hash = [](const json& value) {
        auto bytes = value.dump();
        return sha256(std::as_bytes(std::span(bytes)));
    };
    json assets = json::array();
    std::set<std::string> identities;
    for (const auto& asset : manifest.at("assets")) {
        auto id = asset.at("id").get<std::string>();
        if (!identities.insert(id).second)
            throw std::invalid_argument("duplicate model audit asset");
        assets.push_back({{"id", id}, {"category", asset.at("category")}});
    }
    if (assets.empty())
        throw std::invalid_argument("empty model audit manifest");
    bool complete = true;
    json report = {{"version", 2},
                   {"complete", false},
                   {"score", nullptr},
                   {"gpu_memory_mib", options.memory_mib},
                   {"model_sha256", file_sha256(model_path)},
                   {"binary_sha256", file_sha256("/proc/self/exe")},
                   {"manifest_sha256", file_sha256(manifest_path)},
                   {"settings_sha256", file_sha256(settings_path)},
                   {"visual_settings", visual},
                   {"visual_settings_sha256", hash(visual)},
                   {"execution_settings", execution},
                   {"execution_settings_sha256", hash(execution)},
                   {"expected_assets", assets},
                   {"rows", json::array()}};
    auto start = std::chrono::steady_clock::now();
    gpu::StreamScope stream;
    MemoryScope memory(options);
    AuditSession session(options);
    std::unique_ptr<ActionCuda> collector;
    if (!collector_path.empty())
        collector = std::make_unique<ActionCuda>(load_weights(collector_path), options);
    const fs::path sources_directory = output.string() + ".sources";
    if (source_limit && !fs::create_directory(sources_directory))
        throw std::invalid_argument("model audit source directory exists");
    for (const auto& asset : manifest.at("assets")) {
        if (asset.at("split") != "development" && asset.at("split") != "validation")
            throw std::invalid_argument("model comparison cannot tune on release holdout");
        for (const auto& f : asset.at("files"))
            if (f.at("sha256") != file_sha256(f.at("path").get<std::string>()))
                throw std::invalid_argument("model audit source changed");
        auto mesh = load_mesh(asset.at("path").get<std::string>());
        json prepared;
        if (source_limit) {
            const auto parent_triangles = mesh.view().triangles();
            if (parent_triangles > source_limit) {
                ReduceSettings reduction;
                reduction.target_triangles = source_limit;
                reduction.output = OutputMode::Rebuild;
                reduction.coupled_wedges = true;
                reduction.merge_wedges = true;
                const auto reduced = reduce(mesh.view(), reduction);
                mesh = copy_mesh(reduced.view(mesh.view()));
                compact(mesh);
            }
            // The prepared FP32 streams are explicit test inputs, never an
            // audited output of the parent or an implicit replacement for it.
            const auto reference =
                sources_directory /
                (std::to_string(report["rows"].size() / rankings.size()) + ".bin");
            std::ofstream stream(reference, std::ios::binary);
            write_reference(stream, mesh);
            stream.close();
            if (!stream)
                throw std::runtime_error("model audit source write failed");
            prepared = {{"parent_triangles", parent_triangles},
                        {"triangles", mesh.view().triangles()},
                        {"sha256", file_sha256(reference)}};
            report["prepared_sources"].push_back({{"asset", asset.at("id")},
                                                  {"reference", reference.string()},
                                                  {"source", prepared}});
            write_json(output, report);
            if (!mesh.view().triangles() || mesh.view().triangles() > source_limit)
                throw std::runtime_error("pilot source did not reach its frozen size limit");
        }
        for (auto ranking : rankings) {
            options.ranking = ranking;
            NeuralModel model(model_path.c_str(), options);
            NeuralStats stats;
            json row, trace = json::array(), support_trace = json::array();
            std::optional<RankingSupport> support;
            if (collector && ranking == NeuralRanking::Learned)
                support.emplace(RankingSupport{
                    *collector,
                    [&](const SupportedRanking& value) {
                        const auto action = [](Action a) { return json{a.from, a.to, a.revision}; };
                        support_trace.push_back({{"global_top_action", action(value.global)},
                                                 {"global_top_score", value.global_score},
                                                 {"support_top_action", action(value.supported)},
                                                 {"support_top_score", value.supported_score},
                                                 {"global_top_in_support", value.global_in_support},
                                                 {"pool", value.pool},
                                                 {"seed", value.seed}});
                    },
                    support_seed});
            ExecutionObserver observe;
            if (trace_execution)
                observe = [&](const ActionRequest& request, const ActionTrial& trial,
                              uint8_t failed_gate) {
                    json actions = json::array();
                    for (const auto& action : trial.actions)
                        actions.push_back({action.from, action.to, action.revision});
                    trace.push_back(
                        {{"proposal", request.proposal},
                         {"output", request.output == OutputMode::Reuse ? "reuse" : "rebuild"},
                         {"pixels", request.source_audit.screen_size},
                         {"target_triangles", request.target_triangles},
                         {"trial", trial.trial},
                         {"iteration", trial.iteration},
                         {"faces_before", trial.faces_before},
                         {"faces_after", trial.faces_after},
                         {"first_rank", trial.first_rank},
                         {"rank_cursor", trial.rank_cursor},
                         {"actions", actions},
                         {"geometry_valid", trial.geometry_valid},
                         {"accepted", trial.accepted},
                         {"failed_gate", failed_gate}});
                };
            auto began = std::chrono::steady_clock::now();
            memory.budget.peak.store(memory.budget.live.load());
            try {
                auto result = trace_execution || support
                                  ? generate_observed(mesh.view(), settings,
                                                      load_weights(model_path), options, {}, &stats,
                                                      observe, support ? &*support : nullptr)
                                  : generate_neural(mesh.view(), settings, model, &stats);
                row = result_json(result);
                if (trace_execution)
                    for (size_t i = 0; i < result.lods.size(); ++i) {
                        std::ostringstream bytes(std::ios::out | std::ios::binary);
                        write_reference(bytes, copy_mesh(result.lods[i].view(mesh.view())));
                        const auto data = bytes.str();
                        row["lods"][i]["mesh_sha256"] = sha256(std::as_bytes(std::span(data)));
                    }
                if (result.status != Status::Complete || stats.resource_failures ||
                    stats.confirmation_nonfinite || stats.confirmation_cancelled)
                    complete = false;
            } catch (const gpu::ResourceError& e) {
                row = {{"status", "failed"},
                       {"error", e.what()},
                       {"workspace_limited", e.kind == NeuralResourceLimit::WorkspaceMemory},
                       {"requested_bytes", e.requested},
                       {"limit_bytes", e.limit}};
                complete = false;
            } catch (const std::exception& e) {
                row = {{"status", "failed"}, {"error", e.what()}};
                complete = false;
            }
            row["asset"] = asset.at("id");
            row["category"] = asset.at("category");
            row["ranking"] = ranking_name(ranking);
            if (source_limit)
                row["prepared_source"] = prepared;
            row["seconds"] =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
            row["gpu_workspace_peak_bytes"] = memory.budget.peak.load();
            row["neural"] = neural_json(stats);
            if (trace_execution)
                row["execution_trace"] = std::move(trace);
            if (support)
                row["supported_rankings"] = std::move(support_trace);
            report["rows"].push_back(row);
            write_json(output, report);
        }
    }
    report["complete"] = complete;
    report["seconds"] =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    write_json(output, report);
}
} // namespace blitz::neural::training
