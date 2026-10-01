#pragma once
#include "neural/cuda.cuh"
#include "tools/neural/audit_settings.hpp"
#include "training/data.hpp"
#include "training/json.hpp"
#include <set>
namespace blitz::neural::training {
inline void audit_model(const fs::path& manifest_path, const fs::path& model_path,
                        const fs::path& settings_path, const fs::path& output) {
    if (fs::exists(output))
        throw std::invalid_argument("model audit output exists");
    auto manifest = read_json(manifest_path), config = read_json(settings_path);
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
    for (const auto& asset : manifest.at("assets")) {
        if (asset.at("split") != "development" && asset.at("split") != "validation")
            throw std::invalid_argument("model comparison cannot tune on release holdout");
        for (const auto& f : asset.at("files"))
            if (f.at("sha256") != file_sha256(f.at("path").get<std::string>()))
                throw std::invalid_argument("model audit source changed");
        auto mesh = load_mesh(asset.at("path").get<std::string>());
        for (auto ranking : rankings) {
            options.ranking = ranking;
            NeuralModel model(model_path.c_str(), options);
            NeuralStats stats;
            json row;
            auto began = std::chrono::steady_clock::now();
            memory.budget.peak.store(memory.budget.live.load());
            try {
                auto result = generate_neural(mesh.view(), settings, model, &stats);
                row = result_json(result);
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
            row["seconds"] =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
            row["gpu_workspace_peak_bytes"] = memory.budget.peak.load();
            row["neural"] = neural_json(stats);
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
