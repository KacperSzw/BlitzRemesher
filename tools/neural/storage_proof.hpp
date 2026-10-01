#pragma once
#include "training/json.hpp"
#include "training/resident.hpp"
namespace blitz::neural::training {
inline void storage_proof(const fs::path& shards, const fs::path& output, const fs::path& initial) {
    if (fs::exists(output))
        throw std::invalid_argument("choose a fresh storage proof directory");
    fs::create_directories(output);
    torch::manual_seed(371);
    torch::Device device(torch::kCUDA);
    std::vector<fs::path> directories;
    for (auto& entry : fs::directory_iterator(shards))
        if (entry.is_directory() && fs::exists(entry.path() / "actions.bin"))
            directories.push_back(entry.path());
    std::sort(directories.begin(), directories.end());
    if (directories.empty())
        throw std::invalid_argument("no storage comparison shards");
    json report = {{"complete", false},
                   {"score", nullptr},
                   {"generalization_proven", false},
                   {"updates", 2048},
                   {"batch", 512},
                   {"binary_sha256", file_sha256("/proc/self/exe")},
                   {"initial_model_sha256", file_sha256(initial)},
                   {"shards", json::array()},
                   {"variants", json::array()}};
    ResidentDataset fp32(device, 913, false), compact(device, 913, true);
    std::vector<double> maximum(placement_features), square(placement_features);
    uint64_t rows = 0, legacy_bytes = 0, compact_bytes = 0, exceptions = 0;
    double target_error = 0;
    for (auto& directory : directories) {
        auto data = load_actions(directory / "actions.bin");
        auto index = read_json(directory / "index.json");
        if (!index.at("complete").get<bool>() || !is_placement_schema(data.architecture))
            throw std::invalid_argument("storage proof requires complete placement shards");
        auto packed = compact_actions(data);
        auto decoded = expand_compact(packed);
        fp32.append(data, index.at("asset"), index.at("category"));
        compact.append(packed, index.at("asset"), index.at("category"));
        rows += data.labels.size();
        exceptions += packed.escape_ids.size();
        for (size_t i = 0; i < data.x.size(); ++i) {
            double delta = std::abs(double(data.x[i]) - decoded.x[i]);
            maximum[i % placement_features] = std::max(maximum[i % placement_features], delta);
            square[i % placement_features] += delta * delta;
        }
        for (size_t i = 0; i < data.targets.size(); ++i)
            target_error =
                std::max(target_error, std::abs(double(data.targets[i]) - decoded.targets[i]));
        auto stem = directory.filename().string();
        save_actions(output / (stem + "-fp32.bin"), data, false);
        save_actions(output / (stem + "-compact.bin"), data);
        legacy_bytes += fs::file_size(output / (stem + "-fp32.bin"));
        compact_bytes += fs::file_size(output / (stem + "-compact.bin"));
        report["shards"].push_back({{"path", directory.string()},
                                    {"sha256", file_sha256(directory / "actions.bin")},
                                    {"states", data.states()},
                                    {"rows", data.labels.size()}});
    }
    for (auto& x : square)
        x = std::sqrt(x / rows);
    report["feature_max_error"] = maximum;
    report["feature_rms_error"] = square;
    report["target_max_error"] = target_error;
    report["exceptions"] = exceptions;
    report["disk_bytes"] = {{"fp32", legacy_bytes}, {"compact", compact_bytes}};
    report["resident_bytes"] = {{"fp32", fp32.bytes()}, {"compact", compact.bytes()}};
    auto weights = load_weights(initial);
    if (!is_placement_schema(weights.architecture) || weights.hidden_width != 64)
        throw std::invalid_argument("storage proof needs a width-64 placement policy");
    std::vector<std::vector<float>> final;
    for (bool packed : {false, true}) {
        ActionNetwork model(weights.architecture);
        model->to(device);
        {
            torch::NoGradGuard guard;
            size_t offset = 0;
            for (auto& p : model->parameters()) {
                p.copy_(
                    torch::from_blob(weights.values.data() + offset, p.sizes(), torch::kFloat32));
                offset += p.numel();
            }
        }
        auto& data = packed ? compact : fp32;
        ResidentUpdate update(model, data, 512);
        update.capture();
        auto t = std::chrono::steady_clock::now();
        update.run(2048);
        auto state = update.optimizer.state();
        double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
        if (state.failure || !std::isfinite(state.loss) || !std::isfinite(state.gradient))
            throw std::runtime_error("storage proof nonfinite update");
        auto model_path = output / (packed ? "compact.blzn" : "fp32.blzn");
        auto trained = export_actions(
            model, {{"storage_proof", true}, {"compact", packed}, {"updates", 2048}});
        save_weights(model_path, trained);
        final.push_back(trained.values);
        auto manifest = read_json("research/neural/action-diagnostic.json");
        auto settings = settings_json(read_json("research/neural/refactor-smoke.json"));
        NeuralOptions options;
        options.memory_mib = 1024;
        options.raster_backend = NeuralRasterBackend::Vulkan;
        options.vertex_storage = NeuralVertexStorage::Float32;
        options.action_trials = 8;
        options.action_batch = 16;
        AuditSession session(options);
        NeuralModel inference(model_path.c_str(), options);
        json quality = json::array();
        for (auto& asset : manifest.at("assets")) {
            for (auto& file : asset.at("files"))
                if (file_sha256(file.at("path").get<std::string>()) !=
                    file.at("sha256").get<std::string>())
                    throw std::invalid_argument("diagnostic mesh changed");
            auto mesh = load_mesh(asset.at("path").get<std::string>());
            auto result = generate_neural(mesh.view(), settings, inference);
            auto record = result_json(result);
            record["asset"] = asset.at("id");
            quality.push_back(record);
            if (result.status != Status::Complete)
                throw std::runtime_error("storage proof quality audit incomplete");
        }
        report["variants"].push_back({{"storage", packed ? "compact" : "fp32"},
                                      {"update_seconds", elapsed},
                                      {"loss", state.loss},
                                      {"gradient", state.gradient},
                                      {"sample_ids_sha256",
                                       [&] {
                                           auto ids = update.ids.cpu().contiguous();
                                           return sha256(std::as_bytes(std::span(
                                               ids.data_ptr<int32_t>(), size_t(ids.numel()))));
                                       }()},
                                      {"quality", quality}});
        write_json(output / "report.json", report);
    }
    double maximum_weight = 0;
    for (size_t i = 0; i < final[0].size(); ++i)
        maximum_weight = std::max(maximum_weight, std::abs(double(final[0][i]) - final[1][i]));
    report["maximum_weight_difference"] = maximum_weight;
    if (report["variants"][0]["sample_ids_sha256"] != report["variants"][1]["sample_ids_sha256"])
        throw std::runtime_error("storage variants sampled different examples");
    report["complete"] = true;
    write_json(output / "report.json", report);
    std::cout << report.dump(2) << '\n';
}
} // namespace blitz::neural::training
