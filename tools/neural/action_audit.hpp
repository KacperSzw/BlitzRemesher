#pragma once
#include "neural/memory.hpp"
#include "neural/stream.hpp"
#include "tools/neural/action_probe.hpp"
#include "tools/neural/audit_settings.hpp"
#include "training/mesh_cache.hpp"
#include "training/packing.hpp"
#include <set>

namespace blitz::neural::training {
inline void audit_actions(const fs::path& manifest_path, const fs::path& initial_path,
                          const fs::path& final_path, const fs::path& settings_path,
                          const fs::path& output) {
    using namespace diagnostic;
    auto snapshots = output.parent_path() / (output.stem().string() + "-states");
    if (fs::exists(output) || fs::exists(snapshots))
        throw std::invalid_argument("choose a fresh action audit output");
    auto manifest = read_json(manifest_path), config = read_json(settings_path);
    config.erase("audit_rankings");
    NeuralOptions options;
    options.raster_backend = NeuralRasterBackend::Vulkan;
    options.memory_mib = 1024;
    options.action_trials = 64;
    options.action_batch = 16;
    auto settings = model_audit_settings(config, options);
    if (settings.levels != 4 || settings.profile != Profile::Coverage ||
        manifest.at("assets").size() != 2)
        throw std::invalid_argument(
            "action audit requires two development assets and four coverage LODs");
    const auto initial_weights = load_weights(initial_path),
               final_weights = load_weights(final_path);
    if (initial_weights.architecture != conditioned_placement_schema ||
        final_weights.architecture != conditioned_placement_schema)
        throw std::invalid_argument("action audit requires two v4 models");
    std::set<std::string> identities;
    for (const auto& asset : manifest.at("assets")) {
        if (asset.at("split") != "development" ||
            !identities.insert(asset.at("id").get<std::string>()).second)
            throw std::invalid_argument("action audit requires distinct development assets");
        for (const auto& f : asset.at("files"))
            if (f.at("sha256") != file_sha256(f.at("path").get<std::string>()))
                throw std::invalid_argument("action audit source changed");
    }
    const auto begin = std::chrono::steady_clock::now();
    constexpr uint32_t pool = 16, seed = 0xB1172026, deadline_seconds = 600;
    bool cancelled = false;
    settings.cancelled = [&] {
        return cancelled = cancelled || std::chrono::steady_clock::now() - begin >=
                                            std::chrono::seconds(deadline_seconds);
    };
    json report = {
        {"version", 1},
        {"complete", false},
        {"score", nullptr},
        {"scope",
         "independent single edits in a fixed geometry-selected pool; no chain or batch claim"},
        {"models", {{"initial", file_sha256(initial_path)}, {"final", file_sha256(final_path)}}},
        {"prediction_order", {"initial", "final"}},
        {"binary_sha256", file_sha256("/proc/self/exe")},
        {"manifest_sha256", file_sha256(manifest_path)},
        {"settings_sha256", file_sha256(settings_path)},
        {"visual_settings", settings_json(settings)},
        {"execution_settings", neural_json(options)},
        {"pool", pool},
        {"pool_selection", "half current-plane, half deterministic random; independent of model"},
        {"pool_seed", seed},
        {"tie_order", "stable frozen pool order"},
        {"deadline_seconds", deadline_seconds},
        {"single_edit_budget", std::min(pool, options.action_trials)},
        {"states", json::array()},
        {"initializer_chains", json::array()}};
    auto save = [&] {
        report["seconds"] =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        write_json(output, report);
    };
    auto save_mesh = [&](const Mesh& mesh, const std::string& name) {
        fs::create_directories(snapshots);
        auto path = snapshots / name;
        std::ofstream f(path, std::ios::binary);
        write_reference(f, mesh);
        f.close();
        if (!f)
            throw std::runtime_error("action audit snapshot write failed");
        return json{{"path", path.string()},
                    {"sha256", file_sha256(path)},
                    {"triangles", mesh.view().triangles()}};
    };
    auto measurement = [](const Measurement& m, const EvalSettings& e) {
        auto value = measurement_json(m);
        value["cancelled"] = m.cancelled;
        value["known"] = action_audit_known(m, e);
        return value;
    };
    save();
    bool complete = true;
    try {
        gpu::StreamScope stream;
        MemoryScope memory(options);
        AuditSession session(options);
        ActionCuda initial(initial_weights, options), final(final_weights, options);
        NeuralModel initializer(initial_path.c_str(), options);
        size_t asset_index = 0;
        for (const auto& asset : manifest.at("assets")) {
            if (settings.cancelled())
                break;
            auto source = load_mesh(asset.at("path").get<std::string>());
            auto source_record =
                save_mesh(source, "asset-" + std::to_string(asset_index) + "-source.bin");
            auto bounds = blitz::bounds(source.view());
            auto quantization = vertex_bounds(source.view());
            NeuralStats chain_stats;
            auto chain = generate_neural(source.view(), settings, initializer, &chain_stats);
            report["initializer_chains"].push_back({{"asset", asset.at("id")},
                                                    {"result", result_json(chain)},
                                                    {"neural", neural_json(chain_stats)}});
            save();
            if (chain.status != Status::Complete || chain.lods.size() != 4 ||
                chain_stats.resource_failures || chain_stats.confirmation_cancelled ||
                chain_stats.confirmation_nonfinite)
                throw std::runtime_error("initializer could not supply complete common snapshots");
            const auto steps = schedule(bounds, settings);
            for (unsigned level : {0u, 2u}) {
                if (settings.cancelled())
                    break;
                Mesh previous = copy_mesh(chain.lods[level].view(chain.source));
                GpuActionState state(previous.view(), options, true, &quantization, source.view());
                const auto before = state.snapshot().data;
                const auto next = steps.at(level + 1);
                EvalSettings source_eval;
                source_eval.profile = Profile::Coverage;
                source_eval.screen_size = next.pixels;
                source_eval.limit = next.source;
                source_eval.max_changed_area = settings.max_changed_area;
                source_eval.views = settings.audit_views;
                source_eval.supersample = settings.audit_supersample;
                source_eval.max_supersample = settings.max_supersample;
                source_eval.force_scalar = settings.force_scalar;
                source_eval.cancelled = settings.cancelled;
                auto adjacent_eval = source_eval;
                adjacent_eval.limit = next.transition;
                // Reuse the initializer's actual rebuilt proposal condition.
                // A target is never inferred from whether its search succeeded.
                const auto requested =
                    std::find_if(chain_stats.action_proposals.begin(),
                                 chain_stats.action_proposals.end(), [&](const auto& p) {
                                     return p.output == OutputMode::Rebuild &&
                                            p.screen_pixels == next.pixels &&
                                            p.start_triangles == previous.view().triangles();
                                 });
                if (requested == chain_stats.action_proposals.end())
                    throw std::runtime_error(
                        "initializer has no matching rebuilt proposal condition");
                const auto target = requested->target_triangles;
                const auto c = condition(source_eval, adjacent_eval.limit,
                                         double(target) / source.view().triangles());
                AuditCuda audit(options, source.view());
                audit.bind_reference(previous.view());
                NeuralStats audit_stats;
                auto& row = report["states"].emplace_back(
                    json{{"asset", asset.at("id")},
                         {"initializer_level", level},
                         {"complete", false},
                         {"source", source_record},
                         {"predecessor",
                          save_mesh(previous, "asset-" + std::to_string(asset_index) + "-lod-" +
                                                  std::to_string(level) + ".bin")},
                         {"working",
                          save_mesh(before, "asset-" + std::to_string(asset_index) + "-working-" +
                                                std::to_string(level) + ".bin")},
                         {"pixels", next.pixels},
                         {"source_limit", next.source},
                         {"adjacent_limit", next.transition},
                         {"condition", c},
                         {"requested_target_triangles", target},
                         {"target_condition_origin", "recorded initializer rebuilt proposal"},
                         {"committed_actions", 0},
                         {"actions", json::array()},
                         {"combinations", json::array()}});
                auto a =
                    audit.evaluate(source.view(), state.view(), bounds, source_eval, &audit_stats);
                auto b = audit.evaluate(previous.view(), state.view(), bounds, adjacent_eval,
                                        &audit_stats);
                row["baseline"] = {{"source", measurement(a, source_eval)},
                                   {"adjacent", measurement(b, adjacent_eval)}};
                if (!a.complete || !a.passed || !b.complete || !b.passed || a.cancelled ||
                    b.cancelled) {
                    row["stop_reason"] = "infeasible_or_unknown_snapshot";
                    complete = false;
                    save();
                    continue;
                }
                const std::array probes{capture_actions(state, c, initial, pool, seed),
                                        capture_actions(state, c, final, pool, seed)};
                require_same_actions(probes[0], probes[1]);
                std::array<std::vector<ActionObservation>, 2> observations;
                for (auto& values : observations)
                    values.resize(probes[0].rows.size());
                for (size_t i = 0; i < probes[0].rows.size(); ++i) {
                    const auto& action = probes[0].rows[i];
                    auto& entry =
                        row["actions"].emplace_back(json{{"from", action.action.from},
                                                         {"to", action.action.to},
                                                         {"revision", action.action.revision},
                                                         {"features", action.x},
                                                         {"predictions", json::array()},
                                                         {"placements", json::array()}});
                    for (unsigned m = 0; m < 2; ++m) {
                        const auto& p = probes[m];
                        entry["predictions"].push_back(std::vector<float>(
                            p.predictions.begin() + i * placement_outputs,
                            p.predictions.begin() + (i + 1) * placement_outputs));
                        const auto& place = p.placements[i];
                        auto& value = entry["placements"].emplace_back(json{
                            {"model", m ? "final" : "initial"},
                            {"position", {place.position.x, place.position.y, place.position.z}},
                            {"normals",
                             {{place.normals[0].x, place.normals[0].y, place.normals[0].z},
                              {place.normals[1].x, place.normals[1].y, place.normals[1].z}}},
                            {"status", "not_queried"},
                            {"known", false}});
                        if (settings.cancelled())
                            continue;
                        auto& observation = observations[m][i];
                        DeviceMeshView candidate;
                        observation.valid = state.trial(action.action, place, candidate);
                        if (!observation.valid) {
                            observation.known = true;
                            value["status"] = "invalid_geometry";
                            value["known"] = true;
                        } else {
                            observation.faces = candidate.faces;
                            audit.with_candidate_rasters(std::span{&candidate, 1}, [&] {
                                a = audit.evaluate(source.view(), candidate, bounds, source_eval,
                                                   &audit_stats);
                                b = audit.evaluate(previous.view(), candidate, bounds,
                                                   adjacent_eval, &audit_stats);
                            });
                            observation.known = action_audit_known(a, source_eval) &&
                                                action_audit_known(b, adjacent_eval);
                            observation.passed = observation.known && a.complete && a.passed &&
                                                 b.complete && b.passed;
                            value["status"] = !observation.known   ? "unknown"
                                              : observation.passed ? "safe"
                                                                   : "rejected";
                            value["known"] = observation.known;
                            value["source"] = measurement(a, source_eval);
                            value["adjacent"] = measurement(b, adjacent_eval);
                            value["triangles"] = observation.faces;
                        }
                        save();
                    }
                }
                bool known = true;
                for (unsigned ranking = 0; ranking < 2; ++ranking) {
                    const auto order = ranked_actions(probes[ranking]);
                    for (unsigned placement = 0; placement < 2; ++placement) {
                        const auto result = first_safe(order, observations[placement],
                                                       std::min(pool, options.action_trials));
                        const char* stop = result.stop == ProbeStop::SafeAction ? "safe_action"
                                           : result.stop == ProbeStop::Unknown  ? "unknown"
                                           : result.stop == ProbeStop::TrialBudget
                                               ? "trial_budget"
                                               : "pool_exhausted";
                        json summary = {{"ranking_model", ranking ? "final" : "initial"},
                                        {"placement_model", placement ? "final" : "initial"},
                                        {"order", order},
                                        {"stop_reason", stop},
                                        {"trials", result.trials},
                                        {"first_safe_row", nullptr},
                                        {"triangles_removed", nullptr},
                                        {"top_one_safe", nullptr}};
                        if (!order.empty() && observations[placement][order.front()].known)
                            summary["top_one_safe"] = observations[placement][order.front()].passed;
                        if (result.stop == ProbeStop::SafeAction) {
                            summary["first_safe_row"] = result.row;
                            summary["triangles_removed"] =
                                before.view().triangles() -
                                observations[placement][result.row].faces;
                        }
                        row["combinations"].push_back(summary);
                    }
                }
                for (const auto& values : observations)
                    for (const auto& value : values)
                        known &= value.known;
                row["geometry_unchanged"] =
                    same_mesh_data(before.view(), state.snapshot().data.view());
                if (!row["geometry_unchanged"].get<bool>())
                    throw std::runtime_error("independent action audit mutated the snapshot");
                row["audit"] = neural_json(audit_stats);
                row["complete"] = known && !settings.cancelled();
                row["stop_reason"] = cancelled ? "deadline"
                                     : known   ? "pool_complete"
                                               : "unknown_audit";
                complete &= row["complete"].get<bool>();
                save();
            }
            ++asset_index;
        }
        report["complete"] = complete && !settings.cancelled() && report["states"].size() == 4;
        report["stop_reason"] = cancelled                        ? "deadline"
                                : report["complete"].get<bool>() ? "complete"
                                                                 : "incomplete";
    } catch (const std::exception& error) {
        report["error"] = error.what();
        report["stop_reason"] = "exception";
        if (const auto* resource = dynamic_cast<const gpu::ResourceError*>(&error)) {
            report["stop_reason"] = "resource";
            report["resource"] = {
                {"requested_bytes", resource->requested},
                {"limit_bytes", resource->limit},
                {"workspace_limited", resource->kind == NeuralResourceLimit::WorkspaceMemory}};
        }
        save();
        throw;
    }
    save();
}
} // namespace blitz::neural::training
