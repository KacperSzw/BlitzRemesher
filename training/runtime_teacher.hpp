#pragma once
#include "neural/generation.hpp"
#include "neural/memory.hpp"
#include "training/action_data.hpp"
#include "training/mesh_cache.hpp"
#include "training/packing.hpp"
#include "training/policy_ranking.hpp"
#include "training/teacher_labels.hpp"

namespace blitz::neural::training {
struct RuntimeTeachingSettings {
    uint32_t states_per_proposal{16}, maximum_states{256}, pool{16}, seed{101};
    bool cache_rasters{};
    std::function<bool()> cancelled;
};
struct RuntimeTeachingResult {
    bool complete{};
    ActionData data;
    std::vector<uint8_t> geometry;
    json observations = json::array(), requests = json::array(), verification;
    NeuralStats teacher_stats;
    double teacher_seconds{}, baseline_seconds{}, rollout_seconds{};
};

// Observe genuine requests/trajectories, including the actual emitted predecessor.
// Endpoint targets are deliberately separate from predicted-placement targets.
inline RuntimeTeachingResult prepare_runtime_rankings(MeshView source, const Settings& settings,
                                                      const WeightsData& weights,
                                                      const NeuralOptions& options,
                                                      const RuntimeTeachingSettings& teaching,
                                                      const fs::path& output) {
    if (weights.architecture != conditioned_placement_schema ||
        settings.profile != Profile::Coverage || !teaching.pool || teaching.pool > 16 ||
        !teaching.states_per_proposal || teaching.states_per_proposal > 256 ||
        !teaching.maximum_states || teaching.maximum_states > 4096)
        throw std::invalid_argument("invalid runtime coverage teaching request");
    RuntimeTeachingResult out;
    out.data.architecture = conditioned_placement_schema;
    using Clock = std::chrono::steady_clock;
    auto began = Clock::now();
    NeuralStats baseline_stats;
    std::vector<uint32_t> iterations;
    auto count_iterations = [&](GpuActionState&, const ActionRequest& request) {
        if (request.output != OutputMode::Reuse)
            return;
        if (request.proposal >= iterations.size())
            iterations.resize(size_t(request.proposal) + 1);
        ++iterations[request.proposal];
    };
    const auto baseline =
        generate_observed(source, settings, weights, options, count_iterations, &baseline_stats);
    out.baseline_seconds = std::chrono::duration<double>(Clock::now() - began).count();
    auto baseline_json = result_json(baseline);
    baseline_json["neural"] = neural_json(baseline_stats);
    write_json(output / "baseline.json", baseline_json);
    if (baseline.status != Status::Complete)
        return out;
    auto snapshot = [&](MeshView mesh, const std::string& name) {
        const auto file = output / name;
        std::ofstream stream(file, std::ios::binary);
        // Exact FP32 reference format: packed serialization would change labels.
        write_reference(stream, copy_mesh(mesh));
        stream.close();
        if (!stream)
            throw std::runtime_error("runtime teacher reference write failed");
        return file_sha256(file);
    };
    const auto source_hash = snapshot(source, "source-reference.bin");
    MemoryScope memory(options);
    AuditSession session(options);
    ActionCuda policy(weights, options);
    auto teacher_options = options;
    teacher_options.cache_rasters = teaching.cache_rasters;
    AuditCuda audit(teacher_options, source);
    bool unknown = false;
    std::vector<uint32_t> counts(iterations.size()), visited(iterations.size());
    auto observe = [&](GpuActionState& state, const ActionRequest& request) {
        if (request.output != OutputMode::Reuse)
            return;
        if (request.proposal >= iterations.size())
            throw std::runtime_error("runtime observation changed proposal sequence");
        const auto ordinal = visited[request.proposal]++;
        const auto requested = std::min(iterations[request.proposal], teaching.states_per_proposal);
        if (unknown || out.data.states() >= teaching.maximum_states ||
            counts[request.proposal] >= requested ||
            ordinal != ranking_observation_step(counts[request.proposal],
                                                iterations[request.proposal], requested))
            return;
        const auto start = Clock::now();
        audit.bind_reference(request.previous);
        struct ReferenceScope {
            AuditCuda& audit;
            ~ReferenceScope() {
                audit.clear_reference();
            }
        } reference_scope{audit};
        auto a = request.source_audit, b = request.adjacent_audit, search_a = request.source_search,
             search_b = request.adjacent_search;
        // Caller cancellation can be stateful. Teaching uses its own bounded
        // cancellation source and never consumes the generator's poll sequence.
        for (auto* config : {&a, &b, &search_a, &search_b}) {
            config->cancelled = teaching.cancelled;
            config->performance = nullptr;
            config->max_supersample = bounded_refinement(*config);
        }
        if (!counts[request.proposal]) {
            const auto name = "previous-" + std::to_string(request.proposal) + ".bin";
            const auto previous_hash = snapshot(request.previous, name);
            out.requests.push_back(
                {{"proposal", request.proposal},
                 {"output", "reuse"},
                 {"trajectory_iterations", iterations[request.proposal]},
                 {"requested_observations", requested},
                 {"target_triangles", request.target_triangles},
                 {"target_ratio", double(request.target_triangles) / source.triangles()},
                 {"condition", request.condition},
                 {"pixels", a.screen_size},
                 {"source_limit", a.limit},
                 {"adjacent_limit", b.limit},
                 {"source_reference_sha256", source_hash},
                 {"previous_reference", name},
                 {"previous_reference_sha256", previous_hash},
                 {"previous_triangles", request.previous.triangles()}});
        }
        const auto rows =
            state.teacher_actions(request.condition, teaching.pool,
                                  teaching.seed + uint32_t(out.data.states()) * 0x9e3779b9u,
                                  &policy, TeacherSelection::PolicyMixed);
        if (rows.empty())
            return;
        ++counts[request.proposal];
        const auto first = out.data.labels.size();
        std::vector<TeacherChoice> choices;
        json trace = {{"proposal", request.proposal},
                      {"state", out.data.states()},
                      {"iteration", ordinal},
                      {"triangles", state.view().faces},
                      {"rows", json::array()}};
        for (const auto& row : rows) {
            DeviceMeshView candidate;
            const bool valid = state.trial(row.action, candidate);
            uint8_t label = 0;
            TeacherChoice choice;
            json observation = {
                {"from", row.action.from}, {"to", row.action.to}, {"geometry_rejected", !valid}};
            if (valid) {
                audit.with_candidate_rasters(std::span{&candidate, 1}, [&] {
                    const auto x = audit.evaluate(request.source, candidate, request.bounds, a,
                                                  &out.teacher_stats);
                    const auto y = audit.evaluate(request.previous, candidate, request.bounds, b,
                                                  &out.teacher_stats);
                    const auto sx = audit.evaluate(request.source, candidate, request.bounds,
                                                   search_a, &out.teacher_stats);
                    const auto sy = audit.evaluate(request.previous, candidate, request.bounds,
                                                   search_b, &out.teacher_stats);
                    const bool known_x =
                                   action_audit_known(x, a) && action_audit_known(sx, search_a),
                               known_y =
                                   action_audit_known(y, b) && action_audit_known(sy, search_b);
                    label = uint8_t((known_x ? SourceKnown : 0) | (known_y ? AdjacentKnown : 0) |
                                    (known_x && x.passed && sx.passed ? PlacementSourcePass : 0) |
                                    (known_y && y.passed && sy.passed ? PlacementAdjacentPass : 0));
                    unknown |= !known_x || !known_y;
                    observation["source"] = measurement_json(x);
                    observation["adjacent"] = measurement_json(y);
                    observation["source_search"] = measurement_json(sx);
                    observation["adjacent_search"] = measurement_json(sy);
                    if (known_x && known_y && x.passed && y.passed && sx.passed && sy.passed) {
                        const auto area = [](double error, double limit) {
                            return limit > 0 ? error / limit : (error == 0 ? 0. : INFINITY);
                        };
                        choice = {candidate.faces,
                                  std::max({x.error / a.limit, y.error / b.limit,
                                            sx.error / search_a.limit, sy.error / search_b.limit,
                                            area(x.changed_area, a.max_changed_area),
                                            area(y.changed_area, b.max_changed_area)}),
                                  x.complete && y.complete && sx.complete && sy.complete};
                    }
                });
            }
            observation["known_mask"] = label;
            trace["rows"].push_back(std::move(observation));
            append_geometry_rejection(out.geometry, out.data.labels.size(), !valid);
            out.data.x.insert(out.data.x.end(), row.x.begin(), row.x.end());
            out.data.labels.push_back(label);
            out.data.from.push_back(row.action.from);
            out.data.to.push_back(row.action.to);
            out.data.targets.resize(out.data.labels.size() * 9);
            choices.push_back(choice);
            if (unknown)
                break;
        }
        const auto preferred = preferred_actions(choices);
        for (size_t i = 0; i < choices.size(); ++i)
            if (preferred & (1u << i))
                out.data.labels[first + i] |= PlacementPreferred;
        out.data.offsets.push_back(uint32_t(out.data.labels.size()));
        out.data.progress.push_back(float(double(state.view().faces) / source.triangles()));
        trace["preferred_mask"] = preferred;
        out.observations.push_back(std::move(trace));
        out.teacher_seconds += std::chrono::duration<double>(Clock::now() - start).count();
    };
    NeuralStats observed_stats;
    began = Clock::now();
    const auto observed =
        generate_observed(source, settings, weights, options, observe, &observed_stats);
    out.rollout_seconds = std::chrono::duration<double>(Clock::now() - began).count();
    auto observed_json = result_json(observed);
    observed_json["neural"] = neural_json(observed_stats);
    write_json(output / "observed.json", observed_json);
    bool same = observed.status == baseline.status && observed.lods.size() == baseline.lods.size();
    for (size_t i = 0; same && i < observed.lods.size(); ++i)
        same = same_mesh_data(observed.lods[i].view(source), baseline.lods[i].view(source)) &&
               observed.lods[i].source_error.error == baseline.lods[i].source_error.error &&
               observed.lods[i].adjacent.error == baseline.lods[i].adjacent.error;
    const bool counters = baseline_stats.action_ranked == observed_stats.action_ranked &&
                          baseline_stats.action_trials == observed_stats.action_trials &&
                          baseline_stats.legal_collapses == observed_stats.legal_collapses &&
                          baseline_stats.rejected_collapses == observed_stats.rejected_collapses;
    out.verification = {{"same_meshes_and_errors", same},
                        {"same_action_counts", counters},
                        {"same_trajectory_iterations", visited == iterations},
                        {"unknown_observation", unknown},
                        {"teacher_cache_rasters", teaching.cache_rasters}};
    out.complete = same && counters && visited == iterations && !unknown &&
                   observed.status == Status::Complete;
    return out;
}
} // namespace blitz::neural::training
