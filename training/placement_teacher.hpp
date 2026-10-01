#pragma once
#include "neural/action_gpu.hpp"
#include "neural/cuda.cuh"
#include "neural/memory.hpp"
#include "neural/timeline.hpp"
#include "training/action_data.hpp"
#include "training/json.hpp"
#include "training/packing.hpp"
#include "training/policy_ranking.hpp"
#include "training/teacher_cancellation.hpp"
#include "training/teacher_labels.hpp"
#include "training/teacher_seed.hpp"
#include "training/teacher_strategy.hpp"
#include "training/training_cache.hpp"
#include <iostream>
namespace blitz::neural::training {
struct PlacementRequest {
    uint32_t states{8}, pool{4}, previous_steps{}, seed{101};
    double pixels{128}, minutes{5}, source_limit{3}, adjacent_limit{3};
    bool sparse{true};
    std::string policy_sha256;
    std::function<bool()> cancelled;
    bool compact_data{true};
    uint32_t repair_budget{256};
    fs::path corpus{"research/corpus.json"}, selection{"research/neural/training-manifest.json"},
        mesh_cache;
    Profile profile{Profile::Attributes};
    TrainingCache* cache{};
    fs::path episode;
    double retained{1};
    bool simplifier{};
    uint32_t architecture{placement_schema}, policy_rollout_trials{};
    bool policy_candidates{}, preserve_uv{true};
    double previous_pixels{};
    TeacherStrategy strategy{TeacherStrategy::Exhaustive};
    bool policy_ranking{};
    std::optional<EvalSettings> policy_audit;
};
struct PlacementResult {
    bool complete;
    ActionData data;
    json index;
    CompactActions compact;
};
inline PlacementResult prepare_placements(const std::string& asset, const fs::path& output,
                                          const PlacementRequest& request,
                                          const NeuralOptions& base_options,
                                          ActionCuda* policy = nullptr) {
    const auto& [states, pool, previous_steps, seed, pixels, minutes, source_limit, adjacent_limit,
                 sparse, policy_hash, cancelled, compact_data, repair_budget, corpus, selection,
                 mesh_cache, profile, cache, episode, retained, simplifier, architecture,
                 rollout_trials, policy_candidates, preserve_uv, previous_pixels, strategy,
                 policy_ranking, policy_audit] = request;
    auto options = base_options;
    options.preserve_uv = preserve_uv;
    const bool core_first = strategy == TeacherStrategy::CoverageCoreFirst;
    teacher_strategy_name(strategy);
    if (policy_ranking &&
        (!policy || architecture != conditioned_placement_schema || profile != Profile::Coverage ||
         core_first || compact_data || !policy_audit || policy_audit->profile != Profile::Coverage))
        throw std::invalid_argument("policy ranking requires a v4 coverage policy and FP32 data");
    if (core_first &&
        (architecture != conditioned_placement_schema || profile != Profile::Coverage))
        throw std::invalid_argument("coverage-core-first requires v4 coverage teaching");
    if ((rollout_trials && (!episode.empty() || simplifier)) ||
        !is_placement_schema(architecture) ||
        (!preserve_uv && architecture != conditioned_placement_schema) ||
        (policy && policy->architecture() != architecture) ||
        ((policy_candidates || rollout_trials) && !policy) || rollout_trials > 4096)
        throw std::invalid_argument("teacher policy/schema/UV contract");
    if (previous_pixels && (!previous_steps || !std::isfinite(previous_pixels) ||
                            previous_pixels < pixels || previous_pixels > 512))
        throw std::invalid_argument("predecessor screen size");
    if (!states || states > 4096 || previous_steps > 4096 || !pool || pool > 16 ||
        !std::isfinite(pixels) || pixels < 16 || pixels > 512 || !std::isfinite(minutes) ||
        minutes <= 0 || minutes > 50 || !std::isfinite(source_limit) || source_limit <= 0 ||
        source_limit > 16 || !std::isfinite(adjacent_limit) || adjacent_limit <= 0 ||
        adjacent_limit > 16)
        throw std::invalid_argument("preparation bounds");
    if (fs::exists(output / "contract.json"))
        throw std::invalid_argument("choose a fresh placement dataset directory");
    fs::create_directories(output);
    auto load_start = std::chrono::steady_clock::now();
    double proposals_seconds = 0, trial_seconds = 0, audit_seconds = 0, commit_seconds = 0,
           packing_seconds = 0, seed_seconds = 0, setup_seconds = 0, rollout_seconds = 0,
           predecessor_seconds = 0, confirmation_seconds = 0, final_audit_seconds = 0,
           output_seconds = 0;
    auto timed = [](double& total, auto&& fn) {
        auto t = std::chrono::steady_clock::now();
        auto value = fn();
        total += std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
        return value;
    };
    MemoryScope memory(options);
    auto metadata = cache ? cache->metadata(asset) : training_metadata(asset, corpus, selection);
    Mesh mesh, prepared;
    json cache_provenance;
    std::shared_ptr<const PreparedMesh> borrowed;
    if (cache) {
        borrowed = cache->get(asset);
        cache_provenance = borrowed->provenance;
    } else if (!mesh_cache.empty() && options.draw_storage() == NeuralVertexStorage::Packed) {
        auto cached = prepared_mesh(metadata, mesh_cache);
        mesh = std::move(cached.source);
        prepared = std::move(cached.draw);
        cache_provenance = std::move(cached.provenance);
    } else
        mesh = load_mesh(metadata.at("path").get<std::string>());
    auto source = borrowed ? borrowed->source.view() : mesh.view();
    auto prepared_view = borrowed ? borrowed->draw.view() : prepared.view();
    auto bounds = blitz::bounds(source);
    Lod previous;
    AuditSession session(options);
    AuditCuda audit(options, source);
    NeuralStats stats;
    ActionData data;
    std::vector<uint8_t> geometry_rejections;
    data.architecture = architecture;
    auto start = std::chrono::steady_clock::now();
    double load_seconds = std::chrono::duration<double>(start - load_start).count();
    auto seconds = [&] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    };
    auto record_timings = [&](json& index) {
        const auto now = std::chrono::steady_clock::now();
        const double preparation = std::chrono::duration<double>(now - start).count();
        index["seconds"] = preparation;
        index["total_wall_seconds"] = std::chrono::duration<double>(now - load_start).count();
        index["timing_version"] = 2;
        // Disjoint host wall intervals, ending before the final index write and
        // teardown. The packing breakdown is a subset, never an added phase.
        index["timing_breakdown"] = {
            {"packing_and_baseline_seconds", {{"seed_preparation_seconds", seed_seconds}}}};
        index["timings"] = {
            {"load_and_session_seconds", load_seconds},
            {"packing_and_baseline_seconds", packing_seconds},
            {"state_setup_seconds", setup_seconds},
            {"policy_rollout_seconds", rollout_seconds},
            {"predecessor_snapshot_and_audit_seconds", predecessor_seconds},
            {"features_and_proposals_seconds", proposals_seconds},
            {"candidate_build_seconds", trial_seconds},
            {"candidate_audit_seconds", audit_seconds},
            {"exact_confirmation_seconds", confirmation_seconds},
            {"commit_seconds", commit_seconds},
            {"final_audit_seconds", final_audit_seconds},
            {"final_output_seconds", output_seconds},
            {"other_preparation_seconds",
             std::max(0., preparation - packing_seconds - setup_seconds - rollout_seconds -
                              predecessor_seconds - proposals_seconds - trial_seconds -
                              audit_seconds - confirmation_seconds - commit_seconds -
                              final_audit_seconds - output_seconds)}};
    };
    TeacherCancellation cancellation;
    auto cancel = [&] {
        return cancellation.poll(
            [&] { return (request.cancelled && request.cancelled()) || seconds() > minutes * 60; });
    };
    auto e = action_eval(
        previous_steps ? (previous_pixels ? previous_pixels : std::min(512., pixels * 2)) : pixels,
        source_limit, profile);
    if (policy_audit) {
        const auto size = e.screen_size;
        e = *policy_audit;
        e.screen_size = size;
        e.limit = source_limit;
    }
    e.cancelled = cancel;
    auto adjacent = e;
    adjacent.limit = adjacent_limit;
    MeshView previous_view = source;
    bool emitted = false, exhausted = false, unknown = false;
    uint64_t queries = 0, reused_queries = 0, reused_adjacent = 0, pruned_queries = 0,
             proven_losers = 0;
    uint32_t accepted = 0;
    json contract = {
        {"training_profile", training_profile_name(profile)},
        {"mask_only_coverage", options.mask_only_coverage},
        {"candidate_batch", options.candidate_batch},
        {"data_storage",
         compact_data ? (architecture == conditioned_placement_schema ? "compact-v2" : "compact-v1")
                      : "fp32"},
        {"repair_budget", repair_budget},
        {"quantization", "fixed mesh bounds; grid placements"},
        {"schema", architecture},
        {"teacher_version", 6},
        {"seed_admission_version", 1},
        {"teacher_strategy", teacher_strategy_name(strategy)},
        {"raster", raster_name(options.raster_backend)},
        {"vertex_storage", storage_name(options.draw_storage())},
        {"metric_kind", sparse ? "threshold bounds; exact committed states" : "exact"},
        {"seed", seed},
        {"asset", metadata},
        {"source_manifest_sha256", cache ? cache->corpus_hash : file_sha256(corpus)},
        {"training_selection_sha256", cache ? cache->selection_hash : file_sha256(selection)},
        {"protocol_sha256", cache ? cache->protocol_hash : file_sha256("research/PROTOCOL.md")},
        {"binary_sha256", cache ? cache->binary_hash : file_sha256("/proc/self/exe")},
        {"states_requested", states},
        {"pool", pool},
        {"pixels", pixels},
        {"previous_steps", previous_steps},
        {"source_limit", source_limit},
        {"adjacent_limit", adjacent_limit},
        {"packing_admission_limit", std::min(source_limit, adjacent_limit)},
        {"area_limit", e.max_changed_area},
        {"views", {e.views.orthographic, e.views.perspective}},
        {"view_seed", e.views.rotation_seed},
        {"supersample", e.supersample},
        {"max_supersample", e.max_supersample},
        {"gpu_memory_mib", options.memory_mib},
        {"teacher_candidates", (profile == Profile::Coverage ? 10 : 20) + (policy ? 1 : 0)},
        {"policy_payload_sha256", request.policy_sha256},
        {"prepared_mesh", cache_provenance},
        {"target", "one audited joint XYZ/normal candidate per edge; no averaging"},
        {"preference", "minimum triangles, then exact normalized source/adjacent error; all "
                       "confirmed exact ties preferred"}};
    contract["preserve_uv"] = preserve_uv;
    if (policy_ranking) {
        contract["teacher_version"] = 7;
        contract["teacher_target"] = "policy-placement-v1";
        contract["teacher_candidates"] = 1;
        contract["target"] = "frozen policy placement; rank only; separate geometry rejection bits";
    }
    contract["previous_pixels"] = e.screen_size;
    contract["policy_action_candidates"] = policy_candidates;
    contract["policy_rollout_trials"] = rollout_trials;
    contract["episode"] = {{"kind", rollout_trials    ? "audited_policy_rollout"
                                    : simplifier      ? "audited_simplifier"
                                    : episode.empty() ? "source"
                                                      : "audited_teacher_trajectory"},
                           {"parent_sha256", episode.empty() ? "" : file_sha256(episode)},
                           {"target_retained", retained}};
    write_json(output / "contract.json", contract);
    json trace = json::array();
    uint64_t invalid_candidates = 0;
    auto packing_start = std::chrono::steady_clock::now();
    auto destination = e;
    destination.screen_size = pixels;
    auto packing = teacher_packing_settings(e, adjacent.limit);
    auto baseline = repair_packing(source, options, packing, repair_budget, prepared_view,
                                   previous_steps ? &destination : nullptr);
    write_json(output / "packing.json", baseline.diagnostics.is_null()
                                            ? json{{"initial", measurement_json(baseline.initial)},
                                                   {"final", measurement_json(baseline.final)},
                                                   {"trials", 0}}
                                            : baseline.diagnostics);
    if (!baseline.final.passed || !baseline.final.complete) {
        packing_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - packing_start).count();
        const auto output_start = std::chrono::steady_clock::now();
        save_actions(output / "actions.bin", data, compact_data);
        json index = {{"schema", architecture},
                      {"complete", false},
                      {"status", baseline.final.resource_limited ? "resource_failure"
                                 : cancel()                      ? "cancelled"
                                                                 : "representation_failure"},
                      {"asset", asset},
                      {"category", metadata.at("category")},
                      {"contract_sha256", file_sha256(output / "contract.json")},
                      {"path", "actions.bin"},
                      {"sha256", file_sha256(output / "actions.bin")},
                      {"states", 0},
                      {"queries", 0},
                      {"accepted", 0},
                      {"source_triangles", source.triangles()},
                      {"teacher_triangles", source.triangles()},
                      {"reference_confirmed", false},
                      {"seconds", seconds()},
                      {"training_started", false},
                      {"baseline", measurement_json(baseline.final)}};
        write_json(output / "trajectory.json", trace);
        output_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - output_start).count();
        record_timings(index);
        write_json(output / "index.json", index);
        return {false, std::move(data), index};
    }
    // Seeds are proposals, never targets. Every seed passes source, current
    // adjacent and destination audits; a rejected seed visibly retains LOD0.
    const auto seed_start = std::chrono::steady_clock::now();
    if (!std::isfinite(retained) || retained <= 0 || retained > 1)
        throw std::invalid_argument("episode retained fraction");
    auto quantization = vertex_bounds(source);
    std::unique_ptr<GpuActionState> prepared_state;
    json seed_result = {{"requested", contract.at("episode")}, {"accepted", false}};
    if (!episode.empty() || simplifier) {
        Mesh candidate;
        if (!episode.empty()) {
            std::ifstream f(episode, std::ios::binary);
            candidate = read_packed_mesh(f).first;
        } else {
            ReduceSettings r;
            r.target_triangles = std::max<size_t>(1, size_t(source.triangles() * retained));
            r.output = OutputMode::Rebuild;
            r.coupled_wedges = true;
            r.cancelled = cancel;
            auto reduced = reduce(baseline.mesh.view(), r);
            candidate = copy_mesh(reduced.view(baseline.mesh.view()));
        }
        seed_result["triangles"] = candidate.view().triangles();
        // Audit the exact packed working representation. A QEM proposal may
        // extend beyond the source bounds; the UNorm grid maps it into the
        // supported domain before evaluation. Unsupported UVs/precision remain
        // visible seed rejections, never fatal errors or accepted labels.
        try {
            auto proposed = std::make_unique<GpuActionState>(candidate.view(), options, true,
                                                             &quantization, source);
            auto measured = audit_teacher_seed(
                e, adjacent.limit, previous_steps ? &destination : nullptr,
                [&](const EvalSettings& settings) {
                    return audit.evaluate(source, proposed->view(), bounds, settings, &stats);
                });
            seed_result["audit"] = measurement_json(measured.source);
            seed_result["adjacent"] = measurement_json(measured.adjacent);
            seed_result["destination"] = measurement_json(measured.destination);
            if (measured.passed() && candidate.view().triangles() <= source.triangles()) {
                prepared_state = std::move(proposed);
                baseline.mesh = std::move(candidate);
                seed_result["accepted"] = true;
            }
        } catch (const std::invalid_argument& error) {
            seed_result["rejection"] = error.what();
        }
    }
    seed_result["start_retained"] = double(baseline.mesh.view().triangles()) / source.triangles();
    write_json(output / "seed.json", seed_result);
    seed_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - seed_start).count();
    packing_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - packing_start).count();
    auto state_start = std::chrono::steady_clock::now();
    if (!prepared_state)
        prepared_state = std::make_unique<GpuActionState>(baseline.mesh.view(), options, true,
                                                          &quantization, source);
    auto& state = *prepared_state;
    setup_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - state_start).count();
    if (rollout_trials) {
        const auto rollout_start = std::chrono::steady_clock::now();
        auto gate = [&](DeviceMeshView candidate) {
            TeacherSeedAudit measured;
            audit.with_candidate_rasters(std::span{&candidate, 1}, [&] {
                measured = audit_teacher_seed(
                    e, adjacent.limit, previous_steps ? &destination : nullptr,
                    [&](const EvalSettings& settings) {
                        return audit.evaluate(source, candidate, bounds, settings, &stats);
                    });
            });
            if (!action_audit_known(measured.source, e) ||
                !action_audit_known(measured.adjacent, adjacent) ||
                !action_audit_known(measured.destination, destination))
                unknown = true;
            return measured.passed();
        };
        ActionStats rollout;
        const auto initial_faces = state.view().faces;
        const auto target_faces = std::max<size_t>(1, size_t(source.triangles() * retained));
        auto end = state.execute(condition(e, adjacent.limit, retained), target_faces,
                                 rollout_trials, policy, NeuralRanking::Learned, seed,
                                 options.action_batch, gate, &rollout, cancel);
        seed_result["accepted"] = rollout.accepted > 0;
        seed_result["policy"] = {
            {"payload_sha256", policy_hash},
            {"initial_triangles", initial_faces},
            {"final_triangles", state.view().faces},
            {"ranked", rollout.ranked},
            {"trials", rollout.trials},
            {"accepted_actions", rollout.accepted},
            {"accepted_batches", rollout.accepted_batches},
            {"outcome",
             policy_rollout_outcome(rollout.stop_reason, state.view().faces, target_faces)},
            // Known, noncancelled execution does not imply
            // the requested retained fraction was reached.
            {"complete", !unknown && !cancel()}};
        if (rollout.accepted && !unknown && !cancel()) {
            std::ofstream f(output / "policy-episode.bin", std::ios::binary);
            write_packed_mesh(f, end.view(source), &quantization);
            f.close();
            if (!f)
                throw std::runtime_error("policy episode output failed");
            seed_result["policy"]["episode_sha256"] = file_sha256(output / "policy-episode.bin");
        }
        seed_result["start_retained"] = double(state.view().faces) / source.triangles();
        write_json(output / "seed.json", seed_result);
        rollout_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - rollout_start).count();
    }
    struct TeacherAudit {
        bool known{}, passed{};
        double error{}, changed_area{};
    };
    auto query = [&](MeshView reference, DeviceMeshView candidate, const EvalSettings& config,
                     double cutoff, bool* pruned) {
        if (sparse) {
            auto p = timed(audit_seconds, [&] {
                return audit.certify(reference, candidate, bounds, config, &stats, cutoff, pruned);
            });
            return TeacherAudit{p.verdict != AuditVerdict::Unknown, p.verdict == AuditVerdict::Pass,
                                p.error_upper, p.changed_area};
        }
        auto m = timed(audit_seconds, [&] {
            return audit.evaluate(reference, candidate, bounds, config, &stats, cutoff, pruned);
        });
        return TeacherAudit{action_audit_known(m, config), m.passed, m.error, m.changed_area};
    };
    struct Confirmation {
        Measurement source, adjacent;
        TeacherChoice choice;
        bool valid{}, known{}, passed{};
        uint8_t label() const {
            return uint8_t(SourceKnown | AdjacentKnown | (source.passed ? PlacementSourcePass : 0) |
                           (adjacent.passed ? PlacementAdjacentPass : 0));
        }
    };
    auto confirm = [&](Action action, const GpuActionState::Proposal& proposal,
                       bool allow_geometry_rejection = false) {
        return timed(confirmation_seconds, [&] {
            DeviceMeshView candidate;
            if (!state.trial(action, proposal.placement, candidate)) {
                if (allow_geometry_rejection)
                    return Confirmation{.known = true};
                throw std::runtime_error("teacher finalist became invalid");
            }
            Confirmation result;
            result.valid = true;
            audit.with_candidate_rasters(std::span{&candidate, 1}, [&] {
                result.source = audit.evaluate(source, candidate, bounds, e, &stats);
                result.adjacent =
                    !emitted && e.limit == adjacent.limit
                        ? result.source
                        : audit.evaluate(previous_view, candidate, bounds, adjacent, &stats);
                result.known = action_audit_known(result.source, e) &&
                               action_audit_known(result.adjacent, adjacent);
                result.passed = result.source.passed && result.adjacent.passed;
                // A predecessor also has to remain valid at the destination scale.
                if (result.known && result.passed && previous_steps && !emitted) {
                    auto d = audit.evaluate(source, candidate, bounds, destination, &stats);
                    result.known = action_audit_known(d, destination);
                    result.passed = d.passed;
                }
            });
            if (result.known && result.passed) {
                const auto& a = result.source;
                const auto& b = result.adjacent;
                const double margin = std::max({a.error / e.limit, b.error / adjacent.limit,
                                                a.changed_area / e.max_changed_area,
                                                b.changed_area / adjacent.max_changed_area});
                result.choice = {candidate.faces, margin, a.complete && b.complete};
            }
            return result;
        });
    };
    for (uint32_t step = 0; step < states + previous_steps && !unknown && !cancel(); ++step) {
        if (previous_steps && step == previous_steps) {
            const auto predecessor_start = std::chrono::steady_clock::now();
            previous = state.snapshot();
            auto a = audit.evaluate(source, previous.data.view(), bounds, e, &stats),
                 d = audit.evaluate(source, previous.data.view(), bounds, destination, &stats);
            predecessor_seconds +=
                std::chrono::duration<double>(std::chrono::steady_clock::now() - predecessor_start)
                    .count();
            if (!a.complete || !d.complete) {
                unknown = true;
                break;
            }
            if (!a.passed || !d.passed || previous.data.view().triangles() >= source.triangles()) {
                exhausted = true;
                break;
            }
            previous_view = previous.data.view();
            audit.bind_reference(previous_view);
            emitted = true;
            e.screen_size = pixels;
            adjacent.screen_size = pixels;
        }
        double fraction = std::array{.1, .5, .9, .02}[step % 4];
        std::vector<Placement> policy_placements;
        auto rows = timed(proposals_seconds, [&] {
            return state.teacher_actions(condition(e, adjacent.limit, fraction), pool, seed, policy,
                                         policy_candidates ? TeacherSelection::PolicyMixed
                                                           : TeacherSelection::GeometricRandom,
                                         policy_ranking ? &policy_placements : nullptr);
        });
        if (rows.empty()) {
            exhausted = true;
            break;
        }
        size_t first = data.labels.size();
        std::vector<GpuActionState::Proposal> winners;
        std::vector<Action> actions;
        std::vector<TeacherChoice> core_choices;
        json state_trace = {{"revision", step},
                            {"triangles", state.view().faces},
                            {"previous_triangles", previous_view.triangles()},
                            {"pixels", e.screen_size},
                            {"queries", json::array()}};
        for (auto& row : rows) {
            if (architecture == conditioned_placement_schema)
                row.x[79] = float(preserve_uv);
            if (policy_ranking) {
                const GpuActionState::Proposal proposal{policy_placements[actions.size()], {}, 0};
                const auto result = confirm(row.action, proposal, true);
                const uint8_t label = result.valid && result.known ? result.label() : 0;
                append_geometry_rejection(geometry_rejections, data.labels.size(), !result.valid);
                data.x.insert(data.x.end(), row.x.begin(), row.x.end());
                data.labels.push_back(label);
                data.from.push_back(row.action.from);
                data.to.push_back(row.action.to);
                data.targets.resize(data.labels.size() * 9, 0.f);
                actions.push_back(row.action);
                winners.push_back(proposal);
                core_choices.push_back(result.choice);
                invalid_candidates += !result.valid;
                queries += result.valid;
                json observation = {{"from", row.action.from},
                                    {"to", row.action.to},
                                    {"candidate", "policy"},
                                    {"known_mask", label},
                                    {"geometry_rejected", !result.valid},
                                    {"placement",
                                     {proposal.placement.position.x, proposal.placement.position.y,
                                      proposal.placement.position.z}}};
                if (result.valid) {
                    observation["source"] = measurement_json(result.source);
                    observation["adjacent"] = measurement_json(result.adjacent);
                }
                state_trace["queries"].push_back(std::move(observation));
                unknown |= !result.known;
                if (unknown || cancel())
                    break;
                continue;
            }
            bool found = false;
            uint8_t best_label = 0;
            double best = INFINITY;
            GpuActionState::Proposal winner{};
            TeacherChoice confirmed_choice{};
            auto alternatives = timed(proposals_seconds, [&] {
                return state.teacher_proposals(row.action, profile != Profile::Coverage);
            });
            struct Query {
                TeacherAudit source, adjacent;
                uint32_t faces{};
                bool valid{}, pruned{};
                uint8_t label{};
                double margin{INFINITY};
            };
            std::array<Query, 21> evaluated{};
            std::array<bool, 21> ready{};
            uint32_t queried_mask = 0;
            static_assert(sizeof(Placement) == 9 * sizeof(float));
            auto same_placement = [&](size_t a, size_t b) {
                return !std::memcmp(&alternatives[a].placement, &alternatives[b].placement,
                                    sizeof(Placement));
            };
            auto batch_candidates = [&](std::span<const size_t> pending, double cutoff) {
                size_t count = std::min<size_t>(options.candidate_batch, pending.size());
                for (;;) {
                    std::vector<GpuActionState::Proposal> proposals;
                    std::vector<size_t> ids;
                    for (size_t lane = 0; lane < count; ++lane) {
                        const auto i = pending[lane];
                        bool duplicate = false;
                        for (size_t prior = 0; prior < alternatives.size(); ++prior)
                            if ((core_first ? ready[prior] : prior < i) &&
                                same_placement(i, prior)) {
                                duplicate = true;
                                break;
                            }
                        for (auto prior : ids)
                            duplicate |= same_placement(i, prior);
                        if (!duplicate) {
                            ids.push_back(i);
                            proposals.push_back(alternatives[i]);
                        }
                    }
                    if (proposals.empty())
                        return;
                    try {
                        auto views = timed(trial_seconds, [&] {
                            return state.trial_batch(row.action, proposals);
                        });
                        std::vector<CandidateAudit> a, b;
                        audit.with_candidate_rasters(views, [&] {
                            a = timed(audit_seconds, [&] {
                                Timeline range("candidate-audit");
                                return audit.certify_candidates(source, views, bounds, e, &stats,
                                                                cutoff);
                            });
                            b = a;
                            cancellation.observe(a);
                            if (cancellation.stopped())
                                return;
                            if (emitted || e.limit != adjacent.limit) {
                                std::vector<DeviceMeshView> active;
                                std::vector<size_t> lanes;
                                for (size_t i = 0; i < views.size(); ++i)
                                    if (a[i].valid && !a[i].pruned) {
                                        active.push_back(views[i]);
                                        lanes.push_back(i);
                                    }
                                if (!active.empty()) {
                                    auto values = timed(audit_seconds, [&] {
                                        return audit.certify_candidates(previous_view, active,
                                                                        bounds, adjacent, &stats,
                                                                        cutoff);
                                    });
                                    cancellation.observe(values);
                                    for (size_t i = 0; i < lanes.size(); ++i)
                                        b[lanes[i]] = values[i];
                                }
                            } else
                                reused_adjacent += std::count_if(a.begin(), a.end(), [](auto& x) {
                                    return x.valid && !x.pruned;
                                });
                        });
                        auto value = [](const CandidateAudit& x) {
                            return TeacherAudit{x.value.verdict != AuditVerdict::Unknown,
                                                x.value.verdict == AuditVerdict::Pass,
                                                x.value.error_upper, x.value.changed_area};
                        };
                        for (size_t i = 0; i < ids.size(); ++i) {
                            evaluated[ids[i]] = {value(a[i]), value(b[i]), a[i].faces, a[i].valid,
                                                 a[i].pruned || b[i].pruned};
                            ready[ids[i]] = true;
                            queried_mask |= 1u << ids[i];
                        }
                        ++stats.candidate_batches;
                        stats.candidate_batch_proposals += proposals.size();
                        return;
                    } catch (const gpu::ResourceError&) {
                        if (count <= 1)
                            return;
                        count = (count + 1) / 2;
                    }
                }
            };
            auto evaluate_candidate = [&](size_t index, double cutoff,
                                          std::span<const size_t> pending) -> Query& {
                queried_mask |= 1u << index;
                auto& proposal = alternatives[index];
                auto& q = evaluated[index];
                size_t prior = 0;
                while (prior < alternatives.size() &&
                       (!(core_first ? ready[prior] && prior != index : prior < index) ||
                        !same_placement(index, prior)))
                    ++prior;
                if (prior < alternatives.size()) {
                    q = evaluated[prior];
                    ++reused_queries;
                } else {
                    if (!ready[index] && index && sparse &&
                        options.raster_backend == NeuralRasterBackend::Vulkan &&
                        options.candidate_batch > 1)
                        batch_candidates(pending, cutoff);
                    // No validity result is available when a batch stops before
                    // its first camera. Keep that stop out of geometry counts.
                    if (cancellation.stopped())
                        return q;
                    if (ready[index]) {
                        if (std::isfinite(cutoff) &&
                            ((e.max_changed_area > 0 &&
                              q.source.changed_area / e.max_changed_area >= cutoff) ||
                             (adjacent.max_changed_area > 0 &&
                              q.adjacent.changed_area / adjacent.max_changed_area >= cutoff)))
                            q.pruned = true;
                    } else {
                        DeviceMeshView candidate;
                        q.valid = timed(trial_seconds, [&] {
                            return state.trial(row.action, proposal.placement, candidate);
                        });
                        if (q.valid) {
                            q.faces = candidate.faces;
                            audit.with_candidate_rasters(std::span{&candidate, 1}, [&] {
                                q.source = query(source, candidate, e, cutoff,
                                                 std::isfinite(cutoff) ? &q.pruned : nullptr);
                                if (!q.pruned) {
                                    if (!emitted && e.limit == adjacent.limit) {
                                        q.adjacent = q.source;
                                        ++reused_adjacent;
                                    } else
                                        q.adjacent =
                                            query(previous_view, candidate, adjacent, cutoff,
                                                  std::isfinite(cutoff) ? &q.pruned : nullptr);
                                }
                            });
                        }
                    }
                }
                ready[index] = true;
                if (!q.valid) {
                    ++invalid_candidates;
                    state_trace["invalid_candidates"] =
                        state_trace.value("invalid_candidates", 0u) + 1;
                    return q;
                }
                auto& a = q.source;
                auto& b = q.adjacent;
                ++queries;
                if (q.pruned) {
                    ++pruned_queries;
                    state_trace["queries"].push_back({{"from", row.action.from},
                                                      {"to", row.action.to},
                                                      {"candidate", index},
                                                      {"known_mask", 0},
                                                      {"pruned_by_incumbent", true},
                                                      {"faces", q.faces}});
                    return q;
                }
                bool ka = a.known, kb = b.known;
                q.label = uint8_t((ka ? SourceKnown : 0) | (kb ? AdjacentKnown : 0) |
                                  (ka && a.passed ? PlacementSourcePass : 0) |
                                  (kb && b.passed ? PlacementAdjacentPass : 0));
                q.margin = std::max({a.error / e.limit, b.error / adjacent.limit,
                                     a.changed_area / e.max_changed_area,
                                     b.changed_area / adjacent.max_changed_area});
                if (sparse && (q.label & 27) == 27)
                    q.margin = std::max(a.changed_area / e.max_changed_area,
                                        b.changed_area / adjacent.max_changed_area);
                if (!std::isfinite(q.margin))
                    q.margin = INFINITY;
                state_trace["queries"].push_back({{"from", row.action.from},
                                                  {"to", row.action.to},
                                                  {"candidate", index},
                                                  {"known_mask", q.label},
                                                  {"source_error", a.error},
                                                  {"adjacent_error", b.error},
                                                  {"faces", q.faces}});
                if (!ka || !kb)
                    unknown = true;
                return q;
            };
            if (!core_first) {
                std::array<size_t, 21> order{};
                for (size_t i = 0; i < alternatives.size(); ++i)
                    order[i] = i;
                for (size_t index = 0; index < alternatives.size() && !cancel(); ++index) {
                    // Identical topology and nonnegative coverage-area bounds
                    // make the first certified zero unbeatable for this edge.
                    if (sparse && found && (best_label & 27) == 27 && best == 0) {
                        proven_losers += alternatives.size() - index;
                        break;
                    }
                    const double cutoff = found && (best_label & 27) == 27 ? best : INFINITY;
                    const auto& q = evaluate_candidate(
                        index, cutoff,
                        std::span(order).subspan(index, alternatives.size() - index));
                    if (!q.valid || q.pruned)
                        continue;
                    const bool safe = (q.label & 27) == 27;
                    if (!found || (safe && ((best_label & 3) != 3)) ||
                        (safe == ((best_label & 3) == 3) && q.margin < best)) {
                        found = true;
                        winner = alternatives[index];
                        best_label = q.label;
                        best = q.margin;
                    }
                    if (unknown)
                        break;
                }
                state_trace["candidate_search"].push_back({{"from", row.action.from},
                                                           {"to", row.action.to},
                                                           {"queried_mask", queried_mask}});
            } else {
                CoverageTeacherSearch search(alternatives.size());
                std::optional<uint8_t> selected, last_rejected;
                uint8_t rejected_label = 0;
                while (!unknown && !cancel()) {
                    while (auto index = search.next()) {
                        std::array<size_t, 11> pending{};
                        size_t count = 0;
                        for (auto id : CoverageTeacherSearch::order)
                            if (search.pending() & (1u << id))
                                pending[count++] = id;
                        auto& q = evaluate_candidate(*index, sparse ? search.cutoff() : INFINITY,
                                                     std::span(pending).first(count));
                        search.observe(*index,
                                       {q.margin, q.valid, q.source.known && q.adjacent.known,
                                        (q.label & 27) == 27, q.pruned});
                        if (unknown || cancel())
                            break;
                    }
                    if (unknown || cancel())
                        break;
                    selected = search.best();
                    if (!selected || !search.safe(*selected)) {
                        if (search.expand())
                            continue;
                        break;
                    }
                    const auto result = confirm(row.action, alternatives[*selected]);
                    unknown = !result.known || (result.passed && !result.choice.confirmed);
                    state_trace["confirmations"].push_back(
                        {{"from", row.action.from},
                         {"to", row.action.to},
                         {"candidate", *selected},
                         {"known", !unknown},
                         {"passed", result.passed && !unknown},
                         {"source_passed", result.source.passed},
                         {"adjacent_passed", result.adjacent.passed}});
                    if (unknown || cancel())
                        break;
                    if (result.passed) {
                        confirmed_choice = result.choice;
                        break;
                    }
                    last_rejected = selected;
                    rejected_label = result.label();
                    const auto invalidated = search.reject(*selected);
                    selected.reset();
                    for (size_t i = 0; i < alternatives.size(); ++i)
                        if (invalidated & (1u << i)) {
                            evaluated[i] = {};
                            ready[i] = false;
                        }
                    // Batch lanes may have been prefetched but not consumed
                    // after a zero bound. Their prunes have the same dependency.
                    for (size_t i = 0; i < alternatives.size(); ++i)
                        if (evaluated[i].pruned) {
                            evaluated[i] = {};
                            ready[i] = false;
                        }
                }
                state_trace["candidate_search"].push_back(
                    {{"from", row.action.from},
                     {"to", row.action.to},
                     {"queried_mask", queried_mask},
                     {"exact_rejected_mask", search.rejected()},
                     {"expanded", search.expanded()}});
                if (selected || last_rejected) {
                    found = true;
                    const auto index = selected ? *selected : *last_rejected;
                    winner = alternatives[index];
                    best_label = selected ? evaluated[index].label : rejected_label;
                }
            }
            if (found) {
                bool safe = (best_label & 27) == 27;
                if (safe)
                    best_label |=
                        uint8_t(PositionKnown | (winner.normal_mask & 1 ? Normal0Known : 0) |
                                (winner.normal_mask & 2 ? Normal1Known : 0));
                data.x.insert(data.x.end(), row.x.begin(), row.x.end());
                data.labels.push_back(best_label);
                data.from.push_back(row.action.from);
                data.to.push_back(row.action.to);
                for (unsigned j = 0; j < 9; ++j)
                    data.targets.push_back((best_label & (32u << (j / 3))) ? winner.target[j] : 0);
                actions.push_back(row.action);
                winners.push_back(winner);
                core_choices.push_back(confirmed_choice);
            }
            if (unknown || cancel())
                break;
        }
        if (data.labels.size() > first) {
            data.offsets.push_back(uint32_t(data.labels.size()));
            data.progress.push_back(float(1 - double(state.view().faces) / source.triangles()));
        }
        if (unknown || cancel()) {
            state_trace["complete"] = false;
            trace.push_back(state_trace);
            break;
        }
        std::vector<TeacherChoice> choices = core_first || policy_ranking
                                                 ? core_choices
                                                 : std::vector<TeacherChoice>(actions.size());
        for (size_t i = 0; !core_first && !policy_ranking && i < actions.size() && !cancel(); ++i) {
            auto& label = data.labels[first + i];
            if ((label & 27) != 27)
                continue;
            const auto result = confirm(actions[i], winners[i]);
            if (!result.known) {
                unknown = true;
                break;
            }
            if (!result.source.passed || !result.adjacent.passed) {
                label = result.label();
                std::fill_n(data.targets.begin() + (first + i) * 9, 9, 0.f);
                continue;
            }
            choices[i] = result.choice;
        }
        if (unknown || cancel()) {
            state_trace["complete"] = false;
            state_trace["exact_confirmation_incomplete"] = true;
            trace.push_back(state_trace);
            break;
        }
        auto preferred = preferred_actions(choices);
        if (!preferred) {
            exhausted = true;
            state_trace["stopped"] = "no independently confirmed queried placement";
            trace.push_back(state_trace);
            break;
        }
        auto chosen = std::countr_zero(preferred);
        state_trace["preferred"] = json::array();
        for (size_t i = 0; i < choices.size(); ++i)
            if (preferred & (1u << i)) {
                data.labels[first + i] |= PlacementPreferred;
                state_trace["preferred"].push_back({{"from", actions[i].from},
                                                    {"to", actions[i].to},
                                                    {"margin", choices[i].margin},
                                                    {"triangles", choices[i].faces}});
            }
        timed(commit_seconds, [&] {
            state.commit(actions[chosen], winners[chosen].placement);
            return true;
        });
        ++accepted;
        state_trace["selected"] = {{"from", actions[chosen].from},
                                   {"to", actions[chosen].to},
                                   {"triangles", state.view().faces}};
        trace.push_back(state_trace);
        std::cout << json({{"state", step},
                           {"queries", queries},
                           {"triangles", state.view().faces},
                           {"seconds", seconds()}})
                         .dump()
                  << std::endl;
    }
    const auto final_audit_start = std::chrono::steady_clock::now();
    auto final = state.view();
    Measurement a, b;
    audit.with_candidate_rasters(std::span{&final, 1}, [&] {
        a = audit.evaluate(source, final, bounds, e, &stats);
        b = !emitted && e.limit == adjacent.limit
                ? a
                : audit.evaluate(previous_view, final, bounds, adjacent, &stats);
    });
    const bool reference_confirmed = audit_measurement_passed(a) && audit_measurement_passed(b);
    bool complete = !cancel() && !unknown &&
                    (data.states() == states + previous_steps || exhausted) && reference_confirmed;
    final_audit_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - final_audit_start).count();
    const auto output_start = std::chrono::steady_clock::now();
    CompactActions encoded;
    if (compact_data) {
        encoded = compact_actions(data);
        std::ofstream f(output / "actions.bin", std::ios::binary);
        write_compact(f, encoded);
        f.close();
        if (!f)
            throw std::runtime_error("compact teacher output failed");
    } else
        save_actions(output / "actions.bin", data, false);
    if (complete) {
        auto end = state.snapshot();
        std::ofstream f(output / "episode.bin", std::ios::binary);
        write_packed_mesh(f, end.view(source), &quantization);
        f.close();
        if (!f)
            throw std::runtime_error("episode output failed");
    }
    write_json(output / "trajectory.json", trace);
    write_json(output / "reuse.json", {{"duplicate_proposals", reused_queries},
                                       {"identical_adjacent_audits", reused_adjacent},
                                       {"pruned_candidates", pruned_queries},
                                       {"unbeatable_incumbent_skips", proven_losers}});
    json index = {{"schema", architecture},
                  {"status", complete   ? (previous_steps && !emitted ? "predecessor_unavailable"
                                           : exhausted                ? "search_exhausted"
                                                                      : "complete")
                             : cancel() ? "cancelled"
                             : (a.resource_limited || b.resource_limited || stats.resource_failures)
                                 ? "resource_failure"
                             : unknown ? "unknown_audit"
                                       : "final_audit_failed"},
                  {"baseline", measurement_json(baseline.final)},
                  {"source_audit", measurement_json(a)},
                  {"adjacent_audit", measurement_json(b)},
                  {"complete", complete},
                  {"asset", asset},
                  {"category", metadata.at("category")},
                  {"contract_sha256", file_sha256(output / "contract.json")},
                  {"path", "actions.bin"},
                  {"sha256", file_sha256(output / "actions.bin")},
                  {"states", data.states()},
                  {"queries", queries},
                  {"accepted", accepted},
                  {"source_triangles", source.triangles()},
                  {"teacher_triangles", final.faces},
                  {"reference_confirmed", reference_confirmed},
                  {"requested_condition_available", !previous_steps || emitted},
                  {"preceding_lod_emitted", emitted},
                  {"previous_triangles", previous_view.triangles()},
                  {"seconds", seconds()},
                  {"training_started", false},
                  {"audit", neural_json(stats)}};
    index["seed"] = seed_result;
    if (policy_ranking) {
        std::ofstream f(output / "geometry-rejected.bin", std::ios::binary);
        f.write("BLZRANK1", 8);
        write_vector(f, geometry_rejections);
        f.close();
        if (!f)
            throw std::runtime_error("policy geometry bitmap write failed");
        index["geometry_rejected_sha256"] = file_sha256(output / "geometry-rejected.bin");
        auto ranking = data.labels;
        apply_policy_rank_masks(ranking, geometry_rejections);
        uint32_t informative = 0, pairs = 0;
        for (size_t s = 0; s < data.states(); ++s) {
            const auto count = ranking_pairs(
                std::span(ranking).subspan(data.offsets[s], data.offsets[s + 1] - data.offsets[s]));
            informative += count != 0;
            pairs += count;
        }
        index["ranking_states"] = informative;
        index["ranking_pairs"] = pairs;
    }
    index["invalid_candidates"] = invalid_candidates;
    if (complete)
        index["episode_sha256"] = file_sha256(output / "episode.bin");
    output_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - output_start).count();
    record_timings(index);
    write_json(output / "index.json", index);
    return {complete, std::move(data), std::move(index), std::move(encoded)};
}
} // namespace blitz::neural::training
