#include "../training/packed_data.hpp"
#include "chain_hooks.hpp"
#include "neural/action.hpp"
#include "neural/action_cache.hpp"
#include "neural/audit_cache.hpp"
#include "neural/audit_measurement.hpp"
#include "training/policy_ranking.hpp"
#include <iostream>
#include <numeric>
using namespace blitz;
using namespace blitz::neural;
void check(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}
void policy_label_contracts() {
    using namespace blitz::neural::training;
    std::vector<uint8_t> visual{31, 27, 0, 0, 8, 24, 31, 27, 0};
    std::vector<uint8_t> rejected;
    for (size_t i = 0; i < visual.size(); ++i)
        append_geometry_rejection(rejected, i, i == 2 || i == 8);
    apply_policy_rank_masks(visual, rejected);
    check(visual[2] == 24 && visual[8] == 24 && visual[3] == 0 && visual[4] == 8,
          "unknown audits became ranking negatives");
    check(ranking_pairs(visual) == 10, "ranking pair count lost honest ties/rejections");
    std::array<uint8_t, 3> tied{31, 31, 0};
    check(!ranking_pairs(tied), "honest ties gained artificial ranking supervision");
    auto rejects = [&](auto&& call, const char* reason) {
        bool rejected = false;
        try {
            call();
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected, reason);
    };
    check(ranking_observation_step(0, 8, 3) == 0 && ranking_observation_step(1, 8, 3) == 3 &&
              ranking_observation_step(2, 8, 3) == 7,
          "teacher schedule omitted middle or final trajectory states");
    for (uint32_t iterations : {1u, 4u, 23u})
        for (uint32_t requested : {1u, 7u, 32u}) {
            const auto n = std::min(iterations, requested);
            uint32_t previous = 0;
            for (uint32_t sample = 0; sample < n; ++sample) {
                const auto step = ranking_observation_step(sample, iterations, requested);
                check(step < iterations && (!sample || step > previous),
                      "teacher schedule duplicated or exceeded runtime states");
                previous = step;
            }
            check(n == 1 || previous == iterations - 1, "teacher schedule missed late states");
        }
    rejects([] { ranking_observation_step(0, 0, 4); }, "empty trajectory schedule accepted");
    rejects([] { ranking_observation_step(0, 3, 0); }, "empty observation budget accepted");
    rejects([] { ranking_observation_step(3, 3, 8); }, "out-of-range observation accepted");
    std::array<uint8_t, 1> corrupt{31}, bits{1};
    rejects([&] { apply_policy_rank_masks(corrupt, bits); },
            "geometry rejection accepted fabricated visual labels");
    bits[0] = 128;
    rejects([&] { apply_policy_rank_masks(corrupt, bits); },
            "geometry bitmap accepted out-of-range tail bits");
    rejects([&] { apply_policy_rank_masks(corrupt, {}); }, "missing geometry bitmap accepted");
    corrupt[0] = PositionKnown;
    bits[0] = 0;
    rejects([&] { apply_policy_rank_masks(corrupt, bits); },
            "rank-only data accepted a placement target");
    std::array<uint8_t, 17> oversized{};
    rejects([&] { ranking_pairs(oversized); }, "oversized rank pool accepted");
    apply_policy_rank_masks({}, {});
}
Mesh plane(unsigned n) {
    Mesh m;
    for (unsigned y = 0; y < n; ++y)
        for (unsigned x = 0; x < n; ++x) {
            m.positions.push_back({float(x), float(y), 0});
            m.uv.push_back({float(x), float(y)});
            m.normals.push_back({0, 0, 1});
        }
    for (unsigned y = 0; y + 1 < n; ++y)
        for (unsigned x = 0; x + 1 < n; ++x) {
            uint32_t a = y * n + x;
            m.indices.insert(m.indices.end(), {a, a + 1, a + n, a + 1, a + n + 1, a + n});
        }
    m.double_sided = {1};
    return m;
}
void action_stop_contracts() {
    for (unsigned n : {5u, 7u}) {
        auto mesh = plane(n);
        uint32_t ranks = 0, gates = 0, polls = 0;
        auto rank = [&](const ActionState&, std::span<const ActionRecord> rows) {
            ++ranks;
            return std::vector<float>(rows.size(), 0);
        };
        auto gate = [&](MeshView) {
            ++gates;
            return true;
        };
        auto cancel = [&] {
            ++polls;
            return true;
        };
        ActionStats stats;
        auto reached = execute_actions(mesh.view(), {}, mesh.view().triangles(), 0, rank, gate,
                                       &stats, cancel);
        check(stats.stop_reason == NeuralActionStop::TargetReached && !stats.trials && !ranks &&
                  !gates && !polls && same_mesh_data(reached.view(mesh.view()), mesh.view()),
              "already-reached target did work or lost precedence over zero budget");
        auto exhausted = execute_actions(mesh.view(), {}, 1, 0, rank, gate, &stats, cancel);
        check(stats.stop_reason == NeuralActionStop::TrialBudget && !stats.trials && !ranks &&
                  !gates && !polls && same_mesh_data(exhausted.view(mesh.view()), mesh.view()),
              "zero trial budget did work or reported the wrong stop reason");
        auto cancelled = execute_actions(mesh.view(), {}, 1, 4, rank, gate, &stats, cancel);
        check(stats.stop_reason == NeuralActionStop::Cancelled && !stats.trials && !ranks &&
                  !gates && polls == 1 && same_mesh_data(cancelled.view(mesh.view()), mesh.view()),
              "immediate cancellation did work or reported the wrong stop reason");
        polls = 0;
        auto one_shot = execute_actions(mesh.view(), {}, 1, 4, rank, gate, &stats,
                                        [&] { return ++polls == 3; });
        check(stats.stop_reason == NeuralActionStop::Cancelled && !stats.trials && ranks == 1 &&
                  !gates && polls == 3 && same_mesh_data(one_shot.view(mesh.view()), mesh.view()),
              "one-shot cancellation resumed candidate execution");
        ActionState state(mesh.view());
        auto rows = state.actions({});
        auto first = state.trial(rows.front().action);
        auto target = first.view(mesh.view()).triangles();
        auto accepted = execute_actions(mesh.view(), {}, target, 1, rank, gate, &stats);
        check(stats.stop_reason == NeuralActionStop::TargetReached && stats.trials == 1 &&
                  stats.accepted == 1 && accepted.data.indices == first.data.indices,
              "accepted target did not take precedence over exhausted trial budget");
        bool committed = false;
        auto incumbent = execute_actions(
            mesh.view(), {}, 1, 4, rank,
            [&](MeshView) {
                committed = true;
                return true;
            },
            &stats, [&] { return committed; });
        check(stats.stop_reason == NeuralActionStop::Cancelled && stats.trials == 1 &&
                  stats.accepted == 1 && incumbent.data.indices == first.data.indices,
              "cancellation lost a previously accepted incumbent");
        auto rejected = execute_actions(
            mesh.view(), {}, 1, uint32_t(rows.size() + 1), rank, [](MeshView) { return false; },
            &stats);
        check(stats.stop_reason == NeuralActionStop::NoAcceptedAction &&
                  stats.trials == rows.size() && !stats.accepted &&
                  same_mesh_data(rejected.view(mesh.view()), mesh.view()),
              "fully rejected legal candidates were reported as budget or legality exhaustion");
    }
    Mesh tetra;
    tetra.positions = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    tetra.indices = {0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3};
    ActionStats stats;
    auto unchanged = execute_actions(
        tetra.view(), {}, 1, 2,
        [](const ActionState&, std::span<const ActionRecord>) -> std::vector<float> {
            throw std::runtime_error("ranked a state with no legal actions");
        },
        [](MeshView) -> bool { throw std::runtime_error("audited a state with no legal actions"); },
        &stats);
    check(stats.stop_reason == NeuralActionStop::NoLegalActions && !stats.trials && !stats.ranked &&
              same_mesh_data(unchanged.view(tetra.view()), tetra.view()),
          "state without legal actions reported the wrong stop reason");
}
int main() {
    try {
        action_stop_contracts();
        policy_label_contracts();
        auto cache_mesh = plane(5);
        cache_mesh.materials.resize(cache_mesh.view().triangles(), 0);
        auto memo_source = cache_mesh.view(), memo_candidate = memo_source;
        std::vector<uint32_t> memo_indices(cache_mesh.indices.begin() + 3,
                                           cache_mesh.indices.end());
        memo_candidate.indices = memo_indices;
        memo_candidate.materials = memo_source.materials.subspan(1);
        auto memo_bounds = bounds(memo_source);
        EvalSettings memo_settings;
        Measurement memo_measurement;
        memo_measurement.error = .25;
        memo_measurement.views_evaluated = 7;
        AuditMemo memo(memo_source);
        memo.insert(memo_source, memo_candidate, memo_bounds, memo_settings, memo_measurement);
        check(memo.find(memo_source, memo_candidate, memo_bounds, memo_settings)->error == .25,
              "completed audit was not retained");
        auto changed_settings = memo_settings;
        changed_settings.max_supersample = 16;
        check(!memo.find(memo_source, memo_candidate, memo_bounds, changed_settings),
              "bounded audit reused for full confirmation");
        auto changed_bounds = memo_bounds;
        changed_bounds.radius *= 2;
        check(!memo.find(memo_source, memo_candidate, changed_bounds, memo_settings),
              "audit ignored source bounds");
        std::swap(memo_indices[0], memo_indices[1]);
        check(!memo.find(memo_source, memo_candidate, memo_bounds, memo_settings),
              "audit cached borrowed topology pointers");
        auto incomplete = memo_measurement;
        incomplete.complete = false;
        memo.insert(memo_source, memo_candidate, memo_bounds, memo_settings, incomplete);
        check(!memo.find(memo_source, memo_candidate, memo_bounds, memo_settings),
              "incomplete audit became acceptance");
        auto different_attributes = cache_mesh;
        auto different_view = different_attributes.view();
        different_view.indices = memo_candidate.indices;
        check(!memo.find(memo_source, different_view, memo_bounds, memo_settings),
              "audit reused mutable unrelated attributes");
        AuditMemo tiny_memo(memo_source, 1);
        tiny_memo.insert(memo_source, memo_candidate, memo_bounds, memo_settings, memo_measurement);
        check(!tiny_memo.find(memo_source, memo_candidate, memo_bounds, memo_settings),
              "audit memo exceeded byte cap");
        for (auto bytes : {cache_mesh.indices.size() * sizeof(uint32_t),
                           cache_mesh.indices.size() * sizeof(uint32_t) * 2}) {
            ActionRejections cache(bytes);
            EvalSettings e;
            cache.configure(e);
            cache.insert(cache_mesh.view());
            check(cache.contains(cache_mesh.view()) ==
                      (bytes >= cache_mesh.indices.size() * sizeof(uint32_t) +
                                    cache_mesh.materials.size() * sizeof(uint16_t)),
                  "cache payload bound changed verdict");
        }
        ActionRejections cache;
        EvalSettings cache_settings;
        cache.configure(cache_settings);
        cache.insert(cache_mesh.view());
        check(cache.contains(cache_mesh.view()), "identical rejection was not reused");
        auto altered = cache_mesh;
        altered.materials[0] = 1;
        check(!cache.contains(altered.view()), "cache ignored material identity");
        altered = cache_mesh;
        std::swap(altered.indices[0], altered.indices[1]);
        check(!cache.contains(altered.view()), "cache ignored index identity");
        for (const auto& change : std::array<std::function<void(EvalSettings&)>, 4>{
                 [](auto& e) { e.screen_size = 64; }, [](auto& e) { e.limit = 5; },
                 [](auto& e) { e.weights.normal = 0; }, [](auto& e) { ++e.views.rotation_seed; }}) {
            cache.configure(cache_settings);
            cache.insert(cache_mesh.view());
            auto next = cache_settings;
            change(next);
            cache.configure(next);
            check(!cache.contains(cache_mesh.view()), "cache reused a different audit contract");
        }
        for (double limit : {1.5, 3.5}) {
            EvalSettings e;
            e.limit = limit;
            e.max_changed_area = .25;
            Measurement m;
            check(action_audit_known(m, e), "complete pass lacks a label");
            m.complete = false;
            m.passed = false;
            check(!action_audit_known(m, e), "cancelled audit received a negative label");
            m.views_evaluated = 3;
            m.error = limit;
            check(!action_audit_known(m, e), "partial passing views received a negative label");
            m.error = limit + .5;
            check(action_audit_known(m, e), "witnessed pixel failure was discarded");
            m.cancelled = true;
            check(!action_audit_known(m, e), "one-shot cancellation became a negative label");
            m.cancelled = false;
            m.error = limit;
            m.changed_area = .5;
            check(action_audit_known(m, e), "witnessed area failure was discarded");
            m.resource_limited = true;
            check(!action_audit_known(m, e), "resource failure received a label");
            m.resource_limited = false;
            m.error = NAN;
            check(!action_audit_known(m, e), "NaN audit received a label");
            // Empty/nonempty render disagreement can be a witnessed infinite
            // distance. Keep that negative without accepting invalid diagnostics.
            m = {};
            m.passed = m.complete = false;
            m.views_evaluated = 1;
            m.error = m.coverage = m.coverage_upper = INFINITY;
            check(action_audit_known(m, e) && !audit_measurement_passed(m),
                  "witnessed infinite distance lost its negative label");
            for (const auto corrupt : std::array<std::function<void(Measurement&)>, 5>{
                     [](auto& q) { q.error = NAN; }, [](auto& q) { q.coverage = NAN; },
                     [](auto& q) { q.coverage_upper = NAN; },
                     [](auto& q) { q.changed_area = INFINITY; },
                     [](auto& q) { q.normal_degrees = INFINITY; }}) {
                auto invalid = m;
                corrupt(invalid);
                check(!action_audit_known(invalid, e) && !audit_measurement_passed(invalid),
                      "invalid diagnostic became a known negative");
                invalid = {};
                corrupt(invalid);
                check(!action_audit_known(invalid, e) && !audit_measurement_passed(invalid),
                      "invalid diagnostic became a positive label");
            }
            for (auto field :
                 {&Measurement::error, &Measurement::coverage, &Measurement::coverage_upper}) {
                Measurement invalid;
                invalid.*field = INFINITY;
                check(!action_audit_known(invalid, e) && !audit_measurement_passed(invalid),
                      "infinite distance was accepted as a positive label");
            }
            m = {};
            m.resource_limited = true;
            check(!audit_measurement_passed(m) && !action_audit_known(m, e),
                  "resource flag was ignored on an otherwise passing measurement");
        }
        for (unsigned n : {5u, 7u}) {
            auto m = plane(n);
            auto source = copy_mesh(m.view());
            ActionState state(m.view());
            auto actions = state.actions({});
            check(!actions.empty(), "plane lacks legal actions");
            {
                auto rows = state.actions({.5f, .1f, .3f, .4f, 0, 1, .25f, .6f});
                std::vector<float> features;
                for (size_t i = 0; i < 3; ++i)
                    features.insert(features.end(), rows[i].x.begin(), rows[i].x.end());
                std::vector<uint8_t> labels{15, 8, 11};
                auto packed = training::pack_actions(features, labels, 1, 3, 80);
                for (size_t i = 0; i < features.size(); ++i) {
                    auto c = uint32_t(i % 80);
                    int slot = training::feature_slot(c);
                    float value = slot >= 0 ? packed.values[i / 80 * 52 + unsigned(slot)]
                                  : slot >= -32
                                      ? float((packed.flags[i / 80] >> unsigned(-slot - 1)) & 1)
                                      : packed.conditions[unsigned(-slot - 33)];
                    check(std::bit_cast<uint32_t>(value) == std::bit_cast<uint32_t>(features[i]),
                          "packed action changed feature bits");
                }
                auto bad = features;
                bad[78] += 1;
                bool rejected = false;
                try {
                    training::pack_actions(bad, labels, 1, 3, 80);
                } catch (const std::invalid_argument&) {
                    rejected = true;
                }
                check(rejected, "inconsistent duplicate feature was lost");
                bad = features;
                bad[80 + 48] += 1;
                rejected = false;
                try {
                    training::pack_actions(bad, labels, 1, 3, 80);
                } catch (const std::invalid_argument&) {
                    rejected = true;
                }
                check(rejected, "inconsistent state condition was lost");
            }
            auto action = actions[actions.size() / 2].action;
            auto before = state.lod().data.indices;
            auto trial = state.trial(action);
            check(state.lod().data.indices == before, "trial mutated current mesh");
            check(trial.shared_vertices && trial.data.positions.empty(),
                  "trial copied borrowed vertices");
            check(trial.view(m.view()).triangles() < m.view().triangles(),
                  "legal action did not reduce");
            check(uv_distortion(trial.view(m.view())).negative_uv_faces == 0, "action flipped UVs");
            state.commit(action);
            check(state.lod().data.indices == trial.data.indices, "commit differs from trial");
            check(!state.legal(action), "stale action remained legal");
            bool stale = false;
            try {
                state.trial(action);
            } catch (const std::invalid_argument&) {
                stale = true;
            }
            check(stale, "stale action executed");
            check(same_mesh_data(m.view(), source.view()), "action modified source streams");
            uint32_t calls = 0;
            ActionStats stats;
            auto rank = [&](const ActionState&, std::span<const ActionRecord> rows) {
                ++calls;
                return std::vector<float>(rows.size(), 0);
            };
            auto reduced =
                execute_actions(m.view(), {}, 1, 4, rank, [](MeshView) { return true; }, &stats);
            check(stats.accepted == 4 && calls == 4 &&
                      stats.stop_reason == NeuralActionStop::TrialBudget,
                  "executor reused scores after changing topology");
            check(reduced.view(m.view()).triangles() < m.view().triangles(),
                  "executor did not reduce");
            auto rejected =
                execute_actions(m.view(), {}, 1, 3, rank, [](MeshView) { return false; }, &stats);
            check(stats.trials == 3 && stats.accepted == 0 &&
                      stats.stop_reason == NeuralActionStop::TrialBudget &&
                      same_mesh_data(rejected.view(m.view()), m.view()),
                  "rejected transaction changed incumbent or work budget");
            auto cancelled = execute_actions(
                m.view(), {}, 1, 5, rank, [](MeshView) { return true; }, &stats,
                [] { return true; });
            check(stats.trials == 0 && same_mesh_data(cancelled.view(m.view()), m.view()),
                  "cancellation changed source");
            bool nonfinite = false;
            try {
                execute_actions(
                    m.view(), {}, 1, 2,
                    [](const ActionState&, std::span<const ActionRecord> rows) {
                        return std::vector<float>(rows.size(), NAN);
                    },
                    [](MeshView) { return true; });
            } catch (const std::invalid_argument&) {
                nonfinite = true;
            }
            check(nonfinite, "nonfinite action scores accepted");
        }
        auto seam = plane(5);
        std::vector<uint32_t> duplicate(25, UINT32_MAX);
        for (uint32_t batch_size : {2u, 4u}) {
            auto m = plane(9);
            ActionState s(m.view());
            auto rows = s.actions({});
            std::vector<uint32_t> order(rows.size());
            std::iota(order.begin(), order.end(), 0);
            auto batch = s.independent(rows, order, batch_size, m.view().triangles() - 1);
            check(batch.size() > 1, "independent batch lacks parallel actions");
            auto trial = s.trial(batch);
            auto source = copy_mesh(m.view());
            check(same_mesh_data(s.view(), source.view()), "batch trial mutated incumbent");
            s.commit(batch);
            check(s.lod().data.indices == trial.data.indices,
                  "batch commit differs from audited trial");
            bool stale = false;
            try {
                s.trial(batch);
            } catch (const std::invalid_argument&) {
                stale = true;
            }
            check(stale, "stale batch executed");
            ActionStats stats;
            uint32_t calls = 0;
            auto rank = [&](const ActionState&, std::span<const ActionRecord> a) {
                ++calls;
                return std::vector<float>(a.size(), 0);
            };
            auto combined = execute_actions(
                m.view(), {}, 1, 3, rank, [](MeshView) { return true; }, &stats, {}, batch_size);
            check(stats.accepted > stats.trials && calls == stats.trials,
                  "batch did not share audits or re-rank after commit");
            check(combined.view(m.view()).triangles() < m.view().triangles(),
                  "batch failed to reduce");
            auto limited = execute_actions(
                m.view(), {}, m.view().triangles() - 3, 10, rank, [](MeshView) { return true; },
                &stats, {}, batch_size);
            check(limited.view(m.view()).triangles() >= m.view().triangles() - 3,
                  "batch undershot triangle target");
            auto backed = execute_actions(
                m.view(), {}, 1, 8, rank,
                [&](MeshView candidate) {
                    return candidate.triangles() + 2 >= m.view().triangles();
                },
                &stats, {}, batch_size);
            check(stats.accepted > 0 && stats.rejected > 0 &&
                      backed.view(m.view()).triangles() + 2 >= m.view().triangles(),
                  "batch backoff lost the last audited incumbent");
            ActionState overlap(m.view());
            std::array<Action, 2> duplicate_actions{rows[0].action, rows[0].action};
            bool rejected = false;
            try {
                overlap.trial(duplicate_actions);
            } catch (const std::invalid_argument&) {
                rejected = true;
            }
            check(rejected, "overlapping batch footprints accepted");
        }
        for (uint32_t y = 0; y < 5; ++y) {
            auto i = y * 5 + 2;
            duplicate[i] = uint32_t(seam.positions.size());
            seam.positions.push_back(seam.positions[i]);
            seam.normals.push_back({0, 1, 0});
            seam.uv.push_back({-2, float(y)});
        }
        for (uint32_t f = 0; f < seam.view().triangles(); ++f) {
            bool right = false;
            for (unsigned j = 0; j < 3; ++j)
                right |= seam.positions[seam.indices[f * 3 + j]].x > 2;
            seam.materials.push_back(uint16_t(right));
            if (right)
                for (unsigned j = 0; j < 3; ++j) {
                    auto& i = seam.indices[f * 3 + j];
                    if (duplicate[i] != UINT32_MAX)
                        i = duplicate[i];
                }
        }
        seam.double_sided = {1, 1};
        ActionState charts(seam.view());
        auto chart_actions = charts.actions({});
        auto found = std::find_if(chart_actions.begin(), chart_actions.end(), [](auto& row) {
            return row.action.from == 12 && row.action.to == 17;
        });
        check(found != chart_actions.end(), "unambiguous coupled seam locked");
        auto collapsed = charts.trial(found->action);
        auto out = collapsed.view(seam.view());
        check(std::find(out.indices.begin(), out.indices.end(), 12) == out.indices.end() &&
                  std::find(out.indices.begin(), out.indices.end(), duplicate[12]) ==
                      out.indices.end(),
              "seam collapsed on one chart only");
        check(std::find(out.indices.begin(), out.indices.end(), 17) != out.indices.end() &&
                  std::find(out.indices.begin(), out.indices.end(), duplicate[17]) !=
                      out.indices.end(),
              "seam destination attributes merged");
        for (uint32_t f = 0; f < out.triangles(); ++f)
            for (unsigned j = 0; j < 3; ++j) {
                auto i = out.indices[f * 3 + j];
                if (out.positions[i].x == 2)
                    check(out.material(f) == uint16_t(i >= 25),
                          "wedge moved across material chart");
            }
        Mesh tetra;
        tetra.positions = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        tetra.indices = {0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3};
        check(ActionState(tetra.view()).actions({}).empty(), "tetrahedron creates duplicate faces");
        auto nonmanifold = plane(5);
        nonmanifold.indices.insert(nonmanifold.indices.end(), {6, 7, 12, 6, 7, 17});
        ActionState unsafe(nonmanifold.view());
        check(!unsafe.legal({6, 7, 0}), "nonmanifold edge unlocked");
        Mesh bow;
        bow.positions = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {-1, 0, 0}, {0, -1, 0}};
        bow.indices = {0, 1, 2, 0, 3, 4};
        ActionState bow_state(bow.view());
        for (const auto& row : bow_state.actions({}))
            check(row.action.from != 0 && row.action.to != 0,
                  "disconnected coincident fan unlocked");
        auto chain_mesh = plane(7);
        Settings settings;
        settings.levels = 3;
        settings.base_pixels = 32;
        settings.last_pixels = 16;
        settings.candidate_budget = 2;
        settings.beam_width = 1;
        settings.research.output = OutputMode::Reuse;
        settings.research.chain = ChainMode::Direct;
        settings.search_views = {2, 1, 347};
        settings.audit_views = {4, 2, 349};
        settings.search_supersample = 1;
        settings.audit_supersample = 2;
        settings.max_changed_area = .37;
        bool saw_previous = false;
        detail::GenerationHooks hooks;
        hooks.propose_guarded = [&](MeshView input, MeshView fixed, MeshView previous,
                                    const Bounds&, const ReduceSettings& rs, const EvalSettings& a,
                                    const EvalSettings& b, const EvalSettings& search_a,
                                    const EvalSettings& search_b) {
            check(same_mesh_data(fixed, chain_mesh.view()), "guarded source drifted");
            check(same_mesh_data(input, fixed), "direct proposal changed origin");
            saw_previous |= previous.triangles() < fixed.triangles();
            check(a.screen_size == b.screen_size, "source/adjacent cameras differ");
            check(search_a.views.rotation_seed == settings.search_views.rotation_seed &&
                      search_b.views.rotation_seed == settings.search_views.rotation_seed &&
                      a.views.rotation_seed == settings.audit_views.rotation_seed &&
                      b.views.rotation_seed == settings.audit_views.rotation_seed &&
                      search_a.supersample == settings.search_supersample &&
                      a.supersample == settings.audit_supersample && search_a.limit == a.limit &&
                      search_b.limit == b.limit && search_a.max_changed_area == 1 &&
                      search_b.max_changed_area == 1 &&
                      a.max_changed_area == settings.max_changed_area &&
                      b.max_changed_area == settings.max_changed_area,
                  "guarded action omitted or conflated search and audit gates");
            ActionStats stats;
            return execute_actions(
                input, {}, rs.target_triangles, 4,
                [](const ActionState&, std::span<const ActionRecord> rows) {
                    return std::vector<float>(rows.size(), 0);
                },
                [](MeshView) { return true; }, &stats);
        };
        hooks.evaluate = [](MeshView, MeshView, const Bounds&, const EvalSettings&) {
            return Measurement{};
        };
        hooks.confirm = [](Result&) { return true; };
        auto chain = detail::generate_with_hooks(chain_mesh.view(), settings, {}, &hooks);
        check(chain.lods.size() == 3 && saw_previous, "guarded hook lost preceding emitted LOD");
        std::cout << "action contracts passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
