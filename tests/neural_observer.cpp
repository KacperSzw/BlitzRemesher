#include "neural/cuda.cuh"
#include "neural/generation.hpp"
#include <iostream>

using namespace blitz;
using namespace blitz::neural;
namespace {
void require(bool value, const char* reason) {
    if (!value)
        throw std::runtime_error(reason);
}
Mesh fixture() {
    Mesh mesh;
    for (unsigned y = 0; y < 5; ++y)
        for (unsigned x = 0; x < 5; ++x) {
            mesh.positions.push_back({float(x), float(y), 0});
            mesh.normals.push_back({0, 0, 1});
            mesh.uv.push_back({float(x) / 4, float(y) / 4});
        }
    for (uint32_t y = 0; y < 4; ++y)
        for (uint32_t x = 0; x < 4; ++x) {
            auto a = y * 5 + x;
            mesh.indices.insert(mesh.indices.end(), {a, a + 1, a + 5, a + 1, a + 6, a + 5});
        }
    mesh.double_sided = {1};
    return mesh;
}

void search_guard_contract() {
    // The audit camera sees this plane edge-on. Its passing mask cannot certify
    // the different search views, which see boundary damage from contractions.
    const ViewSet audit_views{1, 0, 351}, search_views{3, 0, 353};
    const auto camera = cameras({{0, 0, 0}, 1}, 32, audit_views).front();
    auto mesh = fixture();
    for (auto& p : mesh.positions)
        p = camera.right * (p.x - 2) + camera.forward * (p.y - 2);
    std::fill(mesh.normals.begin(), mesh.normals.end(), camera.up);
    WeightsData weights;
    weights.architecture = conditioned_placement_schema;
    weights.values.resize(policy_weights(weights.architecture));
    NeuralOptions options;
    options.memory_mib = 128;
    options.vertex_storage = NeuralVertexStorage::Float32;
    options.raster_backend = NeuralRasterBackend::Vulkan;
    options.origin = NeuralOrigin::Source;
    options.action_trials = 64;
    options.action_batch = 4;
    EvalSettings audit;
    audit.profile = Profile::Coverage;
    audit.views = audit_views;
    audit.screen_size = 32;
    audit.limit = 2;
    audit.supersample = 2;
    audit.max_supersample = 4;
    auto search = audit;
    search.views = search_views;
    {
        AuditCuda evaluator(options, mesh.view());
        ActionCuda policy(weights, options);
        GpuActionState state(mesh.view(), options);
        const auto bounds = blitz::bounds(mesh.view());
        auto audit_only = [&](DeviceMeshView candidate) {
            const auto value = evaluator.evaluate(mesh.view(), candidate, bounds, audit);
            return value.complete && value.passed;
        };
        const auto candidate = state.execute(
            condition(audit, audit.limit, .5), mesh.view().triangles() / 2, options.action_trials,
            &policy, NeuralRanking::Learned, 101, options.action_batch, audit_only, nullptr);
        require(
            !evaluator.evaluate(mesh.view(), candidate.view(mesh.view()), bounds, search).passed,
            "fixture no longer distinguishes search from audit acceptance");
    }
    Settings settings;
    settings.levels = 2;
    settings.base_pixels = 64;
    settings.last_pixels = 32;
    settings.profile = Profile::Coverage;
    settings.max_lod0_delta_px = audit.limit;
    settings.transition.points = {{0, float(audit.limit)}, {1, float(audit.limit)}};
    settings.candidate_budget = 2;
    settings.beam_width = 1;
    settings.search_views = search_views;
    settings.audit_views = audit_views;
    settings.search_supersample = settings.audit_supersample = audit.supersample;
    settings.max_supersample = audit.max_supersample;
    settings.research.output = OutputMode::Reuse;
    const auto result = generate_observed(mesh.view(), settings, weights, options, {});
    require(result.status == Status::Complete && result.lods.size() == 2 &&
                result.lods.back().view(mesh.view()).triangles() < mesh.view().triangles(),
            "search-blind proposal discarded useful local reductions");
    require(std::all_of(result.rejected_gates.begin(), result.rejected_gates.end(),
                        [](uint64_t count) { return count == 0; }),
            "committed actions violated an outer visual gate");
}
} // namespace

int main() {
    if (!neural_available())
        return 77;
    try {
        search_guard_contract();
        auto mesh = fixture();
        WeightsData weights;
        weights.architecture = conditioned_placement_schema;
        weights.values.resize(policy_weights(weights.architecture));
        NeuralOptions options;
        options.memory_mib = 128;
        options.vertex_storage = NeuralVertexStorage::Float32;
        options.raster_backend = NeuralRasterBackend::Vulkan;
        options.origin = NeuralOrigin::Both;
        options.action_trials = 12;
        Settings settings;
        settings.levels = 3;
        settings.base_pixels = 32;
        settings.last_pixels = 16;
        settings.profile = Profile::Coverage;
        settings.candidate_budget = 2;
        settings.beam_width = 1;
        settings.search_views = {2, 0, 331};
        settings.audit_views = {3, 0, 337};
        settings.search_supersample = 1;
        settings.audit_supersample = 1;
        settings.max_supersample = 2;
        settings.max_changed_area = 1;
        settings.research.output = OutputMode::Reuse;
        uint64_t sparse_passes = 0, sparse_fallbacks = 0;
        for (bool preserve : {true, false})
            for (uint8_t batch : {1, 16}) {
                options.preserve_uv = preserve;
                options.action_batch = batch;
                NeuralStats before, after;
                uint32_t polls = 0, observations = 0;
                bool predecessor = false;
                settings.cancelled = [&] {
                    ++polls;
                    return false;
                };
                const auto baseline =
                    generate_observed(mesh.view(), settings, weights, options, {}, &before);
                const auto baseline_polls = polls;
                polls = 0;
                ActionCuda policy(weights, options);
                auto observe = [&](GpuActionState& state, const ActionRequest& request) {
                    require(request.output == OutputMode::Reuse, "observer lost execution mode");
                    require(same_mesh_data(request.source, mesh.view()), "observer source changed");
                    require(request.source_audit.screen_size == request.adjacent_audit.screen_size,
                            "observer audit scales differ");
                    require(request.source_search.views.rotation_seed ==
                                    settings.search_views.rotation_seed &&
                                request.adjacent_search.views.rotation_seed ==
                                    settings.search_views.rotation_seed &&
                                request.source_audit.views.rotation_seed ==
                                    settings.audit_views.rotation_seed &&
                                request.source_search.limit == request.source_audit.limit &&
                                request.adjacent_search.limit == request.adjacent_audit.limit,
                            "observer lost distinct search gates");
                    predecessor |= request.previous.triangles() < request.source.triangles();
                    ++observations;
                    const auto rows = state.teacher_actions(request.condition, 4, 919, &policy,
                                                            TeacherSelection::PolicyMixed);
                    auto teacher_options = options;
                    teacher_options.cache_rasters = false;
                    AuditCuda audit(teacher_options, request.source);
                    auto a = request.source_audit, b = request.adjacent_audit,
                         search_a = request.source_search, search_b = request.adjacent_search;
                    for (auto* config : {&a, &b, &search_a, &search_b}) {
                        config->cancelled = {};
                        config->performance = nullptr;
                    }
                    NeuralStats private_stats;
                    for (const auto& row : rows) {
                        DeviceMeshView candidate;
                        require(state.trial(row.action, candidate), "observer legal trial failed");
                        audit.with_candidate_rasters(std::span{&candidate, 1}, [&] {
                            const auto source = audit.evaluate(request.source, candidate,
                                                               request.bounds, a, &private_stats);
                            const auto previous = audit.evaluate(request.previous, candidate,
                                                                 request.bounds, b, &private_stats);
                            const auto sx =
                                audit.evaluate(request.source, candidate, request.bounds, search_a,
                                               &private_stats);
                            const auto sy =
                                audit.evaluate(request.previous, candidate, request.bounds,
                                               search_b, &private_stats);
                            require(!source.resource_limited && !previous.resource_limited &&
                                        !sx.resource_limited && !sy.resource_limited,
                                    "observer exceeded small-fixture memory budget");
                            // The runtime gate consumes only a threshold verdict;
                            // teacher margins still require the exact measurements.
                            for (const auto& [reference, config, exact] :
                                 {std::tuple{request.source, a, source},
                                  std::tuple{request.previous, b, previous},
                                  std::tuple{request.source, search_a, sx},
                                  std::tuple{request.previous, search_b, sy}}) {
                                const auto predicate = audit.certify(
                                    reference, candidate, request.bounds, config, &private_stats);
                                require(!predicate.resource_limited && !predicate.cancelled &&
                                            predicate.verdict != AuditVerdict::Unknown &&
                                            (predicate.verdict == AuditVerdict::Pass) ==
                                                (exact.complete && exact.passed),
                                        "runtime four-gate certificate differs from exact audit");
                            }
                        });
                    }
                    sparse_passes += private_stats.gpu_sparse_passes;
                    sparse_fallbacks += private_stats.gpu_sparse_fallbacks;
                };
                const auto result =
                    generate_observed(mesh.view(), settings, weights, options, observe, &after);
                require(observations && predecessor, "observer did not see real lower-LOD context");
                require(polls == baseline_polls,
                        "teacher audits consumed caller cancellation polls");
                require(baseline.status == Status::Complete && result.status == baseline.status &&
                            result.lods.size() == baseline.lods.size(),
                        "observer changed generation completion");
                for (size_t i = 0; i < result.lods.size(); ++i)
                    require(same_mesh_data(result.lods[i].view(mesh.view()),
                                           baseline.lods[i].view(mesh.view())) &&
                                result.lods[i].source_error.error ==
                                    baseline.lods[i].source_error.error &&
                                result.lods[i].adjacent.error == baseline.lods[i].adjacent.error,
                            "observer changed emitted geometry or audited errors");
                require(before.action_ranked == after.action_ranked &&
                            before.action_trials == after.action_trials &&
                            before.legal_collapses == after.legal_collapses &&
                            before.rejected_collapses == after.rejected_collapses &&
                            before.action_proposals.size() == after.action_proposals.size(),
                        "teacher queries changed runtime budgets or action counts");
                for (size_t i = 0; i < before.action_proposals.size(); ++i) {
                    const auto& a = before.action_proposals[i];
                    const auto& b = after.action_proposals[i];
                    require(a.origin == b.origin && a.output == b.output &&
                                a.stop_reason == b.stop_reason &&
                                a.start_triangles == b.start_triangles &&
                                a.final_triangles == b.final_triangles &&
                                a.target_triangles == b.target_triangles && a.trials == b.trials &&
                                a.accepted_batches == b.accepted_batches,
                            "observer changed proposal execution");
                }
            }
        require(sparse_passes && sparse_fallbacks,
                "runtime parity fixture did not exercise certificates and exact fallbacks");
        settings.cancelled = {};
        options.action_batch = 1;
        for (bool preserve : {true, false}) {
            options.preserve_uv = preserve;
            weights.use = ModelUse::Unrestricted;
            const auto control = generate_observed(mesh.view(), settings, weights, options, {});
            weights.use = ModelUse::EndpointReuseOnly;
            const auto endpoint = generate_observed(mesh.view(), settings, weights, options, {});
            require(control.status == Status::Complete && endpoint.status == control.status &&
                        endpoint.lods.size() == control.lods.size(),
                    "endpoint scope changed valid reuse completion");
            for (size_t i = 0; i < control.lods.size(); ++i)
                require(same_mesh_data(control.lods[i].view(mesh.view()),
                                       endpoint.lods[i].view(mesh.view())) &&
                            control.lods[i].source_error.error ==
                                endpoint.lods[i].source_error.error &&
                            control.lods[i].adjacent.error == endpoint.lods[i].adjacent.error,
                        "model scope tag changed reuse geometry or measurements");
            for (auto output : {std::optional<OutputMode>{}, std::optional{OutputMode::Rebuild}}) {
                auto invalid = settings;
                invalid.research.output = output;
                uint32_t observations = 0;
                NeuralStats stats;
                bool rejected = false;
                try {
                    generate_observed(
                        mesh.view(), invalid, weights, options,
                        [&](GpuActionState&, const ActionRequest&) { ++observations; }, &stats);
                } catch (const std::invalid_argument&) {
                    rejected = true;
                }
                require(rejected && !observations && !stats.action_trials,
                        "endpoint scope permitted repositioning or rejected after mutation");
            }
        }
        {
            weights.use = ModelUse::Unrestricted;
            ActionCuda policy(weights, options);
            require(policy.model_use() == ModelUse::Unrestricted,
                    "inference initializer scope differs");
            gpu::Device device(options);
            gpu::Buffer<float> refreshed(device, weights.values.size());
            refreshed.upload(weights.values);
            policy.refresh_device(refreshed.p, refreshed.n, ModelUse::EndpointReuseOnly);
            GpuActionState state(mesh.view(), options, true);
            const auto before = state.snapshot();
            require(policy.model_use() == ModelUse::EndpointReuseOnly,
                    "native refresh lost endpoint model scope");
            uint32_t observations = 0;
            bool execute_rejected = false, teacher_rejected = false;
            try {
                state.execute(
                    {}, 1, 1, &policy, NeuralRanking::Learned, 101, 1,
                    [](DeviceMeshView) { return true; }, nullptr, {},
                    [&](GpuActionState&) { ++observations; });
            } catch (const std::invalid_argument&) {
                execute_rejected = true;
            }
            try {
                state.teacher_actions({}, 2, 101, &policy, TeacherSelection::PolicyMixed);
            } catch (const std::invalid_argument&) {
                teacher_rejected = true;
            }
            require(
                execute_rejected && teacher_rejected && !observations,
                "free-placement executor admitted endpoint policy or observed before rejection");
            require(same_mesh_data(state.snapshot().view(mesh.view()), before.view(mesh.view())),
                    "rejected endpoint policy mutated free-placement state");
        }
        {
            ActionCuda collector(weights, options), candidate(weights, options);
            GpuActionState state(mesh.view(), options);
            const auto before = state.snapshot();
            require(state.actions({}).size() > 16,
                    "support fixture does not distinguish a pool from the full action set");
            const auto pool =
                state.teacher_actions({}, 16, 101, &collector, TeacherSelection::PolicyMixed);
            const auto member = [&](Action action) {
                return std::any_of(pool.begin(), pool.end(), [&](const PlacementRecord& row) {
                    return row.action == action;
                });
            };
            uint32_t rescored = 0, traced = 0;
            RankingSupport support{collector,
                                   [&](const SupportedRanking& result) {
                                       ++rescored;
                                       require(
                                           result.pool == pool.size() && member(result.supported) &&
                                               result.global_in_support == member(result.global) &&
                                               result.supported_score <= result.global_score,
                                           "supported ranking disagrees with the collector pool");
                                   },
                                   101};
            ActionStats stats;
            state.execute(
                {}, 1, 64, &candidate, NeuralRanking::Learned, 101, 1,
                [](DeviceMeshView) { return false; }, &stats, {}, {},
                [&](const ActionTrial& trial) {
                    ++traced;
                    require(trial.actions.size() == 1 && member(trial.actions.front()),
                            "clipped rollout tried an unsupported action");
                },
                &support);
            require(
                rescored == 1 && traced == pool.size() && stats.trials == pool.size() &&
                    !stats.accepted &&
                    same_mesh_data(state.snapshot().view(mesh.view()), before.view(mesh.view())),
                "support exhaustion leaked into global actions or changed rejected geometry");
        }
        std::cout << "runtime observer contracts passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
