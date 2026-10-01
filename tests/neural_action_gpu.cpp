#include "neural/action_gpu.hpp"
#include "neural/cuda.cuh"
#include "tools/neural/action_probe.hpp"
#include <chrono>
#include <iostream>
using namespace blitz;
using namespace blitz::neural;
void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}
void discarded_vertex_contracts() {
    Mesh mesh;
    mesh.positions = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {2, 0, 0}, {3, 0, 0}, {2, 1, 0}};
    mesh.indices = {0, 1, 2, 3, 4, 5};
    mesh.normals.assign(6, {0, 0, 1});
    mesh.uv.assign(6, {8, -8});
    NeuralOptions options;
    options.memory_mib = 128;
    options.vertex_storage = NeuralVertexStorage::Packed;
    GpuActionState state(mesh.view(), options, true);
    auto rows = state.placements({});
    auto row = std::find_if(rows.begin(), rows.end(),
                            [](const auto& r) { return r.action.from == 0 && r.action.to == 1; });
    require(row != rows.end(), "discarded vertex fixture has no legal edge");
    auto original = state.snapshot().data;
    auto proposals = state.teacher_proposals(row->action, false);
    auto valid = proposals[0];
    valid.placement.position = {.5f, 0, 0};
    DeviceMeshView view;
    require(state.trial(row->action, valid.placement, view) && view.faces == 1,
            "valid disconnected-component collapse rejected");
    for (float x : {-1.f, 4.f, INFINITY, std::numeric_limits<float>::quiet_NaN()}) {
        auto invalid = valid;
        invalid.placement.position.x = x;
        require(!state.trial(row->action, invalid.placement, view),
                "discarded vertex escaped placement domain checks");
        std::array<GpuActionState::Proposal, 3> batch{valid, invalid, valid};
        auto views = state.trial_batch(row->action, batch);
        for (size_t i = 0; i < views.size(); ++i) {
            DeviceTrialStatus status;
            gpu::check(
                gpu::copy(&status, views[i].trial_status, sizeof(status), cudaMemcpyDeviceToHost));
            require(bool(status.invalid) == (i == 1) && status.faces == 1,
                    "batched domain rejection contaminated another lane");
        }
        bool rejected = false;
        try {
            state.commit(row->action, invalid.placement);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected && same_mesh_data(original.view(), state.snapshot().data.view()),
                "invalid discarded vertex was committed");
    }
    for (float value : {INFINITY, std::numeric_limits<float>::quiet_NaN()}) {
        auto bad = valid;
        bad.placement.normals[0].x = value;
        require(!state.trial(row->action, bad.placement, view),
                "nonfinite discarded normal was silently normalized");
        auto views = state.trial_batch(row->action, std::span(&bad, 1));
        DeviceTrialStatus status;
        gpu::check(
            gpu::copy(&status, views[0].trial_status, sizeof(status), cudaMemcpyDeviceToHost));
        require(status.invalid, "batched nonfinite normal was silently normalized");
    }
    state.commit(row->action, valid.placement);
    auto result = state.snapshot().data;
    require(result.view().triangles() == 1 && validate(result.view()).empty(),
            "valid collapse failed after rejected lanes");
    for (auto uv : result.uv)
        require(uv.x == 8 && uv.y == -8, "inclusive UV endpoints changed");
}
Mesh plane(unsigned n) {
    Mesh m;
    for (unsigned y = 0; y < n; ++y)
        for (unsigned x = 0; x < n; ++x) {
            m.positions.push_back({float(x), float(y), 0});
            m.normals.push_back({0, 0, 1});
            m.uv.push_back({float(x), float(y)});
            m.colors.push_back({uint8_t(x * 21), uint8_t(y * 12), 100, 255});
            m.tangents.push_back({1, 0, 0, 1});
        }
    for (unsigned y = 0; y + 1 < n; ++y)
        for (unsigned x = 0; x + 1 < n; ++x) {
            uint32_t a = y * n + x;
            m.indices.insert(m.indices.end(), {a, a + 1, a + n, a + 1, a + n + 1, a + n});
        }
    m.double_sided = {1};
    return m;
}
void parity(const Mesh& mesh, unsigned iterations) {
    NeuralOptions options;
    options.memory_mib = 256;
    ActionState cpu(mesh.view());
    GpuActionState gpu(mesh.view(), options);
    auto before = copy_mesh(mesh.view());
    std::array<float, conditions> c{.7f, .1f, .2f, .8f, .5f, .3f, .05f, .4f};
    for (unsigned step = 0; step < iterations; ++step) {
        auto a = cpu.actions(c), b = gpu.actions(c);
        if (a.size() != b.size())
            throw std::runtime_error("action count " + std::to_string(a.size()) +
                                     " != " + std::to_string(b.size()));
        for (size_t i = 0; i < a.size(); ++i) {
            require(a[i].action == b[i].action, "GPU action ordering or legality differs");
            for (unsigned j = 0; j < action_features; ++j)
                if (std::abs(a[i].x[j] - b[i].x[j]) > 2e-6)
                    throw std::runtime_error("GPU feature " + std::to_string(j) +
                                             " differs: " + std::to_string(a[i].x[j]) + " vs " +
                                             std::to_string(b[i].x[j]));
        }
        if (a.empty())
            break;
        auto action = a[(step * 17 + a.size() / 2) % a.size()].action;
        auto expected = cpu.trial(action), actual = gpu.trial(action);
        require(expected.data.indices == actual.data.indices &&
                    expected.data.materials == actual.data.materials,
                "GPU trial differs");
        require(gpu.view().faces == cpu.view().triangles(), "trial changed incumbent");
        cpu.commit(action);
        gpu.commit(action);
        require(gpu.view().faces == cpu.view().triangles(), "GPU commit count differs");
        bool rejected = false;
        try {
            gpu.trial(action);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "GPU executed stale action");
        // A rejected debug action must not poison the following valid operation.
        auto next = cpu.actions(c);
        if (!next.empty())
            gpu.trial(next.front().action);
    }
    require(same_mesh_data(mesh.view(), before.view()), "GPU action changed input");
}
void executor() {
    auto m = plane(7);
    NeuralOptions options;
    options.memory_mib = 256;
    EvalSettings e;
    e.screen_size = 16;
    e.limit = 4;
    e.supersample = 2;
    e.max_supersample = 4;
    e.views = {2, 1, 819};
    GpuActionState gpu(m.view(), options);
    AuditCuda audit(options, m.view());
    for (auto ranking :
         {NeuralRanking::Constant, NeuralRanking::ShortestEdge, NeuralRanking::CurrentPlane})
        for (uint8_t batch : {uint8_t(1), uint8_t(4)})
            for (bool reject : {false, true}) {
                std::vector<Lod> trials;
                ActionStats a, b;
                auto rank = [&](const ActionState& state, std::span<const ActionRecord> rows) {
                    std::vector<float> scores(rows.size());
                    for (size_t i = 0; i < rows.size(); ++i)
                        if (ranking == NeuralRanking::ShortestEdge)
                            scores[i] = -rows[i].x[59];
                        else if (ranking == NeuralRanking::CurrentPlane)
                            scores[i] = float(-state.teacher_cost(rows[i].action));
                    return scores;
                };
                auto decision = [&](size_t n) { return !reject || n + 2 >= m.view().triangles(); };
                auto expected = execute_actions(
                    m.view(), {}, 1, 5, rank,
                    [&](MeshView candidate) {
                        Lod l;
                        l.data = copy_mesh(candidate);
                        l.shared_vertices = false;
                        trials.push_back(std::move(l));
                        return decision(candidate.triangles());
                    },
                    &a, {}, batch);
                gpu.reset();
                size_t at = 0;
                auto actual = gpu.execute(
                    {}, 1, 5, nullptr, ranking, 741, batch,
                    [&](DeviceMeshView candidate) {
                        require(at < trials.size(), "GPU emitted an extra trial");
                        auto reference = trials[at++].data.view();
                        require(candidate.faces == reference.triangles(),
                                "GPU batch face count differs");
                        auto cpu = evaluate(m.view(), reference, bounds(m.view()), e),
                             device = audit.evaluate(m.view(), candidate, bounds(m.view()), e);
                        require(cpu.complete == device.complete && cpu.passed == device.passed &&
                                    std::abs(cpu.error - device.error) < 1e-5 &&
                                    cpu.changed_area == device.changed_area,
                                "borrowed GPU audit differs");
                        return decision(candidate.faces);
                    },
                    &b);
                require(at == trials.size() && a.ranked == b.ranked && a.trials == b.trials &&
                            a.accepted == b.accepted && a.rejected == b.rejected,
                        "GPU executor work contract differs");
                require(actual.data.indices == expected.data.indices,
                        "GPU executor output differs");
            }
    gpu.reset();
    ActionStats stopped;
    auto intact = gpu.execute(
        {}, 1, 5, nullptr, NeuralRanking::Constant, 0, 4, [](DeviceMeshView) { return true; },
        &stopped, [] { return true; });
    require(!stopped.trials && intact.data.indices == m.indices,
            "cancelled GPU executor changed source");
}
void placement_contracts() {
    auto m = plane(7), original = m;
    NeuralOptions options;
    options.memory_mib = 256;
    GpuActionState gpu(m.view(), options, true);
    auto rows = gpu.placements({});
    auto found = std::find_if(rows.begin(), rows.end(), [](const auto& a) {
        return a.action.from == 24 && a.action.to == 25;
    });
    require(found != rows.end(), "free placement fixture has no interior action");
    auto action = found->action;
    auto alternatives = gpu.teacher_proposals(action);
    require(alternatives.size() == 20, "teacher candidate contract");
    bool off_edge = false;
    for (auto& p : alternatives) {
        for (float v : p.target)
            require(std::isfinite(v), "nonfinite teacher target");
        off_edge |= p.placement.position.z != 0;
    }
    require(off_edge, "teacher searches only the edge segment");
    auto shape = gpu.teacher_proposals(action, false);
    require(shape.size() == 10, "coverage teacher retained duplicate normal variants");
    for (size_t i = 0; i < shape.size(); ++i) {
        require(!shape[i].normal_mask, "coverage proposal acquired normal supervision");
        require(std::memcmp(&shape[i].placement.position, &alternatives[i * 2].placement.position,
                            sizeof(Vec3)) == 0,
                "coverage teacher changed placement domain");
        for (unsigned j = 3; j < 9; ++j)
            require(shape[i].target[j] == 0, "coverage teacher has a normal target");
    }
    Placement placement{{3.5f, 3, .08f}, {normalized({.15f, .2f, 1}), {}}};
    DeviceMeshView candidate;
    require(gpu.trial(action, placement, candidate), "valid XYZ placement rejected");
    require(candidate.faces == m.view().triangles() - 2, "placement removed wrong face count");
    auto prior = gpu.snapshot();
    auto bad = placement;
    bad.position.x = INFINITY;
    require(!gpu.trial(action, bad, candidate), "nonfinite placement accepted");
    require(same_mesh_data(prior.data.view(), gpu.snapshot().data.view()),
            "failed placement mutated incumbent");
    gpu.commit(action, placement);
    auto result = gpu.snapshot();
    require(!result.shared_vertices && validate(result.data.view()).empty(),
            "placement output ownership/layout");
    bool moved = false;
    for (size_t i = 0; i < result.data.positions.size(); ++i)
        if (result.data.positions[i].z != 0) {
            moved = true;
            require(length(result.data.positions[i] - placement.position) < 1e-6,
                    "placement restricted to an endpoint");
            require(length(result.data.normals[i] - placement.normals[0]) < 1e-6,
                    "learned normal was discarded");
            auto t = result.data.tangents[i];
            require(std::abs(dot(result.data.normals[i], {t.x, t.y, t.z})) < 1e-6 &&
                        std::abs(length({t.x, t.y, t.z}) - 1) < 1e-6,
                    "tangent frame not transported");
            require(std::abs(result.data.uv[i].x - 3.5f) < 1e-6 &&
                        std::abs(result.data.uv[i].y - 3) < 1e-6,
                    "attribute correspondence left original chart");
        }
    require(moved && same_mesh_data(m.view(), original.view()),
            "placement did not move or changed source");
    auto committed_revision = gpu.view().revision;
    gpu.reset();
    require(same_mesh_data(gpu.snapshot().data.view(), prior.data.view()),
            "placement reset lost source data");
    require(gpu.view().revision > committed_revision, "reset reused a cached render revision");
    auto fresh = gpu.placements({});
    auto again = std::find_if(fresh.begin(), fresh.end(), [&](const auto& row) {
        return row.action.from == action.from && row.action.to == action.to;
    });
    require(again != fresh.end(), "reset lost fixture action");
    placement.position.z = .04f;
    gpu.commit(again->action, placement);
    require(gpu.view().revision > committed_revision,
            "new trajectory reused a committed render revision");
    WeightsData weights;
    weights.architecture = placement_schema;
    weights.values.resize(placement_weight_count);
    ActionCuda policy(weights, options, 7);
    std::vector<float> input(11 * placement_features);
    auto prediction = policy.predict(input);
    require(prediction.size() == 11 * placement_outputs &&
                std::all_of(prediction.begin(), prediction.end(), [](float x) { return x == 0; }),
            "v3 native policy layout");
    GpuActionState reuse(m.view(), options);
    ActionStats stats;
    auto unchanged = reuse.execute(
        {}, m.view().triangles(), 4, &policy, NeuralRanking::Learned, 5, 2,
        [](DeviceMeshView) { return true; }, &stats);
    require(unchanged.shared_vertices && unchanged.data.positions.empty() &&
                unchanged.data.indices == m.indices,
            "v3 Reuse changed vertex ownership");
}
void exact_position_contracts() {
    auto m = plane(7);
    m.exact_position_bits.resize(2);
    m.exact_position_bits[24 / 32] |= 1u << (24 % 32);
    auto original = m;
    NeuralOptions options;
    options.memory_mib = 256;
    options.exact_position_bps = 500;
    GpuActionState state(m.view(), options, true);
    auto rows = state.placements({});
    auto row = std::find_if(rows.begin(), rows.end(),
                            [](auto& r) { return r.action.from == 24 && r.action.to == 25; });
    require(row != rows.end(), "exact propagation fixture edge");
    auto proposals = state.teacher_proposals(row->action, false);
    DeviceMeshView view;
    require(state.trial(row->action, proposals[2].placement, view),
            "exact precision trial invalid");
    auto trial = state.trial_batch(row->action, std::span(proposals).subspan(2, 1));
    require(trial.size() == 1 && trial[0].exact_position_bits != state.view().exact_position_bits,
            "trial borrowed mutable incumbent precision");
    state.commit(row->action, proposals[2].placement);
    auto out = state.snapshot().data;
    unsigned exact = 0;
    for (size_t i = 0; i < out.positions.size(); ++i)
        if (out.view().exact_position(i)) {
            ++exact;
            require(
                std::memcmp(&out.positions[i], &proposals[2].placement.position, sizeof(Vec3)) == 0,
                "exact flag did not follow merged vertex");
        }
    require(exact == 1, "exception duplicated or lost on compaction");
    state.reset();
    auto reset = state.snapshot().data;
    auto expected = original;
    compact(expected);
    require(same_mesh_data(reset.view(), expected.view()), "reset lost precision bits");
    // 1/49 fits this test-owned cap; 1/48 after a collapse does not.
    options.exact_position_bps = 205;
    GpuActionState capped(m.view(), options, true);
    auto choices = capped.placements({});
    auto edge = std::find_if(choices.begin(), choices.end(),
                             [](auto& r) { return r.action.from == 24 && r.action.to == 25; });
    auto p = capped.teacher_proposals(edge->action, false);
    require(!capped.trial(edge->action, p[2].placement, view),
            "cap counted initial capacity instead of surviving referenced IDs");
    GpuActionState reuse(m.view(), options);
    auto actions = reuse.actions({});
    auto keep = std::find_if(actions.begin(), actions.end(),
                             [](auto& r) { return r.action.from == 25 && r.action.to == 24; });
    require(keep != actions.end(), "reuse precision fixture edge");
    bool rejected = false;
    try {
        reuse.commit(keep->action);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected && reuse.snapshot().data.indices == m.indices,
            "reuse committed an over-cap exception");
    auto drop = std::find_if(actions.begin(), actions.end(),
                             [](auto& r) { return r.action.from == 24 && r.action.to == 25; });
    require(drop != actions.end(), "reuse exact removal fixture edge");
    reuse.commit(drop->action);
    require(reuse.snapshot().data.indices.size() == m.indices.size() - 6,
            "reuse retained stale invalid flag or transferred precision to unchanged target");
}
void initial_precision_cap_contracts() {
    auto mesh = plane(10);
    mesh.normals.clear();
    mesh.uv.clear();
    mesh.colors.clear();
    mesh.tangents.clear();
    // All 100 vertices occur in faces, most in several corners. The denominator
    // is unique referenced vertex IDs, not index count or allocated capacity.
    mesh.exact_position_bits.resize(4);
    auto quantization = vertex_bounds(mesh.view());
    NeuralOptions options;
    options.memory_mib = 128;
    options.exact_position_bps = 500;
    for (auto storage : {NeuralVertexStorage::Float32, NeuralVertexStorage::Packed}) {
        options.vertex_storage = storage;
        for (bool placement : {false, true}) {
            mesh.exact_position_bits[0] = 31;
            GpuActionState accepted(mesh.view(), options, placement, &quantization);
            require(accepted.view().faces == mesh.view().triangles(),
                    "initial precision cap rejected its inclusive boundary");
            mesh.exact_position_bits[0] = 63;
            bool rejected = false;
            try {
                GpuActionState excessive(mesh.view(), options, placement, &quantization);
            } catch (const std::invalid_argument&) {
                rejected = true;
            }
            require(rejected, "initial state admitted 6/100 exact vertices under a 5% cap");
            options.exact_position_bps = 600;
            GpuActionState higher(mesh.view(), options, placement, &quantization);
            require(higher.view().faces == accepted.view().faces,
                    "initial precision admission ignored a larger caller cap");
            options.exact_position_bps = 500;
        }
    }
    // An exact but unreferenced vertex must neither count as an exception nor
    // increase the denominator. Constructor validation must not mutate input.
    mesh.positions.push_back({0, 0, 0});
    mesh.exact_position_bits[0] = 31;
    mesh.exact_position_bits[3] |= 1u << 4;
    auto before = mesh;
    GpuActionState unused(mesh.view(), options, true, &quantization);
    require(same_mesh_data(mesh.view(), before.view()), "precision admission mutated input");
    mesh.exact_position_bits[0] = 63;
    options.exact_position_bps = 595;
    bool rejected = false;
    try {
        GpuActionState excessive(mesh.view(), options, true, &quantization);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "unreferenced capacity diluted the initial precision cap");
    mesh.positions.back() = mesh.positions[11];
    mesh.exact_position_bits[3] = 0;
    *std::find(mesh.indices.begin(), mesh.indices.end(), 11u) = 100;
    options.vertex_storage = NeuralVertexStorage::Float32;
    options.preserve_uv = true;
    GpuActionState separate_ids(mesh.view(), options, true, &quantization);
    options.preserve_uv = false;
    rejected = false;
    try {
        GpuActionState welded(mesh.view(), options, true, &quantization);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "initial precision cap counted IDs removed by UV welding");
}
void placement_coincidence_contracts() {
    Mesh mesh;
    mesh.positions = {{-1, 0, 0},   {1, 0, 0},  {1, 2, 0}, {-1, 2, 0},
                      {0, -0.f, 0}, {0, -1, 0}, {1, -1, 0}};
    mesh.indices = {0, 1, 2, 0, 2, 3, 4, 5, 6};
    NeuralOptions options;
    options.memory_mib = 128;
    auto inventory = [](GpuActionState& state) {
        std::vector<Vec3> positions(state.view().vertices);
        gpu::check(gpu::copy(positions.data(), state.view().positions,
                             positions.size() * sizeof(Vec3), cudaMemcpyDeviceToHost));
        std::vector<std::array<float, 6>> keys;
        for (auto& row : state.placements({})) {
            auto a = positions[row.action.from], b = positions[row.action.to];
            keys.push_back({a.x, a.y, a.z, b.x, b.y, b.z});
        }
        std::sort(keys.begin(), keys.end());
        return keys;
    };
    for (auto storage : {NeuralVertexStorage::Float32, NeuralVertexStorage::Packed}) {
        options.vertex_storage = storage;
        GpuActionState state(mesh.view(), options, true);
        auto rows = state.placements({});
        auto row = std::find_if(rows.begin(), rows.end(), [](const auto& r) {
            return r.action.from == 0 && r.action.to == 1;
        });
        require(row != rows.end(), "coincidence fixture has no legal edge");
        auto action = row->action;
        auto proposals = state.teacher_proposals(action, false);
        auto midpoint = proposals[2];
        // Use the exact decoded position, including the packed grid. Its signed
        // zero may differ from the existing vertex, just as constructor hashing.
        Vec3 collision;
        gpu::check(gpu::copy(&collision, state.view().positions + 4, sizeof(collision),
                             cudaMemcpyDeviceToHost));
        midpoint.placement.position = collision;
        midpoint.placement.position.y = 0.f;
        auto original = state.snapshot().data;
        DeviceMeshView trial;
        if (state.trial(action, midpoint.placement, trial)) {
            state.commit(action, midpoint.placement);
            auto snapshot = state.snapshot().data;
            GpuActionState fresh(snapshot.view(), options, true);
            throw std::runtime_error("cross-class coincidence accepted: continued legal actions=" +
                                     std::to_string(state.placements({}).size()) +
                                     ", fresh=" + std::to_string(fresh.placements({}).size()));
        }
        auto safe = proposals[1];
        // This endpoint is the old source position, now unreferenced by the
        // collapsed component. It must not conflict with a discarded source ID.
        safe.placement.position = original.positions[0];
        require(state.trial(action, safe.placement, trial),
                "discarded source position falsely collided");
        std::array batch{safe, midpoint, safe};
        auto views = state.trial_batch(action, batch);
        for (size_t i = 0; i < views.size(); ++i) {
            DeviceTrialStatus status;
            gpu::check(
                gpu::copy(&status, views[i].trial_status, sizeof(status), cudaMemcpyDeviceToHost));
            require(bool(status.invalid) == (i == 1),
                    "batched coincidence admission contaminated another candidate");
        }
        bool rejected = false;
        try {
            state.commit(action, midpoint.placement);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected && same_mesh_data(original.view(), state.snapshot().data.view()),
                "rejected coincidence modified incumbent geometry");
        state.commit(action, safe.placement);
        auto snapshot = state.snapshot().data;
        GpuActionState fresh(snapshot.view(), options, true);
        require(inventory(state) == inventory(fresh),
                "accepted placement changed legality after snapshot reconstruction");
    }
}
void placement_trial_timing() {
    auto mesh = plane(65);
    mesh.uv.clear();
    NeuralOptions options;
    options.memory_mib = 128;
    options.vertex_storage = NeuralVertexStorage::Float32;
    GpuActionState state(mesh.view(), options, true);
    auto action = state.placements({}).front().action;
    auto proposal = state.teacher_proposals(action, false)[1];
    std::array batch{proposal, proposal, proposal, proposal};
    DeviceMeshView view;
    auto trial = [&](bool batched) {
        if (!batched) {
            require(state.trial(action, proposal.placement, view), "timing trial rejected");
            return;
        }
        auto views = state.trial_batch(action, batch);
        for (auto& candidate : views) {
            DeviceTrialStatus status;
            gpu::check(
                gpu::copy(&status, candidate.trial_status, sizeof(status), cudaMemcpyDeviceToHost));
            require(!status.invalid, "timing batch rejected");
        }
    };
    for (bool batched : {false, true}) {
        for (unsigned i = 0; i < 4; ++i)
            trial(batched);
        auto start = std::chrono::steady_clock::now();
        for (unsigned i = 0; i < 64; ++i)
            trial(batched);
        double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::cout << "{\"vertices\":4225,\"faces\":8192,\"batch\":" << (batched ? 4 : 1)
                  << ",\"iterations\":64,\"seconds\":" << seconds << "}\n";
    }
}
void coupled_placement(const Mesh& mesh) {
    NeuralOptions options;
    options.memory_mib = 256;
    GpuActionState gpu(mesh.view(), options, true);
    auto rows = gpu.placements({});
    auto found = std::find_if(rows.begin(), rows.end(), [](const auto& row) {
        return row.action.from == 7 && row.action.to == 12;
    });
    require(found != rows.end(), "seam action missing");
    Placement p{{2, 1.5f, .02f}, {normalized({.1f, 0, 1}), normalized({0, 1, .1f})}};
    DeviceMeshView v;
    require(gpu.trial(found->action, p, v), "coupled seam placement rejected");
    gpu.commit(found->action, p);
    auto result = gpu.snapshot();
    unsigned left = 0, right = 0;
    for (size_t i = 0; i < result.data.positions.size(); ++i)
        if (result.data.positions[i].z != 0) {
            require(length(result.data.positions[i] - p.position) < 1e-6, "seam cracked");
            if (result.data.uv[i].x < 0) {
                ++right;
                require(length(result.data.normals[i] - p.normals[1]) < 1e-6,
                        "second normal crossed seam");
            } else {
                ++left;
                require(length(result.data.normals[i] - p.normals[0]) < 1e-6,
                        "first normal crossed seam");
            }
        }
    require(left && right && validate(result.data.view()).empty(), "seam lost a wedge");
}
void incremental_audits() {
    auto m = plane(9);
    NeuralOptions options;
    options.memory_mib = 256;
    GpuActionState state(m.view(), options, true);
    AuditCuda incremental(options, m.view());
    auto uncached = options;
    uncached.cache_rasters = false;
    AuditCuda full(uncached, m.view());
    EvalSettings e;
    e.profile = Profile::Attributes;
    e.screen_size = 37;
    e.limit = 4;
    e.views = {3, 2, 719};
    e.supersample = e.max_supersample = 3;
    for (unsigned step = 0; step < 3; ++step) {
        auto rows = state.teacher_actions({}, 2, 137 + step);
        require(!rows.empty(), "incremental raster fixture exhausted");
        for (auto& row : rows) {
            auto proposals = state.teacher_proposals(row.action);
            unsigned valid = 0;
            for (auto& proposal : proposals) {
                DeviceMeshView candidate;
                if (!state.trial(row.action, proposal.placement, candidate))
                    continue;
                ++valid;
                auto a = incremental.evaluate(m.view(), candidate, bounds(m.view()), e),
                     b = full.evaluate(m.view(), candidate, bounds(m.view()), e);
                require(a.complete == b.complete && a.passed == b.passed && a.error == b.error &&
                            a.coverage == b.coverage && a.changed_area == b.changed_area &&
                            a.normal_degrees == b.normal_degrees &&
                            a.views_evaluated == b.views_evaluated,
                        "incremental raster changed exact audit");
                if (valid == 1 && a.complete && a.passed) {
                    double margin =
                        std::max(a.error / e.limit, a.changed_area / e.max_changed_area);
                    bool pruned = false;
                    auto tied = incremental.evaluate(m.view(), candidate, bounds(m.view()), e,
                                                     nullptr, margin, &pruned);
                    require(pruned && !tied.complete && !tied.passed &&
                                tied.views_evaluated <= a.views_evaluated,
                            "incumbent tie was not pruned as an unknown bound");
                    auto better =
                        incremental.evaluate(m.view(), candidate, bounds(m.view()), e, nullptr,
                                             std::nextafter(margin, INFINITY), &pruned);
                    require(!pruned && better.complete && better.error == a.error &&
                                better.changed_area == a.changed_area,
                            "incumbent pruning discarded an improving candidate");
                }
            }
            require(valid > 1, "incremental raster needs distinct valid placements");
        }
        auto row = rows.front();
        bool committed = false;
        for (auto& p : state.teacher_proposals(row.action)) {
            DeviceMeshView v;
            if (state.trial(row.action, p.placement, v)) {
                state.commit(row.action, p.placement);
                committed = true;
                break;
            }
        }
        require(committed, "incremental raster commit fixture");
    }
}
void sparse_teacher() {
    auto m = plane(7);
    NeuralOptions options;
    options.memory_mib = 256;
    GpuActionState state(m.view(), options, true);
    std::array<float, conditions> c{.1f, .2f, .3f, .4f, .5f, .6f, .7f, .8f};
    for (unsigned count : {1u, 4u, 16u}) {
        auto sparse = state.teacher_actions(c, count, 918), full = state.placements(c);
        for (auto& row : sparse) {
            auto found = std::find_if(full.begin(), full.end(),
                                      [&](auto& x) { return x.action == row.action; });
            require(found != full.end() && found->x == row.x,
                    "sparse teacher changed selected features");
            state.teacher_proposals(row.action);
        }
        auto again = state.teacher_actions(c, count, 918);
        require(again.size() == sparse.size(), "teacher selection count changed");
        for (size_t i = 0; i < sparse.size(); ++i) {
            require(again[i].action == sparse[i].action && again[i].x == sparse[i].x,
                    "teacher selection reused stale indices");
            auto proposals = state.teacher_proposals(again[i].action);
            require(proposals[0].placement.position.x == m.positions[again[i].action.from].x &&
                        proposals[0].placement.position.y == m.positions[again[i].action.from].y,
                    "teacher cached a different selected action");
        }
    }
}
void conditioned_teacher() {
    auto mesh = plane(5);
    NeuralOptions options;
    options.memory_mib = 128;
    WeightsData weights;
    weights.architecture = conditioned_placement_schema;
    weights.values.resize(policy_weights(weights.architecture));
    weights.values[0] = 1;
    weights.values[hidden * (placement_features + 1)] = 1;
    weights.values[hidden * (placement_features + 1) + hidden * (hidden + 1)] = 1;
    ActionCuda policy(weights, options, 7);
    for (bool preserve : {true, false}) {
        options.preserve_uv = preserve;
        GpuActionState state(mesh.view(), options, true);
        for (uint32_t count : {1u, 4u, 16u}) {
            auto rows =
                     state.teacher_actions({}, count, 827, &policy, TeacherSelection::PolicyMixed),
                 again =
                     state.teacher_actions({}, count, 827, &policy, TeacherSelection::PolicyMixed);
            require(rows.size() == count && again.size() == count, "mixed teacher lost legal rows");
            for (size_t i = 0; i < rows.size(); ++i) {
                require(rows[i].x[79] == float(preserve),
                        "v4 UV policy not represented in features");
                require(rows[i].action == again[i].action && rows[i].x == again[i].x,
                        "mixed teacher is not deterministic");
                for (size_t j = 0; j < i; ++j)
                    require(rows[j].action != rows[i].action, "mixed teacher repeated an action");
            }
            auto full = state.placements({});
            std::vector<float> inputs;
            for (auto& row : full) {
                row.x[79] = float(preserve);
                inputs.insert(inputs.end(), row.x.begin(), row.x.end());
            }
            auto scores = policy.predict(inputs);
            size_t best = 0;
            for (size_t i = 1; i < full.size(); ++i)
                if (scores[i * placement_outputs] > scores[best * placement_outputs])
                    best = i;
            require(rows[0].action == full[best].action,
                    "mixed teacher did not query the highest-ranked learned action");
        }
    }
}
void action_probe_contracts() {
    using namespace blitz::neural::diagnostic;
    auto mesh = plane(5), input = copy_mesh(mesh.view());
    NeuralOptions options;
    options.memory_mib = 128;
    WeightsData first, second;
    first.architecture = second.architecture = conditioned_placement_schema;
    first.values.resize(policy_weights(first.architecture));
    const size_t middle = hidden * (placement_features + 1), last = middle + hidden * (hidden + 1);
    first.values[hidden * placement_features] = 1;
    first.values[middle] = 1;
    first.values[last] = 1;
    first.values[last + 3 * hidden] = .125f;
    second = first;
    // Identical heads, different backbones: swapping output weights cannot
    // reproduce this test's two distinct scores and decoded XYZ placements.
    second.values[hidden * placement_features] = 2;
    ActionCuda a(first, options, 3), b(second, options, 3);
    for (uint32_t count : {3u, 7u}) {
        GpuActionState state(mesh.view(), options, true);
        auto before = state.snapshot().data;
        auto initial = capture_actions(state, {}, a, count, 937),
             final = capture_actions(state, {}, b, count, 937);
        require_same_actions(initial, final);
        require(initial.rows.size() == count, "action probe lost the fixed pool");
        auto order = ranked_actions(initial);
        for (size_t i = 0; i < order.size(); ++i)
            require(order[i] == i, "equal model scores changed the frozen pool order");
        bool valid = false;
        for (size_t i = 0; i < count; ++i) {
            const auto edge = initial.rows[i].action;
            auto from = mesh.positions[edge.from], to = mesh.positions[edge.to];
            const auto midpoint = (from + to) * .5;
            const auto extent = length(to - from);
            for (unsigned model = 0; model < 2; ++model) {
                const auto& probe = model ? final : initial;
                require(probe.predictions[i * placement_outputs] == float(model + 1),
                        "action probe rank came from the wrong backbone");
                const auto& position = probe.placements[i].position;
                require(std::abs(position.x - (midpoint.x + extent * .125 * (model + 1))) < 1e-6 &&
                            std::abs(position.y - midpoint.y) < 1e-6 &&
                            std::abs(position.z - midpoint.z) < 1e-6,
                        "action probe placement came from the wrong full model");
                DeviceMeshView candidate;
                valid |= state.trial(edge, probe.placements[i], candidate);
            }
        }
        require(valid, "action probe fixture supplied no valid edit");
        auto again = capture_actions(state, {}, a, count, 937);
        require_same_actions(initial, again);
        for (size_t i = 0; i < count; ++i)
            require(!std::memcmp(&initial.placements[i], &again.placements[i], sizeof(Placement)),
                    "trial scratch contaminated captured model placements");
        require(same_mesh_data(before.view(), state.snapshot().data.view()) &&
                    same_mesh_data(input.view(), mesh.view()),
                "independent action probes mutated working or input geometry");
        final.rows.front().x[79] = 1 - final.rows.front().x[79];
        bool rejected = false;
        try {
            require_same_actions(initial, final);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        require(rejected, "action probe accepted differently conditioned model inputs");
    }
    const std::array<size_t, 2> forward{0, 1}, reverse{1, 0};
    std::array<ActionObservation, 2> observations{
        {{0, true, false, false}, {12, true, true, true}}};
    auto unknown = first_safe(forward, observations, 2),
         safe = first_safe(reverse, observations, 2);
    require(unknown.stop == ProbeStop::Unknown && unknown.trials == 1 &&
                safe.stop == ProbeStop::SafeAction && safe.row == 1 && safe.trials == 1,
            "action probe treated an earlier unknown as a rejected edit");
    observations[0] = {0, false, true, false};
    auto bounded = first_safe(forward, observations, 1),
         complete = first_safe(forward, observations, 2);
    require(bounded.stop == ProbeStop::TrialBudget && bounded.trials == 1 &&
                complete.stop == ProbeStop::SafeAction && complete.row == 1 && complete.trials == 2,
            "action probe ignored invalid edits or its declared trial budget");
}
void relaxed_uv_contracts() {
    auto source = plane(5);
    Mesh wedges;
    wedges.double_sided = {1};
    for (size_t face = 0; face < source.view().triangles(); ++face)
        for (unsigned corner = 0; corner < 3; ++corner) {
            auto v = source.indices[face * 3 + corner];
            wedges.indices.push_back(uint32_t(wedges.positions.size()));
            wedges.positions.push_back(source.positions[v]);
            wedges.normals.push_back(source.normals[v]);
            wedges.colors.push_back(source.colors[v]);
            auto uv = source.uv[v];
            uv.x += float(face) * .125f;
            wedges.uv.push_back(uv);
            wedges.tangents.push_back({1, 0, 0, 1});
        }
    const auto original = wedges;
    NeuralOptions options;
    options.memory_mib = 128;
    size_t preserved_actions;
    {
        GpuActionState state(wedges.view(), options, true);
        preserved_actions = state.placements({}).size();
    }
    options.preserve_uv = false;
    {
        GpuActionState state(wedges.view(), options, true);
        auto rows = state.placements({});
        require(rows.size() > preserved_actions,
                "UV-only wedges remained locked when preservation was disabled");
        auto packed = state.snapshot().data;
        require(packed.positions.size() == source.positions.size() &&
                    packed.uv.size() == packed.positions.size() &&
                    packed.tangents.size() == packed.positions.size(),
                "UV relaxation did not retain compact best-effort attributes");
        for (auto& row : rows) {
            auto proposals = state.teacher_proposals(row.action, false);
            DeviceMeshView view;
            auto invalid = proposals[0].placement;
            invalid.position.z = INFINITY;
            require(!state.trial(row.action, invalid, view),
                    "UV relaxation disabled finite-geometry checks");
            break;
        }
        state.reset();
        require(same_mesh_data(packed.view(), state.snapshot().data.view()),
                "UV-relaxed reset changed the seed");
    }
    require(same_mesh_data(wedges.view(), original.view()),
            "UV relaxation mutated supplied streams");
    // A normal discontinuity remains a separate wedge even when its only other
    // difference is UV. The new control must not turn off shading constraints.
    auto center = std::find_if(wedges.positions.begin(), wedges.positions.end(),
                               [](Vec3 p) { return p.x == 2 && p.y == 2; });
    wedges.normals[size_t(center - wedges.positions.begin())] = {0, 1, 0};
    {
        GpuActionState state(wedges.view(), options, true);
        auto out = state.snapshot().data;
        require(out.positions.size() > source.positions.size(),
                "UV relaxation welded a normal discontinuity");
    }
    for (size_t i = 0; i < wedges.uv.size(); ++i)
        require(wedges.uv[i].x == original.uv[i].x && wedges.uv[i].y == original.uv[i].y,
                "UV relaxation mutated supplied attributes");
}
void action_stop_contracts() {
    auto mesh = plane(5);
    NeuralOptions options;
    options.memory_mib = 128;
    GpuActionState state(mesh.view(), options);
    ActionStats stats;
    auto run = [&](size_t target, uint32_t budget, bool accept,
                   const std::function<bool()>& cancelled = {}) {
        state.reset();
        return state.execute(
            {}, target, budget, nullptr, NeuralRanking::Constant, 101, 1,
            [=](DeviceMeshView) { return accept; }, &stats, cancelled);
    };
    run(mesh.view().triangles(), 1, true);
    require(stats.stop_reason == NeuralActionStop::TargetReached && stats.trials == 0,
            "target completion misclassified");
    run(1, 0, true);
    require(stats.stop_reason == NeuralActionStop::TrialBudget && stats.trials == 0,
            "zero action budget reported saturation");
    run(1, 1, true);
    require(stats.stop_reason == NeuralActionStop::TrialBudget && stats.trials == 1 &&
                stats.accepted_batches == 1,
            "accepted capped proposal reported convergence");
    run(1, 65536, false);
    require(stats.stop_reason == NeuralActionStop::NoAcceptedAction && stats.trials > 0 &&
                !stats.accepted,
            "exhausted rejected actions reported a work cap");
    run(1, 4, true, [] { return true; });
    require(stats.stop_reason == NeuralActionStop::Cancelled && stats.trials == 0,
            "cancelled proposal reported saturation");
    Mesh tetra;
    tetra.positions = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    tetra.indices = {0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3};
    GpuActionState locked(tetra.view(), options);
    locked.execute(
        {}, 1, 4, nullptr, NeuralRanking::Constant, 101, 1, [](DeviceMeshView) { return true; },
        &stats);
    require(stats.stop_reason == NeuralActionStop::NoLegalActions && !stats.trials,
            "empty legal action set not distinguished from rejected actions");
}
void workspace_budget() {
    NeuralOptions options;
    options.memory_mib = 128;
    MemoryScope memory(options);
    memory.budget.limit = 1024;
    {
        gpu::Device first(options, true), second(options);
        {
            gpu::Buffer<std::byte> a(first, 640);
            bool rejected = false;
            try {
                gpu::Buffer<std::byte> b(second, 512);
            } catch (const gpu::ResourceError& e) {
                rejected = e.kind == NeuralResourceLimit::WorkspaceMemory && e.limit == 1024;
            }
            require(rejected, "separate workspaces exceeded shared cap");
        }
        require(memory.budget.live == 640, "idle pool capacity disappeared from budget");
        {
            gpu::Buffer<std::byte> reuse(first, 512);
            require(memory.budget.live == 640, "pool reuse double counted budget");
        }
    }
    require(memory.budget.live == 0 && memory.budget.peak == 640,
            "workspace allocation lifetime mismatch");
}
void episode_features() {
    auto origin = plane(7), seed = plane(5);
    NeuralOptions options;
    options.memory_mib = 128;
    GpuActionState state(seed.view(), options, true, nullptr, origin.view());
    auto rows = state.placements({});
    require(!rows.empty(), "episode has no features");
    auto b = bounds(origin.view());
    for (const auto& row : rows) {
        auto p = seed.positions[row.action.from];
        require(std::abs(row.x[66] - 32.f / 72) < 1e-7, "episode progress restarted at LOD0");
        require(std::abs(row.x[0] - (p.x - b.center.x) / (b.radius * 2)) < 2e-6 &&
                    std::abs(row.x[1] - (p.y - b.center.y) / (b.radius * 2)) < 2e-6,
                "episode lost source coordinate normalization");
    }
}
void bounded_inference() {
    auto mesh = plane(7);
    NeuralOptions options;
    options.memory_mib = 128;
    WeightsData weights;
    weights.architecture = placement_schema;
    weights.values.resize(placement_weight_count);
    weights.values[0] = .2f;
    size_t second = hidden * (placement_features + 1), last = second + hidden * (hidden + 1);
    weights.values[second] = .3f;
    weights.values[last] = .4f;
    weights.values[last + hidden] = .02f;
    ActionCuda policy(weights, options, 7);
    std::array<float, conditions> c{};
    Lod expected;
    ActionStats reference;
    for (auto ranking :
         {NeuralRanking::Learned, NeuralRanking::Shuffled, NeuralRanking::ShortestEdge,
          NeuralRanking::Constant, NeuralRanking::CurrentPlane})
        for (bool placement : {false, true}) {
            for (uint32_t chunk : {1024u, 1u, 7u, 64u}) {
                GpuActionState state(mesh.view(), options, placement, nullptr, {}, chunk);
                ActionStats stats;
                auto result = state.execute(
                    c, mesh.view().triangles() - 6, 5, &policy, ranking, 319, 2,
                    [](DeviceMeshView) { return true; }, &stats);
                if (chunk == 1024) {
                    expected = std::move(result);
                    reference = stats;
                    require(stats.accepted > 0,
                            "bounded inference fixture did not exercise a decision");
                } else
                    require(same_mesh_data(expected.view(mesh.view()), result.view(mesh.view())) &&
                                stats.ranked == reference.ranked &&
                                stats.trials == reference.trials &&
                                stats.accepted == reference.accepted &&
                                stats.rejected == reference.rejected,
                            "inference chunk changed ranking, placement or topology");
            }
        }
    for (uint32_t invalid : {0u, 65537u}) {
        bool rejected = false;
        try {
            GpuActionState state(mesh.view(), options, true, nullptr, {}, invalid);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "invalid inference chunk accepted");
    }
}
void strided_upload() {
    auto m = plane(5);
    struct Vertex {
        uint32_t prefix;
        Vec3 position;
        uint8_t gap[7];
        Vec3 normal;
    };
    std::vector<Vertex> vertices(m.positions.size());
    for (size_t i = 0; i < vertices.size(); ++i) {
        vertices[i].position = m.positions[i];
        vertices[i].normal = m.normals[i];
    }
    auto view = m.view();
    view.positions.data = reinterpret_cast<const std::byte*>(&vertices[0].position);
    view.positions.stride = sizeof(Vertex);
    view.normals.data = reinterpret_cast<const std::byte*>(&vertices[0].normal);
    view.normals.stride = sizeof(Vertex);
    NeuralOptions options;
    options.memory_mib = 128;
    GpuActionState packed(m.view(), options), strided(view, options);
    auto a = packed.actions({}), b = strided.actions({});
    require(a.size() == b.size(), "strided upload topology");
    for (size_t i = 0; i < a.size(); ++i)
        require(a[i].action == b[i].action && a[i].x == b[i].x, "strided upload changed features");
}
int main(int argc, char** argv) {
    try {
        if (!neural_available())
            return 77;
        if (argc == 2) {
            if (std::string_view(argv[1]) == "--initial-precision")
                initial_precision_cap_contracts();
            else if (std::string_view(argv[1]) == "--placement-coincidence")
                placement_coincidence_contracts();
            else if (std::string_view(argv[1]) == "--placement-timing")
                placement_trial_timing();
            else
                throw std::invalid_argument("unknown GPU action contract selector");
            std::cout << "GPU action admission contracts passed\n";
            return 0;
        }
        discarded_vertex_contracts();
        for (unsigned n : {5u, 7u}) {
            auto m = plane(n);
            parity(m, 8);
            m.normals.clear();
            m.uv.clear();
            m.colors.clear();
            m.tangents.clear();
            m.positions[n + 1].z = .2f;
            parity(m, 6);
        }
        auto seam = plane(5);
        std::vector<uint32_t> duplicate(25, UINT32_MAX);
        for (uint32_t y = 0; y < 5; ++y) {
            auto i = y * 5 + 2;
            duplicate[i] = uint32_t(seam.positions.size());
            seam.positions.push_back(seam.positions[i]);
            seam.normals.push_back({0, 1, 0});
            seam.uv.push_back({-2, float(y)});
            seam.colors.push_back(seam.colors[i]);
            seam.tangents.push_back({0, 0, 1, -1});
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
        parity(seam, 8);
        Mesh tetra;
        tetra.positions = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        tetra.indices = {0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3};
        parity(tetra, 1);
        auto unsafe = plane(5);
        unsafe.indices.insert(unsafe.indices.end(), {6, 7, 12, 6, 7, 17, 1, 1, 2});
        parity(unsafe, 3);
        Mesh bow;
        bow.positions = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {-1, 0, 0}, {0, -1, 0}};
        bow.indices = {0, 1, 2, 0, 3, 4};
        parity(bow, 1);
        executor();
        placement_contracts();
        exact_position_contracts();
        initial_precision_cap_contracts();
        placement_coincidence_contracts();
        coupled_placement(seam);
        incremental_audits();
        sparse_teacher();
        conditioned_teacher();
        action_probe_contracts();
        relaxed_uv_contracts();
        action_stop_contracts();
        workspace_budget();
        strided_upload();
        episode_features();
        bounded_inference();
        std::cout << "GPU action contracts passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
