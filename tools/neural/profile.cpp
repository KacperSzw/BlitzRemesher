#include "neural/action_gpu.hpp"
#include "training/json.hpp"
#include <chrono>
#include <cuda_runtime_api.h>
#include <iostream>
using namespace blitz;
using namespace blitz::neural;
using json = nlohmann::json;
Mesh fixture(unsigned side) {
    Mesh m;
    for (unsigned y = 0; y < side; ++y)
        for (unsigned x = 0; x < side; ++x) {
            m.positions.push_back({float(x), float(y), 0});
            m.normals.push_back({0, 0, 1});
            m.uv.push_back({float(x) / side, float(y) / side});
        }
    for (unsigned y = 0; y + 1 < side; ++y)
        for (unsigned x = 0; x + 1 < side; ++x) {
            uint32_t a = y * side + x;
            m.indices.insert(m.indices.end(), {a, a + 1, a + side, a + 1, a + side + 1, a + side});
        }
    m.double_sided = {1};
    return m;
}
int main() {
    try {
        if (!neural_available())
            return 77;
        using Clock = std::chrono::steady_clock;
        auto seconds = [](auto start) {
            return std::chrono::duration<double>(Clock::now() - start).count();
        };
        cudaDeviceProp device;
        cudaGetDeviceProperties(&device, 0);
        NeuralOptions options;
        options.memory_mib = 1024;
        json out = {{"fixture", "40x40 planar grid"},
                    {"triangles", 3042},
                    {"gpu", device.name},
                    {"compute_capability",
                     std::to_string(device.major) + "." + std::to_string(device.minor)},
                    {"memory_bytes", device.totalGlobalMem},
                    {"topology", json::array()},
                    {"audit", json::array()},
                    {"quality_score", nullptr}};
        auto mesh = fixture(40);
        auto start = Clock::now();
        GpuActionState state(mesh.view(), options);
        cudaDeviceSynchronize();
        out["resident_setup_seconds"] = seconds(start);
        for (unsigned repeat = 0; repeat < 3; ++repeat) {
            ActionStats cpu_stats, gpu_stats;
            start = Clock::now();
            auto cpu = execute_actions(
                mesh.view(), {}, 1, 8,
                [](const ActionState&, std::span<const ActionRecord> rows) {
                    return std::vector<float>(rows.size(), 0);
                },
                [](MeshView) { return true; }, &cpu_stats, {}, 16);
            double cpu_time = seconds(start);
            start = Clock::now();
            state.reset();
            auto gpu = state.execute(
                {}, 1, 8, nullptr, NeuralRanking::Constant, 371, 16,
                [](DeviceMeshView) { return true; }, &gpu_stats);
            double gpu_time = seconds(start);
            if (cpu.data.indices != gpu.data.indices || cpu_stats.accepted != gpu_stats.accepted)
                throw std::runtime_error("profile topology parity failed");
            out["topology"].push_back({{"repeat", repeat},
                                       {"cpu_seconds", cpu_time},
                                       {"gpu_seconds", gpu_time},
                                       {"trials", gpu_stats.trials},
                                       {"accepted", gpu_stats.accepted},
                                       {"ranked", gpu_stats.ranked},
                                       {"triangles", gpu.data.indices.size() / 3},
                                       {"exact_output", true}});
        }
        ActionState reference(mesh.view());
        auto rows = reference.actions({});
        std::vector<Lod> candidates;
        for (unsigned i = 0; i < 8; ++i)
            candidates.push_back(reference.trial(rows[i * 31].action));
        EvalSettings e;
        e.profile = Profile::Normals;
        e.screen_size = 48;
        e.limit = 3;
        e.supersample = 2;
        e.max_supersample = 4;
        e.views = {4, 2, 718};
        std::vector<Measurement> expected;
        for (bool cache : {false, true, true, false}) {
            options.cache_rasters = cache;
            AuditCuda audit(options, mesh.view());
            NeuralStats stats;
            start = Clock::now();
            std::vector<Measurement> measured;
            for (unsigned i = 0; i < 8; ++i) {
                e.limit = 3 + i * .001;
                measured.push_back(audit.evaluate(mesh.view(), candidates[i].view(mesh.view()),
                                                  bounds(mesh.view()), e, &stats));
            }
            double time = seconds(start);
            if (expected.empty())
                expected = measured;
            else
                for (size_t i = 0; i < expected.size(); ++i)
                    if (expected[i].error != measured[i].error ||
                        expected[i].changed_area != measured[i].changed_area ||
                        expected[i].complete != measured[i].complete ||
                        expected[i].passed != measured[i].passed)
                        throw std::runtime_error("profile audit parity failed");
            out["audit"].push_back({{"cache", cache},
                                    {"seconds", time},
                                    {"stats", neural_json(stats)},
                                    {"exact_metrics", true}});
        }
        std::cout << out.dump(2) << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
