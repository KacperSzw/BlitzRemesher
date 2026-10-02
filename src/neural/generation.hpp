#pragma once
#include "neural/action_gpu.hpp"

namespace blitz::neural {
// Synchronous training/debug borrows. The observer may query features and trial
// actions; it must not commit, reset, or retain any mesh/evaluation reference.
struct ActionRequest {
    MeshView source, previous;
    const Bounds& bounds;
    const EvalSettings& source_audit;
    const EvalSettings& adjacent_audit;
    const EvalSettings& source_search;
    const EvalSettings& adjacent_search;
    const std::array<float, conditions>& condition;
    size_t target_triangles;
    OutputMode output;
    uint32_t proposal;
};
using ActionObserver = std::function<void(GpuActionState&, const ActionRequest&)>;
using ExecutionObserver = std::function<void(const ActionRequest&, const ActionTrial&, uint8_t)>;

// Internal entry for native teachers; the public generator supplies no observer.
Result generate_observed(MeshView, const Settings&, const WeightsData&, const NeuralOptions&,
                         const ActionObserver&, NeuralStats* = nullptr,
                         const ExecutionObserver& = {}, const RankingSupport* = nullptr);
} // namespace blitz::neural
