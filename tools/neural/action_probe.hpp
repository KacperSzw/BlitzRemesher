#pragma once
#include "neural/action_gpu.hpp"
#include <numeric>

namespace blitz::neural::diagnostic {
// An owned capture of a model on a geometry-selected, immutable action pool.
// Capture every decoded proposal before trial() overwrites the device scratch.
struct ActionProbe {
    std::vector<PlacementRecord> rows;
    std::vector<float> predictions;
    std::vector<Placement> placements;
};
inline ActionProbe capture_actions(GpuActionState& state, const std::array<float, conditions>& c,
                                   ActionCuda& model, uint32_t count, uint32_t seed) {
    if (model.architecture() != conditioned_placement_schema || c[3] || c[4] || c[5])
        throw std::invalid_argument("action probe requires v4 coverage models");
    ActionProbe out;
    out.rows = state.teacher_actions(c, count, seed, &model, TeacherSelection::GeometricRandom);
    std::vector<float> features;
    features.reserve(out.rows.size() * placement_features);
    for (const auto& row : out.rows) {
        features.insert(features.end(), row.x.begin(), row.x.end());
        out.placements.push_back(state.teacher_proposals(row.action, false).back().placement);
    }
    out.predictions = model.predict(features);
    return out;
}
inline void require_same_actions(const ActionProbe& a, const ActionProbe& b) {
    if (a.rows.size() != b.rows.size())
        throw std::runtime_error("action probe model changed the pool size");
    for (size_t i = 0; i < a.rows.size(); ++i)
        if (a.rows[i].action != b.rows[i].action ||
            std::memcmp(a.rows[i].x.data(), b.rows[i].x.data(), sizeof(a.rows[i].x)))
            throw std::runtime_error("action probe models did not receive identical features");
}
inline std::vector<size_t> ranked_actions(const ActionProbe& probe) {
    std::vector<size_t> order(probe.rows.size());
    std::iota(order.begin(), order.end(), 0);
    // Equal scores retain the frozen pool order, independent of the other model.
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return probe.predictions[a * placement_outputs] > probe.predictions[b * placement_outputs];
    });
    return order;
}
struct ActionObservation {
    uint32_t faces{};
    bool valid{}, known{}, passed{};
};
enum class ProbeStop : uint8_t { SafeAction, Unknown, PoolExhausted, TrialBudget };
struct FirstSafeAction {
    ProbeStop stop{ProbeStop::PoolExhausted};
    uint32_t trials{};
    size_t row{SIZE_MAX};
};
inline FirstSafeAction first_safe(std::span<const size_t> order,
                                  std::span<const ActionObservation> observations,
                                  uint32_t budget) {
    FirstSafeAction result;
    for (auto row : order) {
        if (result.trials == budget) {
            result.stop = ProbeStop::TrialBudget;
            return result;
        }
        if (row >= observations.size())
            throw std::invalid_argument("action probe row outside observations");
        ++result.trials;
        const auto& q = observations[row];
        if (!q.known) {
            result.stop = ProbeStop::Unknown;
            result.row = row;
            return result;
        }
        if (q.valid && q.passed) {
            result.stop = ProbeStop::SafeAction;
            result.row = row;
            return result;
        }
    }
    return result;
}
} // namespace blitz::neural::diagnostic
