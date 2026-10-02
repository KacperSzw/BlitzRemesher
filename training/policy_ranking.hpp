#pragma once
#include "neural/placement.hpp"
#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

namespace blitz::neural::training {
// A training diagnostic: damp only the learned ranking adjustment. This is not
// an optimizer checkpoint, and must never disguise changes to the frozen actor.
inline WeightsData blend_ranking(const WeightsData& initial, const WeightsData& trained,
                                 double fraction) {
    const auto width = initial.hidden_width;
    const auto count = policy_weights(initial.architecture, width);
    if (initial.use != ModelUse::Unrestricted || trained.use != ModelUse::Unrestricted ||
        initial.architecture != conditioned_placement_schema || !count ||
        trained.architecture != initial.architecture || trained.hidden_width != width ||
        initial.values.size() != count || trained.values.size() != count ||
        !std::isfinite(fraction) || fraction < 0 || fraction > 1)
        throw std::invalid_argument("incompatible ranking blend");
    const auto row = size_t(width) * (placement_features + 1) + size_t(width) * (width + 1);
    const auto bias = row + size_t(width) * placement_outputs;
    auto out = initial;
    for (size_t i = 0; i < count; ++i) {
        const auto a = initial.values[i], b = trained.values[i];
        if (!std::isfinite(a) || !std::isfinite(b))
            throw std::invalid_argument("nonfinite ranking blend");
        if ((i >= row && i < row + width) || i == bias)
            out.values[i] = fraction == 0   ? a
                            : fraction == 1 ? b
                                            : float((1 - fraction) * a + fraction * b);
        else if (std::bit_cast<uint32_t>(a) != std::bit_cast<uint32_t>(b))
            throw std::invalid_argument("ranking blend would hide a changed frozen actor");
    }
    return out;
}

// Sample the entire observed trajectory, including its last visited state.
inline uint32_t ranking_observation_step(uint32_t sample, uint32_t iterations, uint32_t requested) {
    const auto count = std::min(iterations, requested);
    if (!count || sample >= count)
        throw std::invalid_argument("invalid ranking observation schedule");
    return count == 1 ? 0 : uint32_t(uint64_t(sample) * (iterations - 1) / (count - 1));
}
// Native geometry rejection is known infeasibility, not a measured pixel error.
// Store it separately from visual labels, at one bit per row. Only the rank-only
// loss receives the derived joint-feasibility mask; pass heads stay frozen.
inline void append_geometry_rejection(std::vector<uint8_t>& bits, size_t row, bool rejected) {
    if (!(row & 7))
        bits.push_back(0);
    if (rejected)
        bits.at(row >> 3) |= uint8_t(1u << (row & 7));
}
inline void apply_policy_rank_masks(std::span<uint8_t> labels, std::span<const uint8_t> geometry) {
    const auto rows = labels.size();
    if (geometry.size() != (rows + 7) / 8 || (rows % 8 && (geometry.back() >> (rows % 8))))
        throw std::invalid_argument("policy geometry bitmap dimensions/tail");
    for (size_t row = 0; row < rows; ++row) {
        auto& label = labels[row];
        if (label & (PositionKnown | Normal0Known | Normal1Known))
            throw std::invalid_argument("policy ranking data contains placement targets");
        if (geometry[row >> 3] & (1u << (row & 7))) {
            if (label)
                throw std::invalid_argument("geometry rejection has fabricated visual labels");
            label = SourceKnown | AdjacentKnown;
        }
    }
}
inline uint16_t ranking_pairs(std::span<const uint8_t> labels) {
    if (labels.size() > 16)
        throw std::invalid_argument("ranking pool exceeds capacity");
    uint16_t known = 0, preferred = 0;
    for (auto label : labels) {
        known += (label & 24) == 24;
        preferred += (label & 28) == 28;
    }
    return preferred * (known - preferred);
}
} // namespace blitz::neural::training
