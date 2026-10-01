#pragma once
#include "neural/action.hpp"
namespace blitz::neural {
// V3 retains the endpoint context and adds the second coupled source/destination
// wedge. Outputs are rank/source/adjacent logits, midpoint-relative XYZ in edge
// lengths, and two normal residual vectors relative to the destination wedges.
// Position components are bounded to [-1,1] at execution, never to the edge line.
constexpr uint32_t placement_schema = 3, placement_features = 128, placement_outputs = 12;
// V4 replaces the redundant target-edge feature at 79 (also stored at 47)
// with preserve_uv. V3 features and saved weights retain their original meaning.
constexpr uint32_t conditioned_placement_schema = 4;
constexpr bool is_placement_schema(uint32_t architecture) {
    return architecture == placement_schema || architecture == conditioned_placement_schema;
}
constexpr size_t placement_weight_count =
    hidden * (placement_features + 1) + hidden * (hidden + 1) + placement_outputs * (hidden + 1);
constexpr uint32_t policy_inputs(uint32_t architecture) {
    return architecture == action_schema       ? action_features
           : is_placement_schema(architecture) ? placement_features
                                               : 0;
}
constexpr uint32_t policy_outputs(uint32_t architecture) {
    return architecture == action_schema       ? action_outputs
           : is_placement_schema(architecture) ? placement_outputs
                                               : 0;
}
constexpr size_t policy_weights(uint32_t architecture, uint32_t width = 64) {
    return (width == 64 || width == 128 || width == 256) && policy_inputs(architecture)
               ? size_t(width) * (policy_inputs(architecture) + 1) + size_t(width) * (width + 1) +
                     policy_outputs(architecture) * (width + 1)
               : 0;
}
struct Placement {
    Vec3 position;
    Vec3 normals[2];
};
struct PlacementRecord {
    Action action;
    std::array<float, placement_features> x{};
};
// Source/adjacent verdicts are independently known. Unknown is never a failure.
enum PlacementLabel : uint8_t {
    PlacementSourcePass = 1,
    PlacementAdjacentPass = 2,
    PlacementPreferred = 4,
    SourceKnown = 8,
    AdjacentKnown = 16,
    PositionKnown = 32,
    Normal0Known = 64,
    Normal1Known = 128
};
} // namespace blitz::neural
