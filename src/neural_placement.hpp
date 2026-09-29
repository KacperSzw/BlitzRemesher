#pragma once
#include "neural_action.hpp"
namespace blitz::neural {
// V3 retains the endpoint context and adds the second coupled source/destination
// wedge. Outputs are rank/source/adjacent logits, midpoint-relative XYZ in edge
// lengths, and two normal residual vectors relative to the destination wedges.
// Position components are bounded to [-1,1] at execution, never to the edge line.
constexpr uint32_t placement_schema=3,placement_features=128,placement_outputs=12;
constexpr size_t placement_weight_count=hidden*(placement_features+1)+hidden*(hidden+1)+placement_outputs*(hidden+1);
constexpr uint32_t policy_inputs(uint32_t architecture){return architecture==action_schema?action_features:architecture==placement_schema?placement_features:0;}
constexpr uint32_t policy_outputs(uint32_t architecture){return architecture==action_schema?action_outputs:architecture==placement_schema?placement_outputs:0;}
constexpr size_t policy_weights(uint32_t architecture){return architecture==action_schema?action_weight_count:architecture==placement_schema?placement_weight_count:0;}
struct Placement {Vec3 position;Vec3 normals[2];};
struct PlacementRecord {Action action;std::array<float,placement_features> x{};};
// Source/adjacent verdicts are independently known. Unknown is never a failure.
enum PlacementLabel:uint8_t {PlacementSourcePass=1,PlacementAdjacentPass=2,PlacementPreferred=4,SourceKnown=8,AdjacentKnown=16,PositionKnown=32,Normal0Known=64,Normal1Known=128};
}
