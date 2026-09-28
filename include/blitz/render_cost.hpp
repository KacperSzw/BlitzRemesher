#pragma once
#include "evaluate.hpp"
namespace blitz {
// CPU geometry proxies, before depth testing. Not GPU time or actual helper invocations.
// Counts aggregate primitive events over views and therefore require 64-bit storage.
struct RenderCost {
    uint64_t submitted{},culled{},degenerate{},projected{},under_one_px2{},under_four_px2{};
    uint64_t zero_samples{},covered_samples{},primitive_quads{},covered_pixels{};
    uint32_t views{};
    bool complete{true},clipped{};
};
RenderCost render_cost(MeshView,const Bounds&,const Camera&,double screen_pixels);
RenderCost render_cost(MeshView,const Bounds&,ViewSet,double screen_pixels);
}
