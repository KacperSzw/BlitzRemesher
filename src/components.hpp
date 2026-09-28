#pragma once
#include "blitz/remesher.hpp"
namespace blitz::detail {
struct ComponentOrder {
    std::vector<uint16_t> face_component,order; // At most 4096 proposal components.
    std::vector<uint32_t> counts;
    uint64_t sample_visits{};
    bool available{},cancelled{};
    Lod select(MeshView,size_t) const;
};
// Search-camera heuristic only. Every selected subset still needs all four gates.
ComponentOrder component_order(MeshView,const Bounds&,const EvalSettings&,size_t memory_limit=256u*1024u*1024u);
}
