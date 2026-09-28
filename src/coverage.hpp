#pragma once
#include "blitz/evaluate.hpp"
namespace blitz::detail {
// Owning, contiguous packed mask. Padding bits in the last word stay zero.
struct CoverageRaster {
    uint32_t width{},height{};
    std::vector<uint64_t> words;
    bool clipped{};
    bool covered(size_t i) const { return (words[i/64]>>(i%64))&1; }
};
CoverageRaster rasterize_coverage(MeshView,const Bounds&,const Camera&,double,uint8_t,bool);
double coverage_distance(const CoverageRaster&,const CoverageRaster&,uint8_t,bool);
}
