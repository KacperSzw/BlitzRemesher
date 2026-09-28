#pragma once
#include "blitz/evaluate.hpp"
#include <memory>
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

// Private to generation/tests. Bounds, cameras, screen size, culling and backend
// are fixed for this level. Callers supply stable immutable reference IDs and
// clear candidate storage before changing its mesh. No mesh storage is owned.
class CoverageCache {
    struct Entry { uint32_t key{}; CoverageRaster mask; std::vector<float> field; };
    struct Store {
        std::unique_ptr<Entry[]> entries; // Sorted keys; entry moves preserve payload allocations.
        uint32_t count{},capacity{},limit{},bytes{};
        Entry* find(uint32_t);
        const CoverageRaster& raster(uint32_t,MeshView,const Bounds&,const Camera&,const EvalSettings&,uint8_t,
            CoverageRaster&,const Store&);
        std::span<const float> field(uint32_t,const CoverageRaster&,const EvalSettings&,std::vector<float>&,const Store&);
        void clear();
        void peak(const Store&,PerformanceStats*,uint32_t extra=0) const;
    };
    Store references_,candidate_;
    Bounds bounds_;
public:
    CoverageCache(uint32_t bytes,const Bounds&);
    void begin_candidate() { candidate_.clear(); }
    void disable();
    bool enabled() const { return references_.limit!=0; }
    Measurement evaluate(MeshView,MeshView,const EvalSettings&,uint8_t reference_id,bool audit);
    void measure(Measurement&,MeshView,MeshView,const Camera&,const EvalSettings&,uint8_t ss,uint32_t view,uint8_t reference_id,bool audit);
};
}
