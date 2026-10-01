#pragma once
#include "blitz/remesher.hpp"
#include <bit>
namespace blitz::detail {
// One bounded proposal history, owned by its input/output slot. Ratios choose
// requests only; actual packed storage and the visual gates decide admission.
struct DensityTargets {
    uint64_t bytes{};
    uint32_t triangles{},request{}; // 31-bit target (<UINT32_MAX/3) and accepted bit.
    size_t next(MeshView input,size_t maximum,uint64_t remaining,size_t exploratory) {
        maximum=std::max<size_t>(1,maximum);
        if(!bytes) {
            std::vector<uint64_t> used((input.positions.count+63)/64);
            for(auto id:input.indices)used[id/64]|=uint64_t(1)<<(id%64);
            uint32_t vertices=0;for(auto word:used)vertices+=std::popcount(word);
            triangles=uint32_t(input.triangles());bytes=uint64_t(vertices)*(vertex_bytes(input)/input.positions.count);
        }
        // Clamp before converting: the byte cap can exceed the source size.
        auto predicted=std::max<size_t>(1,size_t(std::min(double(maximum),double(triangles)*double(remaining)/double(bytes))));
        const auto requested=request&0x7fffffffu;
        if(!requested)return predicted;
        if(bytes>remaining) {
            // A stalled reducer can miss its target. Still make a materially
            // smaller request; do not loop on the same over-budget output.
            return predicted<requested?std::max<size_t>(1,predicted-std::max<size_t>(1,predicted/16)):std::max<size_t>(1,requested/2);
        }
        if(request&0x80000000u)return std::max<size_t>(1,std::min({maximum,exploratory,size_t(triangles/2)}));
        if(predicted>requested)return predicted;
        // Failed appearance is not monotone in triangle count. Spend the next
        // existing slot on a different density instead of excluding that side.
        return std::max<size_t>(1,std::min(exploratory,size_t(requested)*3/4));
    }
    void observe(size_t target,size_t achieved,uint64_t emitted_bytes,uint8_t gate) {
        if(gate==5||!achieved||!emitted_bytes)return;
        request=uint32_t(target)|((gate==0||gate==7)?0x80000000u:0);triangles=uint32_t(achieved);bytes=emitted_bytes;
    }
};
static_assert(sizeof(DensityTargets)==16);
}
