#include "components.hpp"
#include "coverage.hpp"
#include <bit>
#include <numeric>
#include <queue>
#include <stdexcept>
namespace blitz::detail {
Lod ComponentOrder::select(MeshView source,size_t target) const {
    Lod out;std::vector<uint8_t> removed(counts.size());size_t remaining=source.triangles();
    if(available)for(auto c:order)if(remaining>counts[c]&&remaining-counts[c]>=target){removed[c]=1;remaining-=counts[c];}
    out.data.indices.reserve(remaining*3);if(!source.materials.empty())out.data.materials.reserve(remaining);
    for(size_t f=0;f<source.triangles();++f)if(!available||!removed[face_component[f]]) {
        for(unsigned k=0;k<3;++k)out.data.indices.push_back(source.indices[3*f+k]);
        if(!source.materials.empty())out.data.materials.push_back(source.material(f));
    }
    return out;
}
ComponentOrder component_order(MeshView source,const Bounds& bounds,const EvalSettings& settings,size_t memory_limit) {
    ComponentOrder out;
    // This is a bounded proposer, not a restriction on accepted input meshes.
    if(source.triangles()>UINT32_MAX||source.indices.size()>memory_limit/32)return out;
    auto cancelled=[&]{return settings.cancelled&&settings.cancelled();};
    struct Corner {uint32_t vertex,face;uint16_t material;};
    std::vector<uint32_t> parent(source.triangles());std::iota(parent.begin(),parent.end(),0);
    auto root=[&](uint32_t i){while(parent[i]!=i){parent[i]=parent[parent[i]];i=parent[i];}return i;};
    std::vector<Corner> corners;corners.reserve(source.indices.size());
    for(uint32_t f=0;f<source.triangles();++f)for(unsigned k=0;k<3;++k)corners.push_back({source.indices[3*f+k],f,source.material(f)});
    std::sort(corners.begin(),corners.end(),[](auto a,auto b){return std::pair(a.vertex,a.material)<std::pair(b.vertex,b.material);});
    for(size_t i=1;i<corners.size();++i)if(corners[i].vertex==corners[i-1].vertex&&corners[i].material==corners[i-1].material)
        parent[root(corners[i].face)]=root(corners[i-1].face);
    std::vector<Corner>().swap(corners);
    out.face_component.resize(parent.size());std::vector<uint16_t> ids(parent.size(),UINT16_MAX);
    for(uint32_t f=0;f<parent.size();++f){auto r=root(f);if(ids[r]==UINT16_MAX){if(out.counts.size()==4096)return out;ids[r]=uint16_t(out.counts.size());out.counts.push_back(0);}auto c=ids[r];out.face_component[f]=c;++out.counts[c];}
    if(out.counts.size()<2||out.counts.size()>4096)return out;
    std::vector<std::vector<uint32_t>> faces(out.counts.size()),samples(out.counts.size());
    for(uint32_t f=0;f<parent.size();++f)faces[out.face_component[f]].push_back(f);
    auto pixels=std::clamp(settings.screen_size,16.0,128.0);auto views=cameras(bounds,pixels,settings.views);
    std::vector<uint16_t> overlap;std::vector<uint32_t> view_end,source_covered;
    size_t bytes=source.indices.size()*sizeof(uint32_t)*3;
    try {for(auto camera:views) {
        if(cancelled()){out.cancelled=true;return out;}
        size_t base=overlap.size(),width=0;
        for(uint32_t c=0;c<faces.size();++c) {
            if(cancelled()){out.cancelled=true;return out;}
            std::vector<uint32_t> indices;std::vector<uint16_t> materials;
            for(auto f:faces[c]){for(unsigned k=0;k<3;++k)indices.push_back(source.indices[3*f+k]);materials.push_back(source.material(f));}
            auto component=source;component.indices=indices;component.materials=materials;
            auto raster=rasterize_coverage(component,bounds,camera,pixels,2,false);
            if(raster.clipped)return out;
            if(!c){width=size_t(raster.width)*raster.height;if(width>UINT32_MAX-base||bytes+width*5>memory_limit)return out;
                overlap.resize(base+width);bytes+=width*4;}
            // Account for up to 2x vector capacity. No depth/attribute buffers
            // are needed for component union coverage.
            for(size_t w=0;w<raster.words.size();++w)for(auto word=raster.words[w];word;word&=word-1){
                size_t p=w*64+std::countr_zero(word);if(bytes+8+width>memory_limit)return out;
                samples[c].push_back(uint32_t(base+p));++overlap[base+p];bytes+=8;
            }
        }
        view_end.push_back(uint32_t(overlap.size()));source_covered.push_back(uint32_t(std::count_if(overlap.begin()+base,overlap.end(),[](auto n){return n!=0;})));
    }}catch(const std::length_error&){return out;}
    struct Entry {double cost;uint16_t component,epoch;};
    auto cmp=[](auto a,auto b){return std::pair(a.cost,a.component)>std::pair(b.cost,b.component);};
    std::priority_queue<Entry,std::vector<Entry>,decltype(cmp)> heap(cmp);
    auto cost=[&](uint32_t c){double worst=0;size_t v=0;uint32_t unique=0;
        for(auto p:samples[c]){++out.sample_visits;while(p>=view_end[v]){worst=std::max(worst,double(unique)/std::max(1u,source_covered[v]));unique=0;++v;}unique+=overlap[p]==1;}
        worst=std::max(worst,double(unique)/std::max(1u,source_covered[v]));return worst/out.counts[c];};
    for(uint16_t c=0;c<out.counts.size();++c)heap.push({cost(c),c,0});
    uint16_t epoch=0;
    while(!heap.empty()) {
        if(cancelled()){out.cancelled=true;return out;}
        auto item=heap.top();heap.pop();
        // Unique contribution only increases as other components disappear.
        // Stale keys are lower bounds; lazy refresh preserves greedy ordering.
        if(item.epoch!=epoch){heap.push({cost(item.component),item.component,epoch});continue;}
        out.order.push_back(item.component);for(auto p:samples[item.component])--overlap[p];++epoch;
    }
    out.available=true;return out;
}
}
