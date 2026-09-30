#include "blitz/remesher.hpp"
#include <numeric>
#include <stdexcept>
namespace blitz {
uint64_t vertex_bytes(MeshView v) {
    return uint64_t(v.positions.count)*12+uint64_t(v.normals.count)*12+
        uint64_t(v.uv.count)*8+uint64_t(v.colors.count)*4+uint64_t(v.tangents.count)*16+v.exact_position_bits.size_bytes();
}
StorageStats storage_stats(const Result& r) {
    StorageStats s;s.source_vertex_bytes=vertex_bytes(r.source);
    for(auto i:runtime_levels(r)) {
        auto v=r.lods[i].view(r.source);
        s.index_bytes+=uint64_t(v.indices.size())*4;
        if(!r.lods[i].shared_vertices)s.added_vertex_bytes+=vertex_bytes(v);
    }
    return s;
}
ChainSelection select_chain(std::span<const ChainCost> pool,uint16_t overhead) {
    if(pool.empty()||overhead>10000)throw std::invalid_argument("invalid selection pool or overhead");
    const auto levels=pool[0].triangles.size();
    if(levels<2||levels>32)throw std::invalid_argument("selection requires 2..32 levels");
    auto sum=[](const ChainCost& c){return std::accumulate(c.triangles.begin()+1,c.triangles.end(),uint64_t{});};
    for(auto& c:pool)if(c.triangles.size()!=levels||c.triangles[0]!=pool[0].triangles[0])
        throw std::invalid_argument("inconsistent selection pool");
    ChainSelection choice;
    for(size_t i=1;i<pool.size();++i)
        if(sum(pool[i])<sum(pool[choice.reference]) ||
          (sum(pool[i])==sum(pool[choice.reference])&&pool[i].storage.total()<pool[choice.reference].storage.total()))choice.reference=i;
    choice.selected=choice.reference;
    for(size_t i=0;i<pool.size();++i) {
        bool eligible=true;
        for(size_t l=1;l<levels;++l)
            if(uint64_t(pool[i].triangles[l])*10000>uint64_t(pool[choice.reference].triangles[l])*(10000+overhead)){eligible=false;break;}
        if(eligible&&(pool[i].storage.total()<pool[choice.selected].storage.total() ||
           (pool[i].storage.total()==pool[choice.selected].storage.total()&&sum(pool[i])<sum(pool[choice.selected]))))choice.selected=i;
    }
    return choice;
}
}
