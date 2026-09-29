#include "blitz/remesher.hpp"
#include <numeric>
#include <stdexcept>
namespace blitz {
uint64_t vertex_bytes(MeshView v) {
    return uint64_t(v.positions.count)*12+uint64_t(v.normals.count)*12+
        uint64_t(v.uv.count)*8+uint64_t(v.colors.count)*4+uint64_t(v.tangents.count)*16;
}
StorageStats storage_stats(const Result& r) {
    StorageStats s;s.source_vertex_bytes=vertex_bytes(r.source);
    for(auto level:runtime_storage(r)) {
        s.index_bytes+=level.index_bytes;
        s.added_vertex_bytes+=level.added_vertex_bytes;
    }
    return s;
}
std::vector<RuntimeLevelStorage> runtime_storage(const Result& r) {
    std::vector<RuntimeLevelStorage> out;
    uint64_t cumulative=0;
    for(auto i:runtime_levels(r)) {
        auto v=r.lods[i].view(r.source);
        uint64_t added=r.lods[i].shared_vertices?0:vertex_bytes(v);
        cumulative+=added;
        out.push_back({i,added,uint64_t(v.indices.size())*4,cumulative});
    }
    return out;
}
ChainSelection select_chain(std::span<const ChainCost> pool,uint16_t overhead,std::optional<uint64_t> added_budget,ChainObjective objective) {
    if(pool.empty()||overhead>10000)throw std::invalid_argument("invalid selection pool or overhead");
    const auto levels=pool[0].triangles.size();
    if(levels<2||levels>32)throw std::invalid_argument("selection requires 2..32 levels");
    auto sum=[](const ChainCost& c){return std::accumulate(c.triangles.begin()+1,c.triangles.end(),uint64_t{});};
    auto better=[&](const ChainCost& a,const ChainCost& b){
        if(added_budget&&objective==ChainObjective::TailFirst)for(size_t l=levels;l-->1;)if(a.triangles[l]!=b.triangles[l])
            return a.triangles[l]<b.triangles[l];
        return sum(a)<sum(b);
    };
    for(auto& c:pool)if(c.triangles.size()!=levels||c.triangles[0]!=pool[0].triangles[0])
        throw std::invalid_argument("inconsistent selection pool");
    auto feasible=[&](size_t i){return !added_budget||pool[i].storage.added_vertex_bytes<=*added_budget;};
    size_t first=0;while(first<pool.size()&&!feasible(first))++first;
    if(first==pool.size())throw std::invalid_argument("selection pool exceeds added vertex budget");
    ChainSelection choice{first,first};
    for(size_t i=first+1;i<pool.size();++i)if(feasible(i))
        if(better(pool[i],pool[choice.reference]) ||
          (!better(pool[choice.reference],pool[i])&&pool[i].storage.total()<pool[choice.reference].storage.total()))choice.reference=i;
    choice.selected=choice.reference;
    for(size_t i=0;i<pool.size();++i)if(feasible(i)) {
        bool eligible=true;
        for(size_t l=1;l<levels;++l)
            if(uint64_t(pool[i].triangles[l])*10000>uint64_t(pool[choice.reference].triangles[l])*(10000+overhead)){eligible=false;break;}
        if(eligible&&(pool[i].storage.total()<pool[choice.selected].storage.total() ||
           (pool[i].storage.total()==pool[choice.selected].storage.total()&&sum(pool[i])<sum(pool[choice.selected]))))choice.selected=i;
    }
    return choice;
}
}
