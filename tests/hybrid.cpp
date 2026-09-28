#include "blitz/remesher.hpp"
#include <iostream>
#include <stdexcept>
using namespace blitz;
#define CHECK(x) do{if(!(x))throw std::runtime_error(std::string(__FILE__)+":"+std::to_string(__LINE__)+": " #x);}while(0)
int main(){try{
    std::vector<ChainCost> pool={{{3000,2086,16},{100,900,500}},{{3000,2190,16},{100,100,510}},
        {{3000,2086,17},{100,0,100}},{{3000,2191,16},{100,0,200}}};
    CHECK(select_chain(pool,0).selected==0);
    CHECK(select_chain(pool,500).selected==1); // Both integer boundaries matter.
    CHECK(select_chain(pool,625).selected==2);
    uint64_t last=UINT64_MAX;
    for(uint16_t b:{0,200,500,1000}){auto c=select_chain(pool,b);CHECK(c.reference==0);auto bytes=pool[c.selected].storage.total();CHECK(bytes<=last);last=bytes;}
    pool.push_back(pool[1]);CHECK(select_chain(pool,500).selected==1);
    bool threw=false;try{select_chain({},0);}catch(const std::invalid_argument&){threw=true;}CHECK(threw);
    Mesh m;m.positions={{0,0,0},{1,0,0},{0,1,0}};m.indices={0,1,2};m.normals.resize(3);m.uv.resize(3);m.colors.resize(3);m.tangents.resize(3);
    CHECK(vertex_bytes(m.view())==3*(12+12+8+4+16));
    Result r;r.source=m.view();Lod l;l.data.indices=m.indices;r.lods={l,l,l};
    auto s=storage_stats(r);CHECK(s.source_vertex_bytes==156&&s.added_vertex_bytes==0&&s.index_bytes==12);
    r.lods[1].shared_vertices=false;r.lods[1].data=m;
    CHECK(storage_stats(r).total()==168); // Exact render duplicate despite different ownership.
    r.lods[1].data.positions[0].x=.1f;
    s=storage_stats(r);CHECK(s.source_vertex_bytes==156&&s.added_vertex_bytes==156&&s.index_bytes==36);
    Settings cfg;cfg.triangle_overhead_bps=200;cfg.levels=3;cfg.base_pixels=16;cfg.last_pixels=8;cfg.profile=Profile::Coverage;cfg.candidate_budget=8;cfg.beam_width=2;
    cfg.search_views={4,0,31};cfg.audit_views={4,0,73};cfg.search_supersample=cfg.audit_supersample=cfg.max_supersample=2;
    auto source=m;auto a=generate(m.view(),cfg);cfg.triangle_overhead_bps=1000;auto b=generate(m.view(),cfg);
    CHECK(same_mesh_data(source.view(),m.view()));CHECK(a.candidates.size()==b.candidates.size());CHECK(a.candidate_evaluations==b.candidate_evaluations);
    for(size_t i=0;i<a.candidates.size();++i){CHECK(a.candidates[i].triangles==b.candidates[i].triangles);CHECK(a.candidates[i].storage.total()==b.candidates[i].storage.total());}
    CHECK(storage_stats(b).total()<=storage_stats(a).total());
    cfg.cancelled=[]{return true;};auto cancelled=generate(m.view(),cfg);CHECK(cancelled.status==Status::Cancelled);CHECK(cancelled.candidates.size()==1);
    for(auto& lod:cancelled.lods)CHECK(lod.shared_vertices&&lod.data.indices==m.indices);
    cfg.triangle_overhead_bps=10001;CHECK(!validate(cfg).empty());
    std::cout<<"hybrid contracts passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
