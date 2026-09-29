#include "blitz/remesher.hpp"
#include <algorithm>
#include <cmath>
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
    CHECK(select_chain(pool,500,100).reference==1);
    CHECK(select_chain(pool,500,0).selected==3);
    std::vector<ChainCost> tail_pool={{{100,20,20,20},{100,80,0}},{{100,60,60,4},{100,96,0}}};
    CHECK(select_chain(tail_pool,0,96).reference==1);
    CHECK(select_chain(tail_pool,0,80).reference==0);
    CHECK(select_chain(tail_pool,0).reference==0);
    threw=false;try{select_chain(std::span<const ChainCost>(pool.data(),2),0,0);}catch(const std::invalid_argument&){threw=true;}CHECK(threw);
    Mesh m;m.positions={{0,0,0},{1,0,0},{0,1,0}};m.indices={0,1,2};m.normals.resize(3);m.uv.resize(3);m.colors.resize(3);m.tangents.resize(3);
    CHECK(vertex_bytes(m.view())==3*(12+12+8+4+16));
    Result r;r.source=m.view();Lod l;l.data.indices=m.indices;r.lods={l,l,l};
    auto s=storage_stats(r);CHECK(s.source_vertex_bytes==156&&s.added_vertex_bytes==0&&s.index_bytes==12);
    r.lods[1].shared_vertices=false;r.lods[1].data=m;
    CHECK(storage_stats(r).total()==168); // Exact render duplicate despite different ownership.
    r.lods[1].data.positions[0].x=.1f;
    s=storage_stats(r);CHECK(s.source_vertex_bytes==156&&s.added_vertex_bytes==156&&s.index_bytes==36);
    auto runtime=runtime_storage(r);CHECK(runtime.size()==3);
    CHECK(runtime[0].scheduled_index==0&&runtime[0].added_vertex_bytes==0&&runtime[0].index_bytes==12);
    CHECK(runtime[1].scheduled_index==1&&runtime[1].added_vertex_bytes==156&&runtime[1].cumulative_added_vertex_bytes==156);
    CHECK(runtime[2].scheduled_index==2&&runtime[2].added_vertex_bytes==0&&runtime[2].cumulative_added_vertex_bytes==156);
    CHECK(Settings{}.triangle_overhead_bps==0&&Settings{}.max_added_vertex_bytes_bps==2000);
    Settings cfg;cfg.triangle_overhead_bps=200;cfg.levels=3;cfg.base_pixels=16;cfg.last_pixels=8;cfg.profile=Profile::Coverage;cfg.candidate_budget=8;cfg.beam_width=2;
    cfg.search_views={4,0,31};cfg.audit_views={4,0,73};cfg.search_supersample=cfg.audit_supersample=cfg.max_supersample=2;
    auto source=m;auto a=generate(m.view(),cfg);cfg.triangle_overhead_bps=1000;auto b=generate(m.view(),cfg);
    CHECK(same_mesh_data(source.view(),m.view()));CHECK(a.candidates.size()==b.candidates.size());CHECK(a.candidate_evaluations==b.candidate_evaluations);
    for(size_t i=0;i<a.candidates.size();++i){CHECK(a.candidates[i].triangles==b.candidates[i].triangles);CHECK(a.candidates[i].storage.total()==b.candidates[i].storage.total());}
    CHECK(storage_stats(b).total()<=storage_stats(a).total());
    cfg.cancelled=[]{return true;};auto cancelled=generate(m.view(),cfg);CHECK(cancelled.status==Status::Cancelled);CHECK(cancelled.candidates.size()==1);
    for(auto& lod:cancelled.lods)CHECK(lod.shared_vertices&&lod.data.indices==m.indices);
    cfg.triangle_overhead_bps=10001;CHECK(!validate(cfg).empty());
    Mesh torus;constexpr unsigned n=4;constexpr double turn=6.2831853071795864769;
    for(unsigned i=0;i<n;++i)for(unsigned j=0;j<n;++j) {
        double u=turn*i/n,v=turn*j/n,r=2+.5*std::cos(v);
        torus.positions.push_back({float(r*std::cos(u)),float(r*std::sin(u)),float(.5*std::sin(v))});
    }
    for(unsigned i=0;i<n;++i)for(unsigned j=0;j<n;++j) {
        auto a=i*n+j,b=((i+1)%n)*n+j,c=((i+1)%n)*n+(j+1)%n,d=i*n+(j+1)%n;
        torus.indices.insert(torus.indices.end(),{a,b,c,a,c,d});
    }
    Settings floor;floor.levels=2;floor.base_pixels=32;floor.last_pixels=16;
    floor.transition={{{0,100},{1,100}}};floor.max_changed_area=.95;
    floor.profile=Profile::Coverage;floor.candidate_budget=2;floor.beam_width=2;
    floor.search_views={4,0,31};floor.audit_views={4,0,73};
    floor.search_supersample=floor.audit_supersample=floor.max_supersample=2;
    auto capped=floor;capped.max_added_vertex_bytes_bps=0;capped.candidate_budget=4;capped.research.trace=true;
    auto constrained=generate(torus.view(),capped);
    CHECK(constrained.added_vertex_budget_bytes==0&&storage_stats(constrained).added_vertex_bytes==0);
    CHECK(constrained.vertex_budget_rejections>0);
    CHECK(std::any_of(constrained.proposals.begin(),constrained.proposals.end(),[](auto& p){return p.gate==9;}));
    CHECK(constrained.selection.reference==constrained.selection.selected);
    Settings tail=capped;tail.levels=3;tail.candidate_budget=1;tail.max_added_vertex_bytes_bps=3000;
    tail.max_changed_area=1;tail.transition={{{0,100},{1,100}}};
    unsigned calls=0;
    auto tail_proposer=[&](MeshView input,const ReduceSettings&) {
        Lod l;
        if(++calls==1){l.data.indices.assign(input.indices.begin(),input.indices.end());return l;}
        l.shared_vertices=false;
        for(auto id:{input.indices[0],input.indices[1],input.indices[2],input.indices[5]})l.data.positions.push_back(input.positions[id]);
        l.data.indices={0,1,2,0,2,3,0,3,1,1,3,2};l.data.double_sided={1};return l;
    };
    auto tail_chain=generate(torus.view(),tail,tail_proposer);
    CHECK(calls==2&&tail_chain.added_vertex_budget_bytes==57);
    CHECK(tail_chain.lods[1].shared_vertices&&!tail_chain.lods[2].shared_vertices);
    CHECK(storage_stats(tail_chain).added_vertex_bytes==48);
    auto tail_cost=runtime_storage(tail_chain);
    CHECK(tail_cost.back().scheduled_index==2&&tail_cost.back().cumulative_added_vertex_bytes==48);
    tail.max_added_vertex_bytes_bps=2000;calls=0;
    auto blocked_tail=generate(torus.view(),tail,tail_proposer);
    CHECK(blocked_tail.added_vertex_budget_bytes==38&&blocked_tail.vertex_budget_rejections>0);
    CHECK(storage_stats(blocked_tail).added_vertex_bytes==0);
    tail.max_added_vertex_bytes_bps=10000;
    auto reserved_tail=generate(torus.view(),tail);
    CHECK(reserved_tail.tail_probe_evaluations>0&&reserved_tail.tail_reserved_vertex_bytes>0);
    CHECK(reserved_tail.tail_reserved_vertex_bytes<=*reserved_tail.added_vertex_budget_bytes);
    CHECK(reserved_tail.lods.back().view(torus.view()).triangles()<torus.view().triangles());
    CHECK(storage_stats(reserved_tail).added_vertex_bytes<=*reserved_tail.added_vertex_budget_bytes);
    uint64_t before_tail=0;
    for(auto level:runtime_storage(reserved_tail))if(level.scheduled_index+1<tail.levels)before_tail+=level.added_vertex_bytes;
    CHECK(before_tail+reserved_tail.tail_reserved_vertex_bytes<=*reserved_tail.added_vertex_budget_bytes);
    tail.max_added_vertex_bytes_bps=std::nullopt;calls=0;
    auto unbounded_tail=generate(torus.view(),tail,tail_proposer);
    CHECK(!unbounded_tail.added_vertex_budget_bytes&&storage_stats(unbounded_tail).added_vertex_bytes==48);
    floor.research.output=OutputMode::Rebuild;floor.research.chain=ChainMode::Direct;floor.research.trace=true;
    auto preserving=generate(torus.view(),floor);
    CHECK(!preserving.added_vertex_budget_bytes);
    floor.research.topology_fallback=true;auto relaxed=generate(torus.view(),floor);
    CHECK(relaxed.topology_fallback_proposals==1);
    CHECK(relaxed.candidate_evaluations==preserving.candidate_evaluations+relaxed.topology_fallback_proposals);
    CHECK(relaxed.proposals.size()==preserving.proposals.size()+relaxed.topology_fallback_proposals);
    CHECK(relaxed.lods.back().view(torus.view()).triangles()<preserving.lods.back().view(torus.view()).triangles());
    for(auto& lod:relaxed.lods) {
        CHECK(lod.source_error.passed&&lod.adjacent.passed);
        CHECK(lod.source_error.changed_area<=floor.max_changed_area&&lod.adjacent.changed_area<=floor.max_changed_area);
    }
    CHECK(std::any_of(relaxed.proposals.begin(),relaxed.proposals.end(),[](auto& p){return p.strategy==3&&p.gate==0;}));
    Mesh obj_order;std::vector<uint32_t> remap(torus.positions.size(),uint32_t(-1));
    for(auto old:torus.indices) {
        auto& id=remap[old];
        if(id==uint32_t(-1)) {id=uint32_t(obj_order.positions.size());obj_order.positions.push_back(torus.positions[old]);obj_order.colors.push_back({255,255,255,255});}
        obj_order.indices.push_back(id);
    }
    floor.levels=3;floor.base_pixels=64;floor.transition={{{0,8},{1,100}}};
    floor.candidate_budget=4;floor.research.chain=ChainMode::Progressive;
    floor.max_changed_area=1;floor.search_views={6,2,2971082788u};floor.audit_views={12,4,2971082790u};
    floor.search_supersample=2;floor.audit_supersample=4;floor.max_supersample=8;
    floor.research.topology_fallback=false;auto progressive_base=generate(obj_order.view(),floor);
    floor.research.topology_fallback=true;auto progressive=generate(obj_order.view(),floor);
    CHECK(!progressive.lods[1].shared_vertices);
    CHECK(progressive.lods.back().view(obj_order.view()).triangles()<progressive_base.lods.back().view(obj_order.view()).triangles());
    CHECK(std::any_of(progressive.proposals.begin(),progressive.proposals.end(),[](auto& p){
        return p.level==2&&p.origin==1&&p.gate==7&&p.link_rejections>0;
    }));
    CHECK(std::any_of(progressive.proposals.begin(),progressive.proposals.end(),[](auto& p){
        return p.level==2&&p.origin==1&&p.strategy==3&&p.gate==0;
    }));
    CHECK(progressive.candidate_evaluations==progressive_base.candidate_evaluations+progressive.topology_fallback_proposals);
    for(auto& lod:progressive.lods) {
        CHECK(lod.source_error.passed&&lod.adjacent.passed);
        CHECK(lod.source_error.changed_area<=floor.max_changed_area&&lod.adjacent.changed_area<=floor.max_changed_area);
    }
    std::cout<<"hybrid contracts passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
