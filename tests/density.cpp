#include "blitz/remesher.hpp"
#include <iostream>
#include <stdexcept>
using namespace blitz;
#define CHECK(x) do{if(!(x))throw std::runtime_error(std::string(__FILE__)+":"+std::to_string(__LINE__)+": " #x);}while(0)
static Mesh sheet(unsigned n) {
    Mesh m;m.double_sided={1};
    for(unsigned y=0;y<=n;++y)for(unsigned x=0;x<=n;++x)m.positions.push_back({-1+2.f*x/n,-1+2.f*y/n,0});
    for(unsigned y=0;y<n;++y)for(unsigned x=0;x<n;++x){auto a=y*(n+1)+x;m.indices.insert(m.indices.end(),{a,a+1,a+n+2,a,a+n+2,a+n+1});}
    return m;
}
static Lod unchanged(MeshView m) {Lod l;l.data.indices.assign(m.indices.begin(),m.indices.end());return l;}
static Mesh soup(size_t triangles) {
    const auto quad=sheet(1);Mesh m;m.double_sided={1};
    for(size_t f=0;f<std::max<size_t>(2,triangles);++f)for(unsigned k=0;k<3;++k){m.indices.push_back(uint32_t(m.positions.size()));m.positions.push_back(quad.positions[quad.indices[3*(f%2)+k]]);}
    return m;
}
static Settings config() {
    Settings s;s.levels=2;s.base_pixels=s.last_pixels=16;s.profile=Profile::Coverage;
    s.search_views=s.audit_views={6,0,71};s.search_supersample=s.audit_supersample=s.max_supersample=8;
    s.transition={{{0,2},{1,2}}};s.max_lod0_delta_px=2;s.max_changed_area=.1;s.candidate_budget=8;s.beam_width=2;
    s.research.density_targets=true;return s;
}
int main(){try {
    // A connected 18-triangle mesh fits a sixteen-vertex cap. A triangle-soup
    // target bound excludes it before the reducer can produce it.
    for(unsigned divisions:{8,10})for(uint32_t cap:{2000,3000})for(bool graph:{false,true}) {
        auto source=sheet(divisions),original=source,candidate=sheet(3);auto s=config();s.max_added_vertex_bytes_bps=cap;
        s.research.graph_passes=graph;s.candidate_budget=2;unsigned calls=0;
        auto result=generate(source.view(),s,[&](MeshView input,const ReduceSettings& rs){
            ++calls;if((graph&&calls<=s.candidate_budget)||rs.output==OutputMode::Reuse||rs.target_triangles<candidate.view().triangles())return unchanged(input);
            Lod l;l.shared_vertices=false;l.data=candidate;return l;
        });
        CHECK(result.status==Status::Complete&&result.lods.back().view(source.view()).triangles()==candidate.view().triangles());
        CHECK(storage_stats(result).added_vertex_bytes==vertex_bytes(candidate.view()));
        CHECK(storage_stats(result).added_vertex_bytes<=*result.added_vertex_budget_bytes);
        CHECK(result.candidate_evaluations<=uint64_t(s.candidate_budget)*(1+graph));
        CHECK(result.lods.back().source_error.passed&&result.lods.back().adjacent.passed&&same_mesh_data(source.view(),original.view()));
    }
    // Unreferenced source vertices consume source storage but must not dilute
    // the indexed density used to predict a compact rebuilt candidate.
    {
        auto source=sheet(8),candidate=sheet(3);source.positions.resize(181,{0,0,0});auto s=config();s.max_added_vertex_bytes_bps=900;
        auto result=generate(source.view(),s,[&](MeshView input,const ReduceSettings& rs){
            if(rs.output==OutputMode::Reuse||rs.target_triangles<candidate.view().triangles())return unchanged(input);
            Lod l;l.shared_vertices=false;l.data=candidate;return l;
        });
        CHECK(result.lods.back().view(source.view()).triangles()==candidate.view().triangles());
        CHECK(storage_stats(result).added_vertex_bytes<=*result.added_vertex_budget_bytes);
    }
    // Source density is only a hint: a seam-heavy output can have three stored
    // vertices per face. Actual storage must reject it and inform later probes.
    for(uint32_t cap:{2000,3000}) {
        auto source=sheet(8);auto s=config();s.max_added_vertex_bytes_bps=cap;
        auto result=generate(source.view(),s,[](MeshView input,const ReduceSettings& rs){
            if(rs.output==OutputMode::Reuse)return unchanged(input);
            Lod l;l.shared_vertices=false;l.data=soup(rs.target_triangles);return l;
        });
        CHECK(result.status==Status::Complete&&result.vertex_budget_rejections>0);
        CHECK(result.lods.back().view(source.view()).triangles()<source.view().triangles());
        CHECK(storage_stats(result).added_vertex_bytes>0&&storage_stats(result).added_vertex_bytes<=*result.added_vertex_budget_bytes);
    }
    for(uint32_t cap:{0,1000,2000}) {
        auto source=sheet(8);auto s=config();s.max_added_vertex_bytes_bps=cap;
        auto result=generate(source.view(),s,[](MeshView input,const ReduceSettings& rs){
            if(rs.output==OutputMode::Reuse)return unchanged(input);
            Lod l;l.shared_vertices=false;l.data=sheet(4);return l;
        });
        CHECK(result.vertex_budget_rejections>0&&storage_stats(result).added_vertex_bytes==0);
        CHECK(same_mesh_data(result.lods.back().view(source.view()),source.view()));
    }
    std::cout<<"Indexed proposal density and actual storage contracts passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
