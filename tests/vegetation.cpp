#include "blitz/remesher.hpp"
#include "components.hpp"
#include <iostream>
#include <stdexcept>
using namespace blitz;
#define CHECK(x) do{if(!(x))throw std::runtime_error(std::string(__FILE__)+":"+std::to_string(__LINE__)+": " #x);}while(0)
static Mesh sheet(unsigned n) {
    Mesh m;for(unsigned y=0;y<=n;++y)for(unsigned x=0;x<=n;++x){m.positions.push_back({float(x)/n,float(y)/n,0});m.uv.push_back({float(x)/n,float(y)/n});}
    for(unsigned y=0;y<n;++y)for(unsigned x=0;x<n;++x){uint32_t a=y*(n+1)+x;m.indices.insert(m.indices.end(),{a,a+1,a+n+2,a,a+n+2,a+n+1});}m.double_sided={1};return m;
}
static double area(MeshView m){double a=0;for(size_t f=0;f<m.indices.size();f+=3)a+=length(cross(m.positions[m.indices[f+1]]-m.positions[m.indices[f]],m.positions[m.indices[f+2]]-m.positions[m.indices[f]]))*.5;return a;}
int main(){try {
    for(auto mode:{OutputMode::Reuse,OutputMode::Rebuild})for(double weight:{1.,10.,100.})for(bool reverse:{false,true}) {
        auto m=sheet(5);if(reverse){std::reverse(m.positions.begin(),m.positions.end());std::reverse(m.uv.begin(),m.uv.end());for(auto& i:m.indices)i=uint32_t(m.positions.size()-1-i);}
        auto original=m;ReduceSettings s;s.output=mode;s.target_triangles=40;s.boundary_weight=weight;s.boundary_placement=true;
        auto r=reduce(m.view(),s);CHECK(validate(r.view(m.view())).empty());CHECK(r.data.indices.size()<m.indices.size());
        CHECK(std::abs(area(r.view(m.view()))-1)<1e-6);CHECK(same_mesh_data(original.view(),m.view()));
        if(mode==OutputMode::Reuse){CHECK(r.shared_vertices&&r.data.positions.empty());for(auto i:r.data.indices)CHECK(i<m.positions.size());}
        // Soft quadrics do not guarantee exact area under arbitrary aggressive
        // targets; exact area above tests available zero-error sheet collapses.
        s.target_triangles=2;auto coarse=reduce(m.view(),s);CHECK(validate(coarse.view(m.view())).empty());CHECK(area(coarse.view(m.view()))>0);
        for(auto& p:m.positions)p.z=.1f*std::sin(p.x*3);auto curved=reduce(m.view(),s);CHECK(validate(curved.view(m.view())).empty());
    }
    Mesh cards;auto quad=sheet(1);
    for(float offset:{0.f,0.f,3.f}){uint32_t base=uint32_t(cards.positions.size());for(auto p:quad.positions){p.x+=offset;cards.positions.push_back(p);}for(auto i:quad.indices)cards.indices.push_back(base+i);}cards.double_sided={1};
    EvalSettings e;e.profile=Profile::Coverage;e.views={6,2,73};e.screen_size=32;e.supersample=4;e.max_supersample=8;e.limit=1;
    auto order=detail::component_order(cards.view(),bounds(cards.view()),e);CHECK(order.available&&order.counts.size()==3&&order.order.size()==3);
    auto selected=order.select(cards.view(),4);CHECK(selected.shared_vertices&&selected.data.positions.empty()&&selected.view(cards.view()).triangles()==4);
    CHECK(evaluate(cards.view(),selected.view(cards.view()),bounds(cards.view()),e).passed);
    auto sparse=order.select(cards.view(),2);CHECK(!evaluate(cards.view(),sparse.view(cards.view()),bounds(cards.view()),e).passed);
    auto minimum=order.select(cards.view(),0);CHECK(!minimum.data.indices.empty());
    // Original UV charts can collapse independently without merging streams or
    // mutating the vertex buffer, even where their positions coincide.
    auto seam_source=cards;seam_source.uv.resize(cards.positions.size());for(size_t i=0;i<seam_source.uv.size();++i)seam_source.uv[i]={float(i),float(i%3)};
    auto seam_copy=seam_source;ReduceSettings seams;seams.output=OutputMode::Reuse;seams.target_triangles=2;seams.independent_seams=true;seams.boundary_weight=10;seams.boundary_placement=true;
    auto seam_lod=reduce(seam_source.view(),seams);CHECK(seam_lod.shared_vertices&&seam_lod.data.uv.empty());CHECK(same_mesh_data(seam_copy.view(),seam_source.view()));CHECK(validate(seam_lod.view(seam_source.view())).empty());
    auto limited=detail::component_order(cards.view(),bounds(cards.view()),e,16);CHECK(!limited.available);CHECK(same_mesh_data(limited.select(cards.view(),1).view(cards.view()),cards.view()));
    e.cancelled=[]{return true;};auto cancelled=detail::component_order(cards.view(),bounds(cards.view()),e);CHECK(cancelled.cancelled&&!cancelled.available);e.cancelled={};
    auto materials=quad;materials.indices.insert(materials.indices.end(),quad.indices.begin(),quad.indices.end());materials.materials={0,0,1,1};materials.double_sided={1,1};
    CHECK(detail::component_order(materials.view(),bounds(materials.view()),e).counts.size()==2);
    Settings s;s.levels=4;s.base_pixels=32;s.last_pixels=16;s.profile=Profile::Coverage;s.transition={{{0,2},{1,3}}};s.candidate_budget=7;s.beam_width=2;
    s.search_views={4,1,17};s.audit_views={6,2,83};s.search_supersample=2;s.audit_supersample=4;s.max_supersample=8;
    s.research={10,true,true,true,true};
    for(auto mode:{OutputMode::Reuse,OutputMode::Rebuild})for(auto chain:{ChainMode::Direct,ChainMode::Progressive,ChainMode::Hybrid}) {
        s.output=mode;s.chain=chain;auto r=generate(cards.view(),s);CHECK(r.status==Status::Complete);CHECK(r.proposals.size()==r.candidate_evaluations);
        CHECK(r.candidate_evaluations<=uint64_t(s.levels-1)*s.candidate_budget);
        for(size_t i=1;i<r.lods.size();++i){auto& l=r.lods[i];CHECK(l.source_error.passed&&l.adjacent.passed&&l.data.indices.size()<=r.lods[i-1].data.indices.size());if(mode==OutputMode::Reuse)CHECK(l.shared_vertices);}
        for(auto& p:r.proposals)CHECK(p.level>0&&p.level<s.levels&&p.gate<=8&&p.achieved>0&&p.seconds>=0);
    }
    s.cancelled=[]{return true;};auto r=generate(cards.view(),s);CHECK(r.status==Status::Cancelled);for(auto& l:r.lods)CHECK(same_mesh_data(l.view(cards.view()),cards.view()));
    // A deliberately nonmonotonic provider: intermediate targets fail, but a
    // coarse representation exactly covers the source. Rejections must not
    // permanently exclude exploratory coarse requests.
    auto flat=sheet(6);s.cancelled={};s.levels=2;s.base_pixels=24;s.last_pixels=16;s.chain=ChainMode::Direct;s.research.component_candidates=false;
    for(uint16_t budget:{6,9}) {
        s.candidate_budget=budget;
        auto nonmonotonic=generate(flat.view(),s,[&](MeshView input,const ReduceSettings& proposal){
            Lod l;l.shared_vertices=false;
            if(proposal.target_triangles<input.triangles()/8){l.data=sheet(1);}
            else{l.data=copy_mesh(input);for(auto& p:l.data.positions)p.x+=2;}
            return l;
        });
        CHECK(nonmonotonic.status==Status::Complete&&nonmonotonic.lods.back().data.indices.size()==6);
        CHECK(nonmonotonic.candidate_evaluations<=budget&&nonmonotonic.rejected_gates[0]>0);
    }
    s.research.boundary_weight=-1;CHECK(!validate(s).empty());s.research.boundary_weight=1;s.profile=Profile::Normals;s.research.independent_seams=true;CHECK(!validate(s).empty());
    std::cout<<"Vegetation proposal contracts passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
