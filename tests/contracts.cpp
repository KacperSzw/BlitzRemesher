#include "blitz/remesher.hpp"
#include <iostream>
#include <map>
#include <tuple>
#include <random>
#include <stdexcept>
using namespace blitz;
static int checks=0;
#define CHECK(x) do {++checks;if(!(x))throw std::runtime_error(std::string(__FILE__)+":"+std::to_string(__LINE__)+": " #x);} while(0)
template<class F> void throws(F f){bool yes=false;try{f();}catch(const std::exception&){yes=true;}CHECK(yes);}
static Mesh grid(unsigned n) {
    Mesh m;
    for(unsigned y=0;y<=n;++y)for(unsigned x=0;x<=n;++x){m.positions.push_back({float(x)/n,float(y)/n,0});m.normals.push_back({0,0,1});m.uv.push_back({float(x)/n,float(y)/n});}
    for(unsigned y=0;y<n;++y)for(unsigned x=0;x<n;++x){uint32_t a=y*(n+1)+x,b=a+1,c=a+n+1,d=c+1;m.indices.insert(m.indices.end(),{a,b,d,a,d,c});}
    m.double_sided={1};return m;
}
static Settings small() {
    Settings s;s.levels=3;s.base_pixels=24;s.last_pixels=8;s.profile=Profile::Coverage;s.transition={{{0,3},{1,4}}};
    s.search_views={4,1,27};s.audit_views={8,2,83};s.search_supersample=2;s.audit_supersample=4;s.max_supersample=8;
    s.candidate_budget=4;s.beam_width=2;return s;
}
static Mesh torus() {
    Mesh m;constexpr unsigned major=10,minor=8;
    for(unsigned i=0;i<major;++i)for(unsigned j=0;j<minor;++j) {
        double a=i*6.283185307179586/major,b=j*6.283185307179586/minor;
        m.positions.push_back({float((2+.6*std::cos(b))*std::cos(a)),float((2+.6*std::cos(b))*std::sin(a)),float(.6*std::sin(b))});
    }
    for(unsigned i=0;i<major;++i)for(unsigned j=0;j<minor;++j) {
        uint32_t a=i*minor+j,b=((i+1)%major)*minor+j,c=((i+1)%major)*minor+(j+1)%minor,d=i*minor+(j+1)%minor;
        m.indices.insert(m.indices.end(),{a,b,c,a,c,d});
    }
    return m;
}
static Mesh four_tiles() {
    Mesh m;
    for(float y:{0.f,2.f})for(float x:{0.f,2.f}) {
        uint32_t base=uint32_t(m.positions.size());
        m.positions.insert(m.positions.end(),{{x,y,0},{x+1,y,0},{x+1,y+1,0},{x,y+1,0}});
        m.indices.insert(m.indices.end(),{base,base+1,base+2,base,base+2,base+3});
    }
    m.double_sided={1};return m;
}
static Mesh without_tile(const Mesh& source,unsigned tile) {
    Mesh m=source;
    m.indices.erase(m.indices.begin()+tile*6,m.indices.begin()+(tile+1)*6);
    return m;
}
static double mask_difference(const Raster& a,const Raster& b) {
    CHECK(a.width==b.width&&a.height==b.height&&a.pixels.size()==b.pixels.size());
    size_t changed=0,united=0;
    for(size_t i=0;i<a.pixels.size();++i) {
        changed+=bool(a.pixels[i].covered)!=bool(b.pixels[i].covered);
        united+=bool(a.pixels[i].covered)||bool(b.pixels[i].covered);
    }
    return united?double(changed)/double(united):0;
}
static bool no_overshared_edges(MeshView m) {
    std::map<std::pair<uint32_t,uint32_t>,unsigned> edges;
    for(size_t f=0;f<m.indices.size();f+=3)for(unsigned k=0;k<3;++k) {
        auto a=m.indices[f+k],b=m.indices[f+(k+1)%3];if(a>b)std::swap(a,b);
        if(++edges[{a,b}]>2)return false;
    }
    return true;
}
static double brute(const Raster& a,const Raster& b,unsigned ss) {
    auto directed=[&](const Raster& p,const Raster& q){double worst=0;for(unsigned i=0;i<p.pixels.size();++i)if(p.pixels[i].covered){
        double best=INFINITY;for(unsigned j=0;j<q.pixels.size();++j)if(q.pixels[j].covered){double x=int(i%p.width)-int(j%p.width),y=int(i/p.width)-int(j/p.width);best=std::min(best,x*x+y*y);}worst=std::max(worst,best);}return worst;};
    return std::sqrt(std::max(directed(a,b),directed(b,a)))/ss;
}
int main() {
    try {
        auto m=grid(8);CHECK(validate(m.view()).empty());
        auto empty_optional=m;empty_optional.colors.resize(2);empty_optional.colors.clear();CHECK(validate(empty_optional.view()).empty());
        auto bad=m;bad.indices[0]=UINT32_MAX;CHECK(!validate(bad.view()).empty());
        bad=m;bad.positions[0].x=NAN;CHECK(!validate(bad.view()).empty());
        struct Padded {uint32_t guard;Vec3 p;uint32_t tail;};
        std::vector<Padded> padded;for(auto p:m.positions)padded.push_back({0xB117,p,0xC0DE});
        auto v=m.view();v.positions.data=reinterpret_cast<const std::byte*>(&padded[0].p);v.positions.stride=sizeof(Padded);
        CHECK(validate(v).empty());auto original=padded;
        CHECK(same_mesh_data(v,m.view()));CHECK(same_mesh_data(v,empty_optional.view()));
        auto different=m;different.uv[0].x=.125f;CHECK(!same_mesh_data(v,different.view()));
        different=m;different.normals[0].x=.125f;CHECK(!same_mesh_data(v,different.view()));
        different=m;different.double_sided[0]=0;CHECK(!same_mesh_data(v,different.view()));
        ReduceSettings rs;rs.output=OutputMode::Reuse;rs.target_triangles=24;
        auto reduced=reduce(v,rs);CHECK(reduced.shared_vertices);CHECK(reduced.data.positions.empty());CHECK(reduced.view(v).triangles()<m.view().triangles());
        CHECK(std::memcmp(original.data(),padded.data(),padded.size()*sizeof(Padded))==0);
        for(auto i:reduced.data.indices)CHECK(i<m.positions.size());
        Mesh degenerate;degenerate.positions={{1,.00392f,0},{.447f,0,0},{0,0,0},{.356f,.356f,1}};
        degenerate.indices={3,1,0,2,0,0,0,0,0};rs.target_triangles=2;
        auto clean=reduce(degenerate.view(),rs);CHECK(!clean.data.indices.empty());CHECK(validate(clean.view(degenerate.view())).empty());
        rs.target_triangles=24;
        rs.output=OutputMode::Rebuild;auto rebuilt=reduce(v,rs);CHECK(!rebuilt.shared_vertices);CHECK(validate(rebuilt.view(v)).empty());
        Mesh split;for(auto index:m.indices){split.indices.push_back(uint32_t(split.positions.size()));split.positions.push_back(m.positions[index]);split.normals.push_back(m.normals[index]);split.uv.push_back(m.uv[index]);}
        auto conservative=reduce(split.view(),rs);rs.coupled_wedges=true;auto coupled=reduce(split.view(),rs);
        CHECK(coupled.data.indices.size()<conservative.data.indices.size());CHECK(validate(coupled.view(split.view())).empty());
        CHECK(coupled.data.uv.size()==coupled.data.positions.size());CHECK(uv_distortion(coupled.view(split.view())).negative_uv_faces==0);
        rs.coupled_wedges=false;
        // Guarding common neighbors prevents collapses from manufacturing new
        // overshared edges on an initially manifold handle.
        auto handle=torus();CHECK(no_overshared_edges(handle.view()));
        for(auto mode:{OutputMode::Reuse,OutputMode::Rebuild})for(size_t target:{1u,12u,40u}) {
            ReduceSettings test;test.output=mode;test.target_triangles=target;ReductionStats stats;test.statistics=&stats;
            auto lod=reduce(handle.view(),test);
            CHECK(no_overshared_edges(lod.view(handle.view())));CHECK(validate(lod.view(handle.view())).empty());
            CHECK(stats.initial_triangles==handle.view().triangles()&&stats.final_triangles==lod.view(handle.view()).triangles());
            CHECK(stats.collapsed>0&&stats.attempts>=stats.collapsed);
        }
        auto s=small();auto steps=schedule(bounds(m.view()),s);
        CHECK(steps.size()==3);CHECK(steps[0].pixels==24);CHECK(std::abs(steps.back().pixels-8)<1e-12);
        CHECK(std::abs(steps[2].source-(3*8/steps[1].pixels+4))<1e-12);
        s.max_lod0_delta_px=2;CHECK(schedule(bounds(m.view()),s)[2].source==2);s=small();
        s.levels=2;CHECK(schedule(bounds(m.view()),s)[1].transition==3);s=small();
        s.transition.points={{0,1},{0,2},{1,3}};CHECK(!validate(s).empty());s=small();
        s.last_pixels=100;throws([&]{generate(m.view(),s);});s=small();
        std::mt19937 rng(871);
        for(unsigned trial=0;trial<40;++trial) {
            Raster a{13,11,std::vector<Pixel>(143)},b=a;
            for(auto& p:a.pixels)p.covered=rng()%5==0;for(auto& p:b.pixels)p.covered=rng()%7==0;
            double expected=brute(a,b,2);CHECK(std::abs(coverage_distance(a,b,2,true)-expected)<1e-12);
            CHECK(coverage_distance(a,b,2,true)==coverage_distance(a,b,2,false));
        }
        Raster a{3,3,std::vector<Pixel>(9)},b=a;a.pixels[4].covered=1;
        CHECK(std::isinf(coverage_distance(a,b,1)));b=a;CHECK(coverage_distance(a,b,1)==0);
        a.pixels[4].visible=b.pixels[4].visible=1;a.pixels[4].normal={1,0,0};b.pixels[4].normal={0,1,0};
        EvalSettings e;e.profile=Profile::Normals;e.supersample=1;e.weights.normal=1;
        CHECK(std::abs(attributed_distance(a,b,e,2)-1.5707963267948966)<1e-12);
        CHECK(std::isinf(attributed_distance(a,b,e,1)));e.weights.normal=0;CHECK(attributed_distance(a,b,e,0)==0);
        e.profile=Profile::Attributes;e.weights.material=3;a.pixels[4].material=1;
        CHECK(std::isinf(attributed_distance(a,b,e,2)));e.weights.material=0;CHECK(attributed_distance(a,b,e,0)==0);
        a.pixels[4].color={1,0,0,1};b.pixels[4].color={0,0,0,1};e.weights.color=4;
        CHECK(attributed_distance(a,b,e,5)==4);CHECK(std::isinf(attributed_distance(a,b,e,3)));
        e.weights.color=0;CHECK(attributed_distance(a,b,e,0)==0);
        e=EvalSettings{};e.screen_size=24;e.views={8,2,1};e.supersample=4;e.max_supersample=8;
        CHECK(evaluate(m.view(),m.view(),bounds(m.view()),e).passed);
        auto moved=m;for(auto& p:moved.positions)p.x+=2;CHECK(!evaluate(m.view(),moved.view(),bounds(m.view()),e).passed);
        e.screen_size=4096;e.supersample=e.max_supersample=8;
        auto limited=evaluate(m.view(),moved.view(),bounds(m.view()),e);CHECK(!limited.passed&&limited.resource_limited);
        // Compare the packed coverage evaluator with an independent count of
        // the full raster's XOR and union masks on a fixed near-frontal view.
        auto tiles=four_tiles(),minus_first=without_tile(tiles,0),minus_last=without_tile(tiles,3);
        auto tile_bounds=bounds(tiles.view());
        EvalSettings area_eval;area_eval.profile=Profile::Coverage;area_eval.views={1,0,460796};
        area_eval.screen_size=64;area_eval.supersample=area_eval.max_supersample=4;
        area_eval.force_two_sided=true;area_eval.limit=1000;area_eval.weights={0,0,0};
        auto camera=cameras(tile_bounds,area_eval.screen_size,area_eval.views).front();
        auto source_mask=rasterize(tiles.view(),tile_bounds,camera,area_eval.screen_size,area_eval.supersample,true);
        auto first_mask=rasterize(minus_first.view(),tile_bounds,camera,area_eval.screen_size,area_eval.supersample,true);
        double expected_area=mask_difference(source_mask,first_mask);
        CHECK(expected_area>.2&&expected_area<.35);
        CHECK(mask_difference(source_mask,source_mask)==0);
        auto area_unlimited=evaluate(tiles.view(),minus_first.view(),tile_bounds,area_eval);
        CHECK(area_unlimited.passed&&area_unlimited.error<area_eval.limit);
        CHECK(std::abs(area_unlimited.changed_area-expected_area)<1e-12);
        CHECK(area_unlimited.changed_area_worst_view==0);
        area_eval.views={4,0,460796};
        auto many_views=evaluate(tiles.view(),minus_first.view(),tile_bounds,area_eval);
        double largest_area=0;uint32_t largest_view=0;
        auto audit_cameras=cameras(tile_bounds,area_eval.screen_size,area_eval.views);
        for(uint32_t i=0;i<audit_cameras.size();++i) {
            auto full=rasterize(tiles.view(),tile_bounds,audit_cameras[i],area_eval.screen_size,area_eval.supersample,true);
            auto sparse=rasterize(minus_first.view(),tile_bounds,audit_cameras[i],area_eval.screen_size,area_eval.supersample,true);
            double area=mask_difference(full,sparse);
            if(area>largest_area){largest_area=area;largest_view=i;}
        }
        CHECK(many_views.passed&&many_views.views_evaluated==audit_cameras.size());
        CHECK(std::abs(many_views.changed_area-largest_area)<1e-12);
        CHECK(many_views.changed_area_worst_view==largest_view);
        area_eval.views={1,0,460796};
        for(auto profile:{Profile::Coverage,Profile::Normals,Profile::Attributes}) {
            area_eval.profile=profile;area_eval.max_changed_area=.1;
            auto blocked=evaluate(tiles.view(),minus_first.view(),tile_bounds,area_eval);
            CHECK(!blocked.passed&&blocked.error<area_eval.limit);
            CHECK(std::abs(blocked.changed_area-expected_area)<1e-12);
            area_eval.max_changed_area=1;
            CHECK(evaluate(tiles.view(),minus_first.view(),tile_bounds,area_eval).passed);
        }
        area_eval.profile=Profile::Coverage;area_eval.max_changed_area=0;
        CHECK(evaluate(tiles.view(),tiles.view(),tile_bounds,area_eval).passed);
        auto empty=tiles;empty.indices.clear();auto other_empty=empty;
        for(auto& p:other_empty.positions)p.x+=.125f;
        auto empty_area=evaluate(empty.view(),other_empty.view(),tile_bounds,area_eval);
        CHECK(empty_area.passed&&empty_area.changed_area==0);
        auto lost_all=evaluate(tiles.view(),empty.view(),tile_bounds,area_eval);
        CHECK(!lost_all.passed&&lost_all.changed_area==1);
        CHECK(mask_difference(source_mask,rasterize(empty.view(),tile_bounds,camera,area_eval.screen_size,area_eval.supersample,true))==1);
        for(double bad_area:{-0.01,1.01,std::numeric_limits<double>::quiet_NaN()}) {
            area_eval.max_changed_area=bad_area;
            throws([&]{evaluate(tiles.view(),minus_first.view(),tile_bounds,area_eval);});
        }
        // The first reduction drops one tile. The second proposal drops a
        // different tile: each source audit changes about 1/4 of the mask,
        // while the predecessor audit changes about 1/2.
        area_eval.max_changed_area=.35;
        CHECK(evaluate(tiles.view(),minus_first.view(),tile_bounds,area_eval).passed);
        CHECK(evaluate(tiles.view(),minus_last.view(),tile_bounds,area_eval).passed);
        auto changed_parent=evaluate(minus_first.view(),minus_last.view(),tile_bounds,area_eval);
        CHECK(!changed_parent.passed&&changed_parent.error<area_eval.limit&&changed_parent.changed_area>.4);
        Settings area_chain=small();area_chain.levels=3;area_chain.base_pixels=64;area_chain.last_pixels=64;
        area_chain.max_changed_area=.35;area_chain.transition={{{0,1000},{1,1000}}};
        area_chain.search_views=area_chain.audit_views=area_eval.views;
        area_chain.search_supersample=area_chain.audit_supersample=area_chain.max_supersample=4;
        area_chain.candidate_budget=1;area_chain.research.output=OutputMode::Reuse;
        area_chain.research.chain=ChainMode::Progressive;
        Proposer swap_missing_tile=[&](MeshView input,const ReduceSettings&){
            Lod l;l.data.indices=input.triangles()==tiles.view().triangles()?minus_first.indices:minus_last.indices;
            return l;
        };
        auto chain_result=generate(tiles.view(),area_chain,swap_missing_tile);
        CHECK(chain_result.status==Status::Complete&&chain_result.lods.size()==3);
        CHECK(chain_result.lods[1].view(tiles.view()).triangles()==minus_first.view().triangles());
        CHECK(chain_result.area_rejected_gates[3]>0&&chain_result.rejected_gates[3]>0);
        CHECK(same_mesh_data(chain_result.lods[2].view(tiles.view()),minus_first.view()));
        area_chain.levels=2;area_chain.max_changed_area=.1;
        auto source_rejected=generate(tiles.view(),area_chain,swap_missing_tile);
        CHECK(source_rejected.area_rejected_gates[2]>0&&source_rejected.rejected_gates[2]>0);
        CHECK(same_mesh_data(source_rejected.lods.back().view(tiles.view()),tiles.view()));
        for(double bad_area:{-0.01,1.01,std::numeric_limits<double>::quiet_NaN()}) {
            area_chain.max_changed_area=bad_area;CHECK(!validate(area_chain).empty());
        }
        for(auto mode:{OutputMode::Reuse,OutputMode::Rebuild})for(auto chain:{ChainMode::Direct,ChainMode::Progressive,ChainMode::Hybrid}) {
            s=small();s.research.output=mode;s.research.chain=chain;
            auto r=generate(m.view(),s);CHECK(r.status==Status::Complete);CHECK(r.lods.size()==s.levels);
            for(size_t i=1;i<r.lods.size();++i){auto& l=r.lods[i];CHECK(l.adjacent.passed&&l.source_error.passed);
                CHECK(l.data.indices.size()<=r.lods[i-1].data.indices.size());if(mode==OutputMode::Reuse)CHECK(l.shared_vertices);}
            CHECK(r.lods.back().data.indices.size()<m.indices.size());
        }
        s=small();s.cancelled=[]{return true;};auto r=generate(m.view(),s);CHECK(r.status==Status::Cancelled);
        for(auto& l:r.lods)CHECK(l.data.indices==m.indices&&l.shared_vertices);
        CHECK(runtime_levels(r)==std::vector<uint8_t>{0});
        r.lods[1].shared_vertices=false;r.lods[1].data=copy_mesh(m.view());
        CHECK(runtime_levels(r)==std::vector<uint8_t>{0}); // Ownership is not render data.
        r.lods[1].data.uv[0].x=.25f;auto active=runtime_levels(r);
        CHECK(active.size()==3&&active[0]==0&&active[1]==1&&active[2]==2);
        CHECK(runtime_levels(Result{}).empty());
        // Progressive reducers borrow the previous rebuilt vertex buffer, whose
        // compact IDs cannot be interpreted against the original dense grid.
        s=small();s.research.chain=ChainMode::Progressive;s.levels=4;s.candidate_budget=3;
        bool borrowed_previous=false,trim_borrowed=false;
        Proposer local_proposal=[&](MeshView input,const ReduceSettings&){
            Lod l;
            if(input.positions.count==4) {
                borrowed_previous=true;l.data.indices.assign(input.indices.begin(),input.indices.end());
                if(trim_borrowed)l.data.indices.resize(3);return l;
            }
            l.shared_vertices=false;l.data.positions={{0,0,0},{1,0,0},{1,1,0},{0,1,0}};
            l.data.indices={0,1,2,0,2,3};l.data.double_sided={1};return l;
        };
        auto local_ids=generate(m.view(),s,local_proposal);
        CHECK(borrowed_previous&&local_ids.status==Status::Complete);
        CHECK(std::all_of(local_ids.rejected_gates.begin(),local_ids.rejected_gates.end(),[](auto n){return n==0;}));
        CHECK(local_ids.lods.back().view(m.view()).triangles()==2&&!local_ids.lods.back().shared_vertices);
        trim_borrowed=true;s.transition={{{0,10},{1,11}}};
        auto trimmed=generate(m.view(),s,local_proposal);auto trimmed_view=trimmed.lods.back().view(m.view());
        CHECK(trimmed_view.triangles()==1&&!trimmed.lods.back().shared_vertices);
        CHECK(trimmed_view.positions[1].x==1&&trimmed_view.positions[2].y==1);
        // Hybrid's root has only one distinct input. Deterministic requests
        // must not repeat it with identical settings under the two origin labels.
        // The budget is a ceiling; valid early termination and retuning are allowed.
        for(uint16_t budget:{5,7}) {
            s=small();s.levels=2;s.candidate_budget=budget;
            std::vector<std::tuple<size_t,OutputMode,Objective,double,double,bool,bool>> requests;
            auto counted=generate(m.view(),s,[&](MeshView input,const ReduceSettings& settings){
                CHECK(same_mesh_data(input,m.view()));
                requests.emplace_back(settings.target_triangles,settings.output,settings.objective,
                    settings.normal_weight,settings.regularization,settings.prune,settings.coupled_wedges);
                Lod unchanged;unchanged.data.indices.assign(input.indices.begin(),input.indices.end());return unchanged;
            });
            CHECK(counted.candidate_evaluations<=budget&&counted.candidate_evaluations==requests.size()&&!requests.empty());
            std::sort(requests.begin(),requests.end());CHECK(std::adjacent_find(requests.begin(),requests.end())==requests.end());
        }
        s=small();s.transition={{{0,0},{1,0}}};r=generate(m.view(),s);
        for(auto& l:r.lods)CHECK(l.data.indices==m.indices);
        s=small();s.levels=2;s.base_pixels=s.last_pixels=4096;s.search_supersample=s.audit_supersample=s.max_supersample=8;s.candidate_budget=1;s.research.chain=ChainMode::Direct;
        s.max_added_vertex_bytes_bps=std::nullopt; // Reach the evaluator's resource limit before any storage gate.
        r=generate(m.view(),s);CHECK(r.status==Status::BudgetLimited);CHECK(r.lods.back().data.indices==m.indices);
        std::cout<<checks<<" contract checks passed ("<<evaluator_backend()<<")\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
