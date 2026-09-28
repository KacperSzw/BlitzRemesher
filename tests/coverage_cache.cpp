#include "blitz/remesher.hpp"
#include "coverage.hpp"
#include <cstdlib>
#include <iostream>
#include <new>
#include <stdexcept>
#include <tuple>
using namespace blitz;
// Fail one allocation after a view's cancellation poll, inside cached measurement.
static bool fail_allocation=false;
void* operator new(std::size_t n) {
    if(fail_allocation){fail_allocation=false;throw std::bad_alloc();}
    if(auto* p=std::malloc(n?n:1))return p;throw std::bad_alloc();
}
void* operator new[](std::size_t n){return ::operator new(n);}
void* operator new(std::size_t n,const std::nothrow_t&) noexcept {try{return ::operator new(n);}catch(...){return nullptr;}}
void* operator new[](std::size_t n,const std::nothrow_t&) noexcept {try{return ::operator new[](n);}catch(...){return nullptr;}}
void operator delete(void* p) noexcept {std::free(p);}
void operator delete[](void* p) noexcept {std::free(p);}
void operator delete(void* p,const std::nothrow_t&) noexcept {std::free(p);}
void operator delete[](void* p,const std::nothrow_t&) noexcept {std::free(p);}
void operator delete(void* p,std::size_t) noexcept {std::free(p);}
void operator delete[](void* p,std::size_t) noexcept {std::free(p);}
#define CHECK(x) do{if(!(x))throw std::runtime_error("line "+std::to_string(__LINE__)+": " #x);}while(0)
static Mesh tiles() {
    Mesh m;
    for(float x:{0.f,1.7f,3.4f}) {
        auto b=uint32_t(m.positions.size());
        m.positions.insert(m.positions.end(),{{x,0,0},{x+1,0,0},{x+1,1,0},{x,1,0}});
        m.indices.insert(m.indices.end(),{b,b+1,b+2,b,b+2,b+3});
    }
    m.double_sided={1};return m;
}
static auto values(const Measurement& m) {
    return std::tie(m.error,m.coverage,m.coverage_upper,m.changed_area,m.normal_degrees,
        m.worst_view,m.changed_area_worst_view,m.views_evaluated,m.supersample,m.complete,m.passed,m.resource_limited);
}
static void same(const Result& a,const Result& b) {
    CHECK(a.status==b.status&&a.lods.size()==b.lods.size());
    CHECK(a.candidate_evaluations==b.candidate_evaluations&&a.rejected_gates==b.rejected_gates);
    CHECK(a.area_rejected_gates==b.area_rejected_gates&&a.duplicate_proposals==b.duplicate_proposals);
    CHECK(a.selection.reference==b.selection.reference&&a.selection.selected==b.selection.selected);
    CHECK(a.transition_reconnections==b.transition_reconnections&&a.topology_fallback_proposals==b.topology_fallback_proposals);
    CHECK(a.component_builds==b.component_builds&&a.component_unavailable==b.component_unavailable);
    CHECK(runtime_levels(a)==runtime_levels(b)&&storage_stats(a).total()==storage_stats(b).total());
    for(size_t i=0;i<4;++i)CHECK(values(a.worst_rejected[i])==values(b.worst_rejected[i]));
    for(size_t i=0;i<a.lods.size();++i) {
        CHECK(same_mesh_data(a.lods[i].view(a.source),b.lods[i].view(b.source)));
        CHECK(a.lods[i].shared_vertices==b.lods[i].shared_vertices);
        CHECK(values(a.lods[i].adjacent)==values(b.lods[i].adjacent));
        CHECK(values(a.lods[i].source_error)==values(b.lods[i].source_error));
    }
    CHECK(a.candidates.size()==b.candidates.size());
    for(size_t i=0;i<a.candidates.size();++i) {
        CHECK(a.candidates[i].triangles==b.candidates[i].triangles);
        auto x=a.candidates[i].storage,y=b.candidates[i].storage;
        CHECK(std::tie(x.source_vertex_bytes,x.added_vertex_bytes,x.index_bytes)==std::tie(y.source_vertex_bytes,y.added_vertex_bytes,y.index_bytes));
    }
}
int main(){try {
    auto source=tiles(),candidate=source;candidate.indices.resize(12);
    auto b=bounds(source.view());
    EvalSettings e;e.profile=Profile::Coverage;e.screen_size=31;e.views={9,3,198};
    e.supersample=1;e.max_supersample=8;e.limit=10;
    for(bool scalar:{true,false})for(uint32_t budget:{0u,1u,12000u,65536u,2u*1024*1024}) {
        PerformanceStats stats;e.force_scalar=scalar;e.performance=&stats;
        detail::CoverageCache cache(budget,b);
        for(unsigned sweep=0;sweep<3;++sweep) {
            auto expected=evaluate(source.view(),candidate.view(),b,e);
            auto got=cache.evaluate(source.view(),candidate.view(),e,0,false);
            CHECK(values(expected)==values(got));
            CHECK(stats.coverage_cache_peak_bytes<=budget);
        }
        if(budget>=12000)CHECK(stats.coverage_mask_hits>0);
        if(budget==12000)CHECK(stats.coverage_cache_bypasses>0); // Oversized sweep retains a useful subset.
        cache.begin_candidate();auto shifted=candidate;for(auto& p:shifted.positions)p.y+=.2f;
        CHECK(values(cache.evaluate(source.view(),shifted.view(),e,0,false))==values(evaluate(source.view(),shifted.view(),b,e)));
        // Different immutable reference IDs, including high view indices in a separate test below.
        auto parent=source;parent.indices.erase(parent.indices.begin(),parent.indices.begin()+6);
        auto adjacent=e;adjacent.limit=3;
        CHECK(values(cache.evaluate(parent.view(),shifted.view(),adjacent,33,true))==values(evaluate(parent.view(),shifted.view(),b,adjacent)));
    }
    e.performance=nullptr;
    // Refinement, area-only failure, empty masks, identity, clipping, and culling.
    for(double screen:{17.25,32.})for(double limit:{.4,1.2,20.})for(double area:{0.,.3,1.})for(bool two:{false,true}) {
        e.screen_size=screen;e.limit=limit;e.max_changed_area=area;e.force_two_sided=two;
        detail::CoverageCache cache(1024*1024,b);
        for(unsigned variant=0;variant<5;++variant) {
            auto mesh=candidate;
            if(variant==0)mesh=source;
            if(variant==1)mesh.indices.clear();
            if(variant==2)for(auto& p:mesh.positions)p.x+=100;
            if(variant==3){mesh=source;for(auto& p:mesh.positions)p.x+=.001f;}
            if(variant==4){mesh.double_sided={0};for(size_t i=0;i<mesh.indices.size();i+=3)std::swap(mesh.indices[i],mesh.indices[i+1]);}
            cache.begin_candidate();
            auto expected=evaluate(source.view(),mesh.view(),b,e);
            CHECK(values(cache.evaluate(source.view(),mesh.view(),e,0,false))==values(expected));
            CHECK(values(cache.evaluate(source.view(),mesh.view(),e,0,false))==values(expected));
        }
    }
    e.screen_size=4096;e.supersample=e.max_supersample=8;
    {detail::CoverageCache cache(65536,b);auto got=cache.evaluate(source.view(),candidate.view(),e,0,false);
     CHECK(got.resource_limited&&values(got)==values(evaluate(source.view(),candidate.view(),b,e)));}
    // Exact same polling points, including cancellation after cached views.
    e.screen_size=24;e.supersample=2;e.max_supersample=4;e.max_changed_area=1;e.limit=20;
    for(int stop:{1,4,30}) {
        int calls=0;e.cancelled=[&]{return ++calls==stop;};
        auto expected=evaluate(source.view(),candidate.view(),b,e);const int polls=calls;calls=0;
        detail::CoverageCache cache(1024*1024,b);
        CHECK(values(cache.evaluate(source.view(),candidate.view(),e,0,false))==values(expected));CHECK(calls==polls);
    }
    bool inject=false,armed=false;e.cancelled=[&]{if(inject&&!armed){armed=true;fail_allocation=true;}return false;};
    {detail::CoverageCache cache(1024*1024,b);
     auto expected=evaluate(source.view(),candidate.view(),b,e);
     cache.evaluate(source.view(),candidate.view(),e,0,false);cache.begin_candidate();inject=true;
     auto got=cache.evaluate(source.view(),candidate.view(),e,0,false);fail_allocation=false;
     CHECK(armed&&!cache.enabled());CHECK(values(got)==values(expected));}
    e.cancelled={};
    // A mutable callback's state must persist across comparisons, just as it
    // does with the public evaluator. Copying it per call silently resets it.
    e.limit=100;e.views={9,3,198};
    e.cancelled=[remaining=13]() mutable {return --remaining==0;};
    auto cached_settings=e;
    {detail::CoverageCache cache(1024*1024,b);
     for(int trial=0;trial<2;++trial) {
         auto expected=evaluate(source.view(),candidate.view(),b,e);
         auto got=cache.evaluate(source.view(),candidate.view(),cached_settings,0,false);
         CHECK(values(got)==values(expected));
     }}
    e.cancelled={};
    // The last camera exceeds u16; keys must not alias camera 0/65536.
    e.screen_size=1;e.views={65535,2,41};e.supersample=e.max_supersample=1;
    {detail::CoverageCache cache(4*1024*1024,b);
     CHECK(values(cache.evaluate(source.view(),candidate.view(),e,0,false))==values(evaluate(source.view(),candidate.view(),b,e)));}
    Settings s;s.levels=4;s.base_pixels=48;s.last_pixels=8;s.profile=Profile::Coverage;
    s.candidate_budget=8;s.beam_width=2;s.search_views={4,1,35};s.audit_views={8,2,97};
    s.search_supersample=2;s.audit_supersample=4;s.max_supersample=8;s.max_changed_area=.5;
    for(auto output:{std::optional<OutputMode>{},std::optional{OutputMode::Reuse},std::optional{OutputMode::Rebuild}}) {
        s.research.output=output;s.research.coverage_cache_mib=0;auto expected=generate(source.view(),s);
        for(uint16_t budget:{1,3,256}){s.research.coverage_cache_mib=budget;same(expected,generate(source.view(),s));}
    }
    for(auto profile:{Profile::Normals,Profile::Attributes}) {
        s.profile=profile;s.research.coverage_cache_mib=0;auto expected=generate(source.view(),s);
        s.research.coverage_cache_mib=1;same(expected,generate(source.view(),s));
    }
    s.research.coverage_cache_mib=257;CHECK(!validate(s).empty());
    std::cout<<"Coverage cache exactness, bounded admission, lifetimes and fallback passed\n";
}catch(const std::exception& e){fail_allocation=false;std::cerr<<e.what()<<'\n';return 1;}}
