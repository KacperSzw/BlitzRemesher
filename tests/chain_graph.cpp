#include "blitz/remesher.hpp"
#include <iostream>
#include <numeric>
#include <stdexcept>
using namespace blitz;
#define CHECK(x) do{if(!(x))throw std::runtime_error(std::string(__FILE__)+":"+std::to_string(__LINE__)+": " #x);}while(0)
static Mesh rectangle(float shift,unsigned divisions) {
    Mesh m;m.double_sided={1};
    for(unsigned y=0;y<=divisions;++y)for(unsigned x=0;x<=divisions;++x)
        m.positions.push_back({shift-1+2.f*x/divisions,-1+2.f*y/divisions,0});
    for(unsigned y=0;y<divisions;++y)for(unsigned x=0;x<divisions;++x) {
        auto a=y*(divisions+1)+x,b=a+1,c=a+divisions+1,d=c+1;
        m.indices.insert(m.indices.end(),{a,b,d,a,d,c});
    }
    return m;
}
static Mesh bridge(float shift) {
    auto m=rectangle(shift,1);m.positions.push_back({shift,0,0});m.indices={0,1,4,1,3,4,3,2,4,2,0,4};return m;
}
static uint64_t total(const Result& r) {uint64_t n=0;for(size_t i=1;i<r.lods.size();++i)n+=r.lods[i].data.indices.size()/3;return n;}
int main(){try {
    const auto source=rectangle(0,2),away=bridge(-.2f),towards=bridge(.2f),tail=rectangle(.4f,1);
    const auto original=source;
    Settings s;s.levels=3;s.base_pixels=s.last_pixels=16;s.profile=Profile::Coverage;
    s.search_views=s.audit_views={6,0,31};s.search_supersample=s.audit_supersample=s.max_supersample=8;
    s.candidate_budget=2;s.beam_width=2;s.max_added_vertex_bytes_bps=10000;s.research.trace=true;
    EvalSettings e;e.profile=Profile::Coverage;e.views=s.audit_views;e.screen_size=16;e.supersample=e.max_supersample=8;e.limit=100;
    const auto reference=bounds(source.view());
    const auto sb=evaluate(source.view(),towards.view(),reference,e).error;
    const auto sa=evaluate(source.view(),away.view(),reference,e).error;
    const auto bt=evaluate(towards.view(),tail.view(),reference,e).error;
    const auto st=evaluate(source.view(),tail.view(),reference,e).error;
    const auto at=evaluate(away.view(),tail.view(),reference,e).error;
    const auto good=std::max({sb,sa,bt});CHECK(good<st&&st<at);
    const auto transition=(good+st)/2;s.transition={{{0,float(transition)},{1,float(transition)}}};
    s.max_lod0_delta_px=st+.1;s.max_changed_area=.75;
    auto provider=[&](unsigned& calls){return [&,counter=&calls](MeshView input,const ReduceSettings&) {
        Lod l;auto call=++*counter;
        if(call<=4){l.data.indices.assign(input.indices.begin(),input.indices.end());return l;}
        l.shared_vertices=false;
        switch((call-5)%4){case 0:l.data=away;break;case 1:l.data=towards;break;default:l.data=tail;break;}
        return l;
    };};
    unsigned calls=0;auto baseline=generate(source.view(),s,provider(calls));CHECK(calls==4&&total(baseline)==16);
    s.research.graph_passes=1;calls=0;auto r=generate(source.view(),s,provider(calls));
    CHECK(r.status==Status::Complete&&r.graph_passes_completed==1&&r.candidate_evaluations==8&&calls==8);
    // The first equal-count predecessor points away from the tail. The graph
    // must retain the other geometry and audit that alternative transition.
    CHECK(same_mesh_data(r.lods[1].view(source.view()),towards.view()));
    CHECK(same_mesh_data(r.lods[2].view(source.view()),tail.view()));CHECK(total(r)==6);
    CHECK(r.lods[2].source_error.error>transition&&r.lods[2].adjacent.error<=transition);
    CHECK(storage_stats(r).added_vertex_bytes==vertex_bytes(towards.view())+vertex_bytes(tail.view()));
    CHECK(storage_stats(r).added_vertex_bytes==*r.added_vertex_budget_bytes);
    CHECK(r.chain_objective==ChainObjective::WholeChain&&r.search_progress.size()==2);
    CHECK(r.graph_edges>0&&r.rejected_gates[1]>0&&same_mesh_data(source.view(),original.view()));
    for(size_t i=1;i<r.lods.size();++i) {
        e.limit=r.lods[i].schedule.source;e.max_changed_area=s.max_changed_area;
        CHECK(evaluate(source.view(),r.lods[i].view(source.view()),reference,e).passed);
        e.limit=r.lods[i].schedule.transition;
        CHECK(evaluate(r.lods[i-1].view(source.view()),r.lods[i].view(source.view()),reference,e).passed);
    }
    for(uint8_t passes:{2,3}) {
        s.research.graph_passes=passes;calls=0;auto longer=generate(source.view(),s,provider(calls));
        CHECK(longer.status==Status::Complete&&total(longer)<=total(r));
        CHECK(longer.graph_passes_completed==passes&&longer.search_progress.size()==size_t(passes+1));
        for(size_t i=1;i<longer.search_progress.size();++i)CHECK(longer.search_progress[i].triangle_total<=longer.search_progress[i-1].triangle_total);
    }
    for(uint32_t cap:{0,9999}) {
        s.max_added_vertex_bytes_bps=cap;calls=0;auto limited=generate(source.view(),s,provider(calls));
        CHECK(limited.status==Status::Complete&&storage_stats(limited).added_vertex_bytes<=*limited.added_vertex_budget_bytes);
        CHECK(total(limited)>6&&limited.vertex_budget_rejections>0);
    }
    s.max_added_vertex_bytes_bps=10000;s.research.graph_passes=1;calls=0;
    s.cancelled=[&]{return calls>4;};auto cancelled=generate(source.view(),s,provider(calls));
    CHECK(cancelled.status==Status::Cancelled&&cancelled.graph_passes_completed==0&&total(cancelled)==total(baseline));
    CHECK(runtime_levels(cancelled).size()==1&&storage_stats(cancelled).added_vertex_bytes==0);
    s.research.graph_passes=3;calls=0;s.cancelled=[&]{return calls>8;};
    auto later_cancel=generate(source.view(),s,provider(calls));
    CHECK(later_cancel.status==Status::Cancelled&&later_cancel.graph_passes_completed==1&&total(later_cancel)==6);
    s.cancelled={};calls=0;auto produce=provider(calls);
    auto no_memory=generate(source.view(),s,[&](MeshView input,const ReduceSettings& rs){if(calls==8)throw std::bad_alloc{};return produce(input,rs);});
    CHECK(no_memory.status==Status::BudgetLimited&&no_memory.graph_passes_completed==1&&total(no_memory)==6);
    s.cancelled={};s.research.graph_passes=4;CHECK(!validate(s).empty());
    s.research.graph_passes=1;s.research.output=OutputMode::Reuse;CHECK(!validate(s).empty());
    s.research.output.reset();s.research.chain=ChainMode::Progressive;CHECK(!validate(s).empty());
    // A smaller triangle count can be valid after a rejected denser candidate.
    s.research.chain=ChainMode::Hybrid;s.levels=2;s.candidate_budget=2;s.transition={{{0,100},{1,100}}};s.max_lod0_delta_px=100;
    auto degenerate=source;for(auto& p:degenerate.positions)p={0,0,0};
    calls=0;auto nonmonotone=generate(source.view(),s,[&](MeshView input,const ReduceSettings&) {
        Lod l;if(++calls<=2){l.data.indices.assign(input.indices.begin(),input.indices.end());return l;}
        l.shared_vertices=false;l.data=calls==3?degenerate:rectangle(0,1);return l;
    });
    CHECK(nonmonotone.lods.back().view(source.view()).triangles()==2);
    // Borrow a compact predecessor's index space after the original source IDs
    // have changed. Retrangulating the fan preserves coverage and forces
    // ownership conversion; no borrowed view may outlive its arena payload.
    s.levels=3;s.candidate_budget=8;s.research.graph_passes=2;s.max_added_vertex_bytes_bps=100000;
    unsigned progressive_borrows=0;
    auto borrowed=generate(source.view(),s,[&](MeshView input,const ReduceSettings&) {
        Lod l;if(input.positions.count==source.positions.size()) {l.shared_vertices=false;l.data=bridge(0);}
        else {++progressive_borrows;l.data.indices={0,1,3,0,3,2};}
        return l;
    });
    CHECK(progressive_borrows>0&&borrowed.status==Status::Complete);
    for(auto& level:borrowed.lods)CHECK(validate(level.view(source.view())).empty());
    std::cout<<"chain graph contracts passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
