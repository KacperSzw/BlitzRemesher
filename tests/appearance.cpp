#include "blitz/remesher.hpp"
#include "appearance.hpp"
#include <iostream>
#include <stdexcept>
using namespace blitz;
#define CHECK(x) do{if(!(x))throw std::runtime_error(std::string(__FILE__)+":"+std::to_string(__LINE__)+": " #x);}while(0)
static Mesh patch(bool diagonal) {
    Mesh m;m.positions={{-1,-1,0},{1,-1,0},{1,1,0},{-1,1,0}};
    m.normals.assign(4,{0,0,1});m.colors.assign(4,{40,90,120,255});m.double_sided={1};
    m.indices=diagonal?std::vector<uint32_t>{0,1,2,0,2,3}:std::vector<uint32_t>{0,1,3,1,2,3};return m;
}
int main(){try {
    // Independently computed affine RGB residual on one right triangle.
    Mesh triangle;triangle.positions={{0,0,0},{1,0,0},{0,1,0}};triangle.indices={0,1,2};
    triangle.colors={{0,0,0,255},{254,0,0,255},{0,0,0,255}};
    ReduceSettings metric;metric.appearance_stage=AppearanceStage::Ordering;metric.appearance_weights={0,1,0};
    const uint32_t ids[]={0,1,2};detail::Appearance order(triangle.view(),metric,ids,3);
    auto tb=bounds(triangle.view());auto p0=(triangle.positions[0]-tb.center)*(1/tb.diameter()),p1=(triangle.positions[1]-tb.center)*(1/tb.diameter());
    auto mid=(p0+p1)*.5;double expected=(1/(tb.diameter()*tb.diameter()))*2*std::pow(127./255,2);
    CHECK(std::abs(order.cost(0,1,mid,p0,p1)-expected)<1e-7);
    metric.appearance_stage=AppearanceStage::Attributes;detail::Appearance fit(triangle.view(),metric,ids,3);
    CHECK(fit.cost(0,1,mid,p0,p1)<1e-12);CHECK(fit.bytes()>0);
    // Equivalent rendered surface through distinct index streams bypasses identity.
    auto a=patch(false),b=patch(true);auto ref=bounds(a.view());
    for(unsigned ss:{2,4,8}) {
        EvalSettings e;e.profile=Profile::Attributes;e.views={5,2,73};e.screen_size=16;e.supersample=e.max_supersample=uint8_t(ss);e.limit=2;
        CHECK(evaluate(a.view(),b.view(),ref,e).passed);
        auto bad=b;bad.colors.assign(4,{255,255,255,255});
        EvaluationWitness witness;e.witness=&witness;
        CHECK(!evaluate(a.view(),bad.view(),ref,e).passed);CHECK(witness.present&&witness.color>4);
        e.conservative_screen=true;CHECK(evaluate(a.view(),bad.view(),ref,e).passed);CHECK(!witness.present);
        auto away=b;for(auto& p:away.positions)p.x+=.75f;
        e.limit=.1;CHECK(!evaluate(a.view(),away.view(),ref,e).passed);
    }
    // Coarse false rejection / fine acceptance and the reverse are both legal.
    // Controlled visible samples isolate attribute correspondence from coverage.
    Raster a0,b0;a0.width=b0.width=3;a0.height=b0.height=1;a0.pixels.resize(3);b0.pixels.resize(3);
    a0.pixels[1].visible=1;a0.pixels[1].normal={0,0,1};
    EvalSettings e;e.profile=Profile::Normals;e.weights.normal=1;e.supersample=1;
    CHECK(!std::isfinite(attributed_distance(a0,b0,e,1)));
    b0.pixels[1]=a0.pixels[1];CHECK(attributed_distance(a0,b0,e,1)==0);
    a0.pixels[0]=a0.pixels[1];a0.pixels[0].normal={0,1,0};
    CHECK(!std::isfinite(attributed_distance(a0,b0,e,1)));
    // Public raster comparisons accept rectangular caller-owned buffers; their
    // witness coordinates are not restricted by rasterize()'s square-view cap.
    Raster wide,missing;wide.width=missing.width=70000;wide.height=missing.height=1;
    wide.pixels.resize(70000);missing.pixels.resize(70000);wide.pixels.back().visible=1;wide.pixels.back().normal={0,0,1};
    EvaluationWitness wide_witness;e.witness=&wide_witness;
    CHECK(!std::isfinite(attributed_distance(wide,missing,e,1)));CHECK(wide_witness.present&&wide_witness.x==69999);
    e.witness=nullptr;
    // Cancellation and resource limits cannot be interpreted as deferred success.
    e.conservative_screen=true;e.screen_size=16;e.views={2,0,7};e.cancelled=[] {return true;};
    auto m=evaluate(a.view(),b.view(),ref,e);CHECK(!m.passed&&!m.complete);
    e.cancelled={};e.screen_size=4096;e.supersample=e.max_supersample=8;
    m=evaluate(a.view(),b.view(),ref,e);CHECK(!m.passed&&m.resource_limited);
    // A transient cancellation inside the attribute loop is an interruption,
    // not an infinite-error failure witness, even when the next poll is false.
    e=EvalSettings{};e.profile=Profile::Attributes;e.screen_size=16;e.views={2,0,7};e.supersample=e.max_supersample=4;e.limit=2;
    unsigned polls=0;e.cancelled=[&]{return ++polls==2;};
    auto cancelled_sample=evaluate(a.view(),b.view(),ref,e);
    CHECK(cancelled_sample.cancelled&&!cancelled_sample.complete&&!cancelled_sample.passed&&cancelled_sample.views_evaluated==0);
    // A smooth attributed grid must reduce under each stage; output ownership,
    // normal normalization, alpha and immutable input are independent contracts.
    Mesh grid;grid.double_sided={1};constexpr unsigned n=6;
    for(unsigned y=0;y<=n;++y)for(unsigned x=0;x<=n;++x){float u=float(x)/n,v=float(y)/n;
        grid.positions.push_back({u,v,.15f*u*u});grid.normals.push_back(normalized({-.3f*u,0,1}));
        grid.colors.push_back(quantize_color(u,v,.25,.5));grid.uv.push_back({u,v});grid.tangents.push_back({1,0,.3f*u,-1});}
    for(unsigned y=0;y<n;++y)for(unsigned x=0;x<n;++x){auto i=y*(n+1)+x;grid.indices.insert(grid.indices.end(),{i,i+1,i+n+2,i,i+n+2,i+n+1});}
    const auto original=grid;
    for(auto stage:{AppearanceStage::Ordering,AppearanceStage::Attributes,AppearanceStage::Position})for(auto mode:{OutputMode::Reuse,OutputMode::Rebuild}) {
        ReduceSettings rs;rs.appearance_stage=stage;rs.output=mode;rs.screen_size=32;rs.target_triangles=24;
        auto l=reduce(grid.view(),rs);auto v=l.view(grid.view());CHECK(validate(v).empty());CHECK(v.triangles()<grid.view().triangles());
        CHECK(same_mesh_data(grid.view(),original.view()));
        if(mode==OutputMode::Reuse){CHECK(l.shared_vertices);CHECK(v.positions.data==grid.view().positions.data);}
        else for(size_t i=0;i<v.positions.count;++i){CHECK(std::abs(length(v.normals[i])-1)<1e-5);CHECK(v.colors[i].a==128);CHECK(v.tangents[i].w==-1);
            if(stage>=AppearanceStage::Attributes){auto t=v.tangents[i];CHECK(std::abs(dot(v.normals[i],{t.x,t.y,t.z}))<1e-5);}}
        rs.cancelled=[] {return true;};auto cancelled=reduce(grid.view(),rs);CHECK(same_mesh_data(cancelled.view(grid.view()),grid.view()));
    }
    // Separate attribute wedges survive positional contractions without mixing.
    Mesh split;split.double_sided={1};for(auto index:grid.indices){split.indices.push_back(uint32_t(split.positions.size()));split.positions.push_back(grid.positions[index]);
        split.normals.push_back(grid.normals[index]);split.colors.push_back(grid.colors[index]);split.uv.push_back(grid.uv[index]);}
    for(auto stage:{AppearanceStage::Ordering,AppearanceStage::Attributes,AppearanceStage::Position}) {
        ReduceSettings rs;rs.coupled_wedges=true;rs.appearance_stage=stage;rs.target_triangles=24;rs.screen_size=16;
        auto l=reduce(split.view(),rs);CHECK(validate(l.view(split.view())).empty());CHECK(l.data.indices.size()<split.indices.size());
        CHECK(uv_distortion(l.view(split.view())).negative_uv_faces==0);
    }
    // Coincident disconnected wedges carry different constant fields; fitting
    // must retain both values rather than creating a blended seam color.
    auto seams=split;
    for(size_t i=0;i<seams.positions.size();++i){bool side=seams.positions[i].x>.5f;
        seams.colors[i]=side?ColorRGBA8{255,0,0,77}:ColorRGBA8{0,0,255,77};}
    // Make each face a constant discontinuous field, including boundary corners.
    for(size_t f=0;f<seams.indices.size()/3;++f){auto c=seams.colors[seams.indices[3*f]];for(unsigned k=0;k<3;++k)seams.colors[seams.indices[3*f+k]]=c;}
    ReduceSettings seam_settings;seam_settings.coupled_wedges=true;seam_settings.appearance_stage=AppearanceStage::Attributes;
    seam_settings.target_triangles=36;seam_settings.screen_size=16;
    auto seam_lod=reduce(seams.view(),seam_settings);auto sv=seam_lod.view(seams.view());bool red=false,blue=false;
    for(size_t i=0;i<sv.colors.count;++i){auto c=sv.colors[i];CHECK(c==ColorRGBA8({255,0,0,77})||c==ColorRGBA8({0,0,255,77}));red|=c.r==255;blue|=c.b==255;}
    CHECK(red&&blue);
    // Disabled fields allocate no appearance storage and retain baseline bytes.
    ReduceSettings zero;zero.target_triangles=24;zero.appearance_weights={0,0,0};ReductionStats stats;zero.statistics=&stats;
    auto plain=reduce(grid.view(),zero);zero.appearance_stage=AppearanceStage::Position;auto empty_fields=reduce(grid.view(),zero);
    CHECK(stats.appearance_bytes==0&&same_mesh_data(plain.view(grid.view()),empty_fields.view(grid.view())));
    std::cout<<"appearance contracts passed\n";
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
