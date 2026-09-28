#include "neural_internal.hpp"
#include "chain_hooks.hpp"
#include "blitz/render_cost.hpp"
#include "blitz/blitz.h"
#include <iostream>
#include <fstream>
#include <numeric>
using namespace blitz;
void require(bool good,const char* message){if(!good)throw std::runtime_error(message);}
Mesh grid(unsigned side=9) {
    Mesh m;for(unsigned y=0;y<side;++y)for(unsigned x=0;x<side;++x){m.positions.push_back({float(x)/side,float(y)/side,0});m.normals.push_back({0,0,1});m.uv.push_back({float(x)/side,float(y)/side});m.colors.push_back({uint8_t(x*7),uint8_t(y*9),128,255});}
    for(unsigned y=0;y+1<side;++y)for(unsigned x=0;x+1<side;++x){uint32_t a=y*side+x;m.indices.insert(m.indices.end(),{a,a+1,a+side,a+1,a+side+1,a+side});}m.double_sided={1};return m;
}
void graph_contracts() {
    auto m=grid();auto original=copy_mesh(m.view());auto g=neural::graph(m.view());neural::Prediction p;p.values.resize(g.size()*neural::outputs);
    for(size_t i=0;i<g.size();++i){p.values[i*4]=float(i);p.values[i*4+1]=float(1./(9*bounds(m.view()).diameter()));}
    for(size_t target:{size_t(40),size_t(80)}){
        auto l=neural::decode(m.view(),g,p,target,OutputMode::Reuse);require(l.shared_vertices,"reuse lost ownership contract");require(l.data.indices.size()<m.indices.size(),"grid did not reduce");require(l.view(m.view()).triangles()>=target,"decoder undershot target");
        auto again=neural::decode(m.view(),g,p,target,OutputMode::Reuse);require(l.data.indices==again.data.indices,"decoder is nondeterministic");
        require(uv_distortion(l.view(m.view())).negative_uv_faces==0,"decoder flipped UVs");require(validate(l.view(m.view())).empty(),"decoder emitted invalid mesh");
        for(uint32_t i=0;i<g.size();++i)if(g.flags[i]&neural::Boundary)require(std::find(l.data.indices.begin(),l.data.indices.end(),i)!=l.data.indices.end(),"lost boundary vertex");
        auto rebuilt=neural::decode(m.view(),g,p,target,OutputMode::Rebuild);require(!rebuilt.shared_vertices&&validate(rebuilt.data.view()).empty(),"compact output invalid");
    }
    require(same_mesh_data(m.view(),original.view()),"input mutated");
    auto exact=neural::decode(m.view(),g,p,m.view().triangles(),OutputMode::Reuse);require(same_mesh_data(exact.view(m.view()),m.view()),"identity changed input");
    uint32_t core=40;auto patch=neural::patch(g,{&core,1});require(patch.core==1&&patch.ids[0]==core,"patch core order");
    std::vector<uint32_t> frontier{core};for(unsigned hop=0;hop<3;++hop){auto next=frontier;for(auto v:frontier)for(auto k=g.offsets[v];k<g.offsets[v+1];++k)next.push_back(g.neighbors[k]);frontier=next;}
    for(auto v:frontier)require(std::find(patch.ids.begin(),patch.ids.end(),v)!=patch.ids.end(),"three-hop halo truncated");
    bool threw=false;p.values[0]=NAN;try{neural::decode(m.view(),g,p,20,OutputMode::Reuse);}catch(const std::invalid_argument&){threw=true;}require(threw,"accepted NaN prediction");
    auto duplicate=m.positions.size();m.positions.push_back(m.positions[40]);m.normals.push_back(m.normals[40]);m.uv.push_back(m.uv[40]);m.colors.push_back(m.colors[40]);
    auto merged=neural::graph(m.view());require(merged.canonical[duplicate]==40,"identical attribute vertex not canonicalized");m.uv.back().x+=.5f;
    auto seam=neural::graph(m.view());require(seam.flags[40]&neural::Seam,"UV seam unlocked");
    Mesh tetra;tetra.positions={{0,0,0},{1,0,0},{0,1,0},{0,0,1}};tetra.indices={0,2,1,0,1,3,0,3,2,1,2,3};auto tg=neural::graph(tetra.view());neural::Prediction tp;tp.values.resize(16);for(unsigned i=0;i<4;++i)tp.values[i*4]=float(i);
    auto intact=neural::decode(tetra.view(),tg,tp,2,OutputMode::Reuse);require(intact.data.indices==tetra.indices,"tetrahedron collapsed to duplicate faces");
    // Cancellation while confirming a GPU finalist must retain the exact incumbent
    // and report cancellation even when the minimum-triangle reference was rejected.
    Settings s;s.levels=2;s.base_pixels=24;s.last_pixels=12;s.candidate_budget=2;s.beam_width=2;bool stop=false;s.cancelled=[&]{return stop;};detail::GenerationHooks hooks;
    hooks.propose=[](MeshView input,const ReduceSettings& rs,const EvalSettings&,double){return reduce(input,rs);};
    hooks.evaluate=[](MeshView,MeshView,const Bounds&,const EvalSettings&){return Measurement{};};hooks.confirm=[&](Result&){stop=true;return false;};
    auto cancelled=detail::generate_with_hooks(original.view(),s,{},&hooks);require(cancelled.status==Status::Cancelled&&same_mesh_data(cancelled.lods.back().view(original.view()),original.view()),"confirmation cancellation lost status or incumbent");
}
#ifdef BLITZ_CUDA
void cuda_contracts() {
    auto m=grid(5);auto b=bounds(m.view());NeuralOptions options;options.memory_mib=512;
    for(double screen:{16.,33.})for(auto c:cameras(b,screen,{3,2,42}))for(uint8_t ss:{uint8_t(1),uint8_t(4)}) {
        auto cpu=rasterize(m.view(),b,c,screen,ss,false);auto gpu=neural::raster_cuda(m.view(),b,c,screen,ss,false,options);require(cpu.pixels.size()==gpu.pixels.size(),"raster shape differs");
        for(size_t i=0;i<cpu.pixels.size();++i){auto a=cpu.pixels[i],d=gpu.pixels[i];require(a.covered==d.covered&&a.visible==d.visible,"CUDA coverage/visibility mismatch");if(a.visible){require(std::abs(a.depth-d.depth)<1e-6,"CUDA depth mismatch");require(length(a.normal-d.normal)<1e-5,"CUDA normal mismatch");require(std::abs(a.color.x-d.color.x)<1e-5,"CUDA color mismatch");}}
    }
    // Overlapping coplanar faces stress deterministic depth ties and material ownership.
    auto overlap=m;auto vertex_count=uint32_t(m.positions.size());
    for(auto v:m.positions)overlap.positions.push_back(v);for(auto n:m.normals)overlap.normals.push_back(n*-1);
    overlap.uv.insert(overlap.uv.end(),m.uv.begin(),m.uv.end());overlap.colors.insert(overlap.colors.end(),m.colors.begin(),m.colors.end());
    for(auto id:m.indices)overlap.indices.push_back(id+vertex_count);overlap.materials.resize(m.view().triangles(),0);overlap.materials.resize(overlap.view().triangles(),1);overlap.double_sided={1,1};
    for(auto camera:cameras(b,17,{3,2,313})) {
        auto cpu=rasterize(overlap.view(),b,camera,17,4,false),gpu=neural::raster_cuda(overlap.view(),b,camera,17,4,false,options);
        for(size_t i=0;i<cpu.pixels.size();++i)if(cpu.pixels[i].visible){require(cpu.pixels[i].material==gpu.pixels[i].material,"coplanar material ownership differs");require(length(cpu.pixels[i].normal-gpu.pixels[i].normal)<1e-5,"coplanar normal ownership differs");}
    }
    auto altered=m;altered.positions[12].z=.03f;altered.normals[12]=normalized({.1f,0,1});
    for(auto profile:{Profile::Coverage,Profile::Normals,Profile::Attributes})for(double limit:{.3,2.,4.}) {
        EvalSettings e;e.profile=profile;e.limit=limit;e.screen_size=24;e.views={5,2,321};e.supersample=2;e.max_supersample=8;e.max_changed_area=.5;
        auto cpu=evaluate(m.view(),altered.view(),b,e),gpu=evaluate_cuda(m.view(),altered.view(),b,e,options);
        require(cpu.passed==gpu.passed&&cpu.complete==gpu.complete,"CUDA decision mismatch");require(cpu.views_evaluated==gpu.views_evaluated,"CUDA view order mismatch");
        require((!std::isfinite(cpu.error)&&!std::isfinite(gpu.error))||std::abs(cpu.error-gpu.error)<1e-5,"CUDA metric mismatch");require(cpu.changed_area==gpu.changed_area,"CUDA changed area mismatch");
    }
    neural::WeightsData weights;weights.values.resize(neural::weight_count,.001f);weights.provenance="test";
    auto path=std::filesystem::temp_directory_path()/"blitz-neural-contract.blzn";neural::save_weights(path,weights);auto restored=neural::load_weights(path);require(restored.values==weights.values,"model roundtrip");
    NeuralModel model(path.c_str(),options);Settings config;config.levels=3;config.base_pixels=24;config.last_pixels=12;config.profile=Profile::Attributes;config.candidate_budget=2;config.beam_width=1;config.search_views={2,1,14};config.audit_views={3,1,17};config.search_supersample=2;config.audit_supersample=4;config.max_supersample=8;
    auto result=generate_neural(m.view(),config,model);require(result.lods.size()==3,"neural chain missing levels");for(auto& l:result.lods)require(l.adjacent.passed&&l.source_error.passed,"neural chain bypassed audits");
    config.cancelled=[]{return true;};result=generate_neural(m.view(),config,model);require(result.status==Status::Cancelled&&same_mesh_data(result.lods.back().view(m.view()),m.view()),"neural cancellation lost exact incumbent");
    blitz_neural_options coptions;blitz_neural_options_init(&coptions,sizeof(coptions));blitz_neural_model* cmodel=nullptr;char error[256];
    require(blitz_neural_model_load(path.c_str(),&coptions,&cmodel,error,sizeof(error))==BLITZ_OK,"C model load failed");require(blitz_neural_model_sha256(cmodel)!=nullptr,"C model hash missing");blitz_neural_model_destroy(cmodel);
    {std::fstream f(path,std::ios::in|std::ios::out|std::ios::binary);f.seekp(50);f.put('x');}bool rejected=false;try{neural::load_weights(path);}catch(const std::invalid_argument&){rejected=true;}std::filesystem::remove(path);require(rejected,"model corruption accepted");
    auto g=neural::graph(m.view());auto encoded=neural::encode_cuda(g,weights,options);require(encoded.size()==g.size()*neural::hidden,"embedding shape");auto prediction=neural::predict_cuda(encoded,{},weights,options);require(prediction.values.size()==g.size()*4,"head shape");
    auto single=overlap_cuda(m.view(),b,24,{1,0,0xB1172026},options);auto twice=m;twice.indices.insert(twice.indices.end(),m.indices.begin(),m.indices.end());auto doubled=overlap_cuda(twice.view(),b,24,{1,0,0xB1172026},options);require(std::abs(doubled-2*single)<1e-8,"overdraw counts do not scale");
    for(double screen:{16.,25.}){auto costs=render_cost(twice.view(),b,ViewSet{1,0,0xB1172026},screen);auto cuda=overlap_cuda(twice.view(),b,screen,{1,0,0xB1172026},options);require(costs.covered_pixels&&std::abs(cuda-double(costs.covered_samples)/costs.covered_pixels)<1e-8,"overdraw differs from CPU proxy");}
    EvalSettings stopped;stopped.cancelled=[]{return true;};auto cancelled=evaluate_cuda(m.view(),altered.view(),b,stopped,options);require(!cancelled.complete&&!cancelled.passed,"CUDA cancellation ignored");
    EvalSettings huge;huge.screen_size=1024;huge.supersample=8;huge.max_supersample=8;huge.views={1,0,42};auto limited=evaluate_cuda(m.view(),altered.view(),b,huge,options);require(limited.resource_limited&&!limited.passed&&!limited.complete,"CUDA resource failure became acceptance");
}
#endif
int main(int argc,char**) {try {
    if(argc==1){graph_contracts();if(!neural_available()){bool rejected=false;try{NeuralModel unavailable("missing.blzn");}catch(const NeuralUnavailable&){rejected=true;}require(rejected,"missing CUDA silently accepted");}}
#ifdef BLITZ_CUDA
    else {if(!neural_available())return 77;cuda_contracts();}
#endif
    std::cout<<"neural contracts passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
