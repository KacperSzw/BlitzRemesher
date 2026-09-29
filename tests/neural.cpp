#include "neural_internal.hpp"
#include "neural_numeric.hpp"
#include "neural_action.hpp"
#include "metric_angle.hpp"
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
    for(int i=-10000;i<=10000;++i){double x=i/10000.,value=detail::metric_acos(x);require(std::abs(value-std::acos(x))<=9e-16,"shared angular metric differs from independent libm oracle");}
    require(detail::metric_acos(1)==0&&detail::metric_acos(-1)==std::acos(-1.),"angular metric endpoint contract");
    for(double bad:{NAN,INFINITY,-INFINITY}){std::array<double,2>a{0,bad},b{};require(!neural::legacy_numeric_pass(neural::numeric_difference(a,b)),"nonfinite prediction escaped numeric gate");}
    std::array<double,1>a{0},b{.0003};require(!neural::legacy_numeric_pass(neural::numeric_difference(a,b)),"legacy tolerance silently widened");
    auto numeric_graph=neural::graph(grid(4).view());neural::WeightsData w;w.values.resize(neural::weight_count);size_t at=0;
    for(unsigned l=0;l<5;++l){for(unsigned j=0;j<neural::layer_out[l];++j)w.values[at+size_t(neural::layer_in[l])*neural::layer_out[l]+j]=float(l+1);at+=size_t(neural::layer_out[l])*(neural::layer_in[l]+1);}
    auto trace=neural::reference_fp64(numeric_graph,{},w);for(auto value:trace.back().values)require(value==5,"FP64 bias oracle wrong");
    std::fill(w.values.begin(),w.values.end(),0);at=0;
    for(unsigned l=0;l<5;++l){w.values[at+(l==0?neural::features:0)]=1;at+=size_t(neural::layer_out[l])*(neural::layer_in[l]+1);}
    neural::Graph directed;directed.flags.resize(3);directed.offsets={0,2,3,3};directed.neighbors={1,2,0};directed.x.resize(3*neural::features);
    directed.x[0]=2;directed.x[neural::features]=4;directed.x[2*neural::features]=8;
    auto numeric_result=neural::reference_fp64(directed,{},w).back().values;
    require(numeric_result[0]==6&&numeric_result[4]==2&&numeric_result[8]==0,"oracle changed directed mean or isolated-vertex semantics");
    auto full=neural::reference_fp64(numeric_graph,{},w).back().values;uint32_t root_id=5;auto sub=neural::patch(numeric_graph,{&root_id,1});
    auto partial=neural::reference_fp64(sub.graph,{},w).back().values;require(full[root_id*4]==partial[0],"three-hop core differs from full graph");
    auto corrupt=numeric_graph;corrupt.neighbors.front()=uint32_t(corrupt.size());bool invalid=false;try{neural::validate_graph(corrupt);}catch(const std::invalid_argument&){invalid=true;}require(invalid,"numeric oracle accepted invalid adjacency");
    for(auto [screen,initial,maximum,expected]:{std::tuple{312.0674954763457,4,32,16},std::tuple{242.,8,32,32},std::tuple{242.01,8,32,16},std::tuple{350.,3,32,12},std::tuple{200.,3,23,23},std::tuple{1024.,8,32,8}}) {
        EvalSettings e;e.screen_size=screen;e.supersample=uint8_t(initial);e.max_supersample=uint8_t(maximum);
        require(neural::bounded_refinement(e)==expected,"bounded refinement lost sequence, sample boundary or initial resolution");
        require(e.max_supersample==maximum,"bounded refinement mutated settings");
    }
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
#include <cuda_runtime_api.h>
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
    // Empty renders are valid even though mesh streams themselves are nonempty.
    // Exercise culling, forced two-sided rendering and out-of-frame clipping.
    Camera front{{1,0,0},{0,1,0},{0,0,1},4*b.radius,1,20/b.diameter(),false};
    auto back=m;back.double_sided={0};for(size_t i=0;i<back.indices.size();i+=3)std::swap(back.indices[i+1],back.indices[i+2]);
    for(bool two:{false,true}){
        auto cpu=rasterize(back.view(),b,front,20,2,two),gpu=neural::raster_cuda(back.view(),b,front,20,2,two,options);size_t covered=0;
        for(size_t i=0;i<cpu.pixels.size();++i){covered+=cpu.pixels[i].covered;require(cpu.pixels[i].covered==gpu.pixels[i].covered&&cpu.pixels[i].visible==gpu.pixels[i].visible,"backface culling CPU/GPU mismatch");}
        require(two?covered>0:covered==0,"culling fixture did not exercise empty and visible renders");
    }
    auto outside=m;for(auto& p:outside.positions)p.x+=float(4*b.diameter());
    auto clipped_cpu=rasterize(outside.view(),b,front,20,2,false),clipped_gpu=neural::raster_cuda(outside.view(),b,front,20,2,false,options);
    require(clipped_cpu.clipped&&clipped_gpu.clipped,"clipping flag was lost during bin readback");
    auto empty=m,other_empty=m;empty.indices={0,0,0};other_empty.indices={1,1,1};
    EvalSettings empty_settings;empty_settings.views={2,1,833};empty_settings.screen_size=20;empty_settings.supersample=2;empty_settings.max_supersample=4;empty_settings.limit=3;
    for(auto pair:{std::pair{empty.view(),other_empty.view()},std::pair{m.view(),empty.view()}}){
        auto cpu=evaluate(pair.first,pair.second,b,empty_settings),gpu=evaluate_cuda(pair.first,pair.second,b,empty_settings,options);
        require(cpu.passed==gpu.passed&&cpu.complete==gpu.complete&&cpu.changed_area==gpu.changed_area,"empty-render audit CPU/GPU mismatch");
    }
    for(auto profile:{Profile::Coverage,Profile::Normals,Profile::Attributes})for(double limit:{.3,2.,4.}) {
        EvalSettings e;e.profile=profile;e.limit=limit;e.screen_size=24;e.views={5,2,321};e.supersample=2;e.max_supersample=8;e.max_changed_area=.5;
        auto cpu=evaluate(m.view(),altered.view(),b,e),gpu=evaluate_cuda(m.view(),altered.view(),b,e,options);
        require(cpu.passed==gpu.passed&&cpu.complete==gpu.complete,"CUDA decision mismatch");require(cpu.views_evaluated==gpu.views_evaluated,"CUDA view order mismatch");
        require((!std::isfinite(cpu.error)&&!std::isfinite(gpu.error))||std::abs(cpu.error-gpu.error)<1e-5,"CUDA metric mismatch");require(cpu.changed_area==gpu.changed_area,"CUDA changed area mismatch");
    }
    // Holes and unequal edges exercise both distance-transform directions, empty
    // scan lines and lower-envelope backtracking. Extents straddle tile/block
    // boundaries; compare exact float-distance results with the CPU evaluator.
    for(unsigned pattern=0;pattern<3;++pattern) {
        auto cut=m;cut.indices.clear();
        for(size_t face=0;face<m.view().triangles();++face)
            if((face*7+pattern*3)%11>pattern+1)cut.indices.insert(cut.indices.end(),m.indices.begin()+face*3,m.indices.begin()+face*3+3);
        for(double screen:{23.,24.,25.,57.,58.,59.,121.}) {
            EvalSettings e;e.profile=Profile::Coverage;e.screen_size=screen;e.views={3,1,1901+pattern};e.supersample=e.max_supersample=uint8_t(pattern+1);e.limit=100;
            auto cpu=evaluate(m.view(),cut.view(),b,e),gpu=evaluate_cuda(m.view(),cut.view(),b,e,options);
            require(cpu.coverage>0&&cpu.complete&&gpu.complete,"distance fixture must finish with a nonzero distance");
            require(cpu.coverage==gpu.coverage&&cpu.coverage_upper==gpu.coverage_upper&&cpu.changed_area==gpu.changed_area&&cpu.passed==gpu.passed,"distance-transform CPU/GPU result differs");
        }
    }
    // Reused workspace and topology must not reuse stale contents or settings.
    auto endpoint=m.view();std::vector<uint32_t> endpoint_indices(m.indices.begin()+3,m.indices.end());endpoint.indices=endpoint_indices;
    neural::AuditCuda workspace(options,m.view());EvalSettings reuse;reuse.screen_size=20;reuse.views={3,1,765};reuse.supersample=2;reuse.max_supersample=4;reuse.limit=4;
    NeuralStats reuse_stats;auto first=workspace.evaluate(m.view(),endpoint,b,reuse,&reuse_stats);require(first.complete&&first.passed,"workspace fixture must pass");
    auto uploads=reuse_stats.gpu_upload_bytes,allocations=reuse_stats.gpu_allocations;
    auto again=workspace.evaluate(m.view(),endpoint,b,reuse,&reuse_stats);
    require(again.error==first.error&&reuse_stats.gpu_measurement_cache_hits==1&&reuse_stats.gpu_upload_bytes==uploads&&reuse_stats.gpu_allocations==allocations,"completed audit repeated GPU work");
    reuse.limit=3;workspace.evaluate(m.view(),endpoint,b,reuse,&reuse_stats);
    require(reuse_stats.gpu_upload_bytes==uploads&&reuse_stats.gpu_buffer_reuses>0,"unchanged topology was re-uploaded");
    endpoint_indices.erase(endpoint_indices.begin(),endpoint_indices.begin()+3);endpoint.indices=endpoint_indices;
    auto changed=workspace.evaluate(m.view(),endpoint,b,reuse,&reuse_stats),expected=evaluate(m.view(),endpoint,b,reuse);
    require(reuse_stats.gpu_upload_bytes>uploads&&changed.passed==expected.passed&&changed.complete==expected.complete,"topology revision reused old device data");
    reuse.cancelled=[]{return true;};auto stopped_cached=workspace.evaluate(m.view(),endpoint,b,reuse,&reuse_stats);
    require(!stopped_cached.passed&&!stopped_cached.complete,"cache bypassed cancellation");
    // Thresholds come from controlled measurements. Check both sides of exact
    // pixel and area boundaries; tolerances must never excuse a decision mismatch.
    auto threshold_mesh=m;for(auto& normal:threshold_mesh.normals)normal=normalized({.3f,0,1});for(auto& color:threshold_mesh.colors)color.r+=32;
    for(auto profile:{Profile::Coverage,Profile::Normals,Profile::Attributes}){
        EvalSettings e;e.profile=profile;e.screen_size=20;e.views={1,0,71423};e.supersample=4;e.max_supersample=4;e.limit=10;e.max_changed_area=1;
        auto value=evaluate(m.view(),threshold_mesh.view(),b,e);require(std::isfinite(value.error),"threshold fixture error is not finite");
        for(double threshold:{std::nextafter(value.error,0.),value.error,std::nextafter(value.error,INFINITY)}){
            e.limit=threshold;auto cpu=evaluate(m.view(),threshold_mesh.view(),b,e),gpu=evaluate_cuda(m.view(),threshold_mesh.view(),b,e,options);
            if(cpu.passed!=gpu.passed||cpu.complete!=gpu.complete){auto loose=e;loose.limit=10;auto g=evaluate_cuda(m.view(),threshold_mesh.view(),b,loose,options);std::cerr.precision(17);std::cerr<<"profile="<<unsigned(profile)<<" threshold="<<threshold<<" cpu="<<cpu.error<<" gpu="<<gpu.error<<" unbounded_gpu="<<g.error<<" coverage="<<cpu.coverage_upper<<","<<gpu.coverage_upper<<" normal="<<cpu.normal_degrees<<","<<gpu.normal_degrees<<'\n';
            for(auto camera:cameras(b,e.screen_size,e.views)){auto cr=rasterize(threshold_mesh.view(),b,camera,e.screen_size,e.supersample,false),gr=neural::raster_cuda(threshold_mesh.view(),b,camera,e.screen_size,e.supersample,false,options);size_t mismatch=0;for(size_t i=0;i<cr.pixels.size();++i)if(cr.pixels[i].visible&&std::memcmp(&cr.pixels[i].normal,&gr.pixels[i].normal,sizeof(Vec3)))++mismatch;std::cerr<<"normal raster byte mismatches="<<mismatch<<'\n';}}
            require(cpu.passed==gpu.passed&&cpu.complete==gpu.complete,"pixel boundary CPU/GPU decision mismatch");}
        e.limit=10;auto area=evaluate(m.view(),endpoint,b,e).changed_area;require(area>0&&area<1,"area boundary fixture must change coverage");
        for(double threshold:{std::max(0.,std::nextafter(area,0.)),area,std::min(1.,std::nextafter(area,INFINITY))}){
            e.max_changed_area=threshold;auto cpu=evaluate(m.view(),endpoint,b,e),gpu=evaluate_cuda(m.view(),endpoint,b,e,options);
            require(cpu.passed==gpu.passed&&cpu.changed_area==gpu.changed_area,"area boundary CPU/GPU decision mismatch");}
    }
    // Caller device state survives the workspace's construction, calls and destruction.
    int original_device=0,device_count=0;cudaGetDevice(&original_device);cudaGetDeviceCount(&device_count);
    int caller=device_count>1?(original_device+1)%device_count:original_device;cudaSetDevice(caller);
    {neural::AuditCuda other(options,m.view());other.evaluate(m.view(),altered.view(),b,EvalSettings{.views={2,1,73},.supersample=2,.max_supersample=4,.screen_size=16,.limit=3},nullptr);int observed=-1;cudaGetDevice(&observed);require(observed==caller,"audit changed caller CUDA device");}
    int restored_device=-1;cudaGetDevice(&restored_device);require(restored_device==caller,"audit destruction changed caller CUDA device");cudaSetDevice(original_device);
    neural::WeightsData weights;weights.values.resize(neural::weight_count,.001f);weights.provenance="test";
    auto path=std::filesystem::temp_directory_path()/"blitz-neural-contract.blzn";neural::save_weights(path,weights);auto restored=neural::load_weights(path);require(restored.values==weights.values,"model roundtrip");
    NeuralModel model(path.c_str(),options);Settings config;config.levels=3;config.base_pixels=24;config.last_pixels=12;config.profile=Profile::Attributes;config.candidate_budget=2;config.beam_width=1;config.search_views={2,1,14};config.audit_views={3,1,17};config.search_supersample=2;config.audit_supersample=4;config.max_supersample=8;
    auto result=generate_neural(m.view(),config,model);require(result.lods.size()==3,"neural chain missing levels");for(auto& l:result.lods)require(l.adjacent.passed&&l.source_error.passed,"neural chain bypassed audits");
    config.cancelled=[]{return true;};result=generate_neural(m.view(),config,model);require(result.status==Status::Cancelled&&same_mesh_data(result.lods.back().view(m.view()),m.view()),"neural cancellation lost exact incumbent");
    blitz_neural_options coptions;blitz_neural_options_init(&coptions,sizeof(coptions));blitz_neural_model* cmodel=nullptr;char error[256];
    require(blitz_neural_model_load(path.c_str(),&coptions,&cmodel,error,sizeof(error))==BLITZ_OK,"C model load failed");require(blitz_neural_model_sha256(cmodel)!=nullptr,"C model hash missing");blitz_neural_model_destroy(cmodel);
    {std::fstream f(path,std::ios::in|std::ios::out|std::ios::binary);f.seekp(50);f.put('x');}bool rejected=false;try{neural::load_weights(path);}catch(const std::invalid_argument&){rejected=true;}std::filesystem::remove(path);require(rejected,"model corruption accepted");
    auto g=neural::graph(m.view());auto encoded=neural::encode_cuda(g,weights,options);require(encoded.size()==g.size()*neural::hidden,"embedding shape");auto prediction=neural::predict_cuda(encoded,{},weights,options);require(prediction.values.size()==g.size()*4,"head shape");
    neural::WeightsData action_weights;action_weights.architecture=neural::action_schema;action_weights.values.resize(neural::action_weight_count,.001f);action_weights.provenance="v2 test";
    neural::save_weights(path,action_weights);auto action_restored=neural::load_weights(path);require(action_restored.architecture==neural::action_schema&&action_restored.values==action_weights.values,"action architecture roundtrip");
    options.action_trials=2;NeuralModel action_model(path.c_str(),options);config.cancelled={};NeuralStats action_stats;
    auto action_result=generate_neural(m.view(),config,action_model,&action_stats);require(action_result.lods.size()==3&&action_stats.action_ranked>0&&action_stats.action_trials>0,"v2 inference not wired into chain");
    for(auto& l:action_result.lods)require(l.adjacent.passed&&l.source_error.passed,"v2 chain bypassed audits");
    for(auto backend:{NeuralConfirmation::Gpu,NeuralConfirmation::Compare}){
        options.confirmation=backend;NeuralModel confirmed(path.c_str(),options);NeuralStats health;
        auto r=generate_neural(m.view(),config,confirmed,&health);require(r.status==Status::Complete&&health.confirmation_disagreements==0,"GPU confirmation disagreed with CPU");
        require(health.reference_audit_ns==0||backend==NeuralConfirmation::Compare,"GPU confirmation executed CPU audit");
        for(size_t i=0;i<r.lods.size();++i)require(same_mesh_data(r.lods[i].view(m.view()),action_result.lods[i].view(m.view())),"confirmation backend changed audited topology");
    }
    options.confirmation=NeuralConfirmation::Cpu;
    for(auto control:{NeuralRanking::Constant,NeuralRanking::Shuffled,NeuralRanking::ShortestEdge,NeuralRanking::CurrentPlane}){
        options.ranking=control;options.ranking_seed=83;NeuralModel controlled(path.c_str(),options);NeuralStats a,b;
        auto first=generate_neural(m.view(),config,controlled,&a),second=generate_neural(m.view(),config,controlled,&b);
        require(a.action_trials<=a.decoded*options.action_trials&&a.action_trials==b.action_trials,"ranking control changed or exceeded deterministic work budget");
        require(same_mesh_data(first.lods.back().view(m.view()),second.lods.back().view(m.view())),"ranking control is nondeterministic");
        for(auto& l:first.lods)require(l.adjacent.passed&&l.source_error.passed,"ranking control bypassed audits");}
    std::filesystem::remove(path);
    auto single=overlap_cuda(m.view(),b,24,{1,0,0xB1172026},options);auto twice=m;twice.indices.insert(twice.indices.end(),m.indices.begin(),m.indices.end());auto doubled=overlap_cuda(twice.view(),b,24,{1,0,0xB1172026},options);require(std::abs(doubled-2*single)<1e-8,"overdraw counts do not scale");
    for(double screen:{16.,25.}){auto costs=render_cost(twice.view(),b,ViewSet{1,0,0xB1172026},screen);auto cuda=overlap_cuda(twice.view(),b,screen,{1,0,0xB1172026},options);require(costs.covered_pixels&&std::abs(cuda-double(costs.covered_samples)/costs.covered_pixels)<1e-8,"overdraw differs from CPU proxy");}
    EvalSettings stopped;stopped.cancelled=[]{return true;};auto cancelled=evaluate_cuda(m.view(),altered.view(),b,stopped,options);require(!cancelled.complete&&!cancelled.passed,"CUDA cancellation ignored");
    EvalSettings huge;huge.screen_size=1024;huge.supersample=8;huge.max_supersample=8;huge.views={1,0,42};NeuralStats diagnostic;
    auto limited=evaluate_cuda(m.view(),altered.view(),b,huge,options,&diagnostic);require(limited.resource_limited&&!limited.passed&&!limited.complete,"CUDA resource failure became acceptance");
    auto& failure=diagnostic.first_resource_failure;require(diagnostic.resource_failures==1&&failure.kind==NeuralResourceLimit::SampleCount&&failure.requested>failure.limit&&failure.supersample==8&&failure.view==0,"CUDA resource failure lost its cause or location");
    neural::AuditCuda recovering(options,m.view());NeuralStats recovered_stats;
    auto denied=recovering.evaluate(m.view(),altered.view(),b,huge,&recovered_stats);
    auto recovery_settings=empty_settings;recovery_settings.profile=Profile::Coverage;recovery_settings.limit=4;
    auto recovery_cpu=evaluate(m.view(),altered.view(),b,recovery_settings);require(recovery_cpu.complete&&recovery_cpu.passed,"recovery fixture must pass its visual gate");
    auto recovered=recovering.evaluate(m.view(),altered.view(),b,recovery_settings,&recovered_stats);
    require(denied.resource_limited&&recovered.complete&&recovered.passed==recovery_cpu.passed&&!recovered.resource_limited,"pooled workspace did not recover after a resource failure");
}
#endif
int main(int argc,char**) {try {
    if(argc==1){graph_contracts();if(!neural_available()){bool rejected=false;try{NeuralModel unavailable("missing.blzn");}catch(const NeuralUnavailable&){rejected=true;}require(rejected,"missing CUDA silently accepted");}}
#ifdef BLITZ_CUDA
    else {if(!neural_available())return 77;cuda_contracts();}
#endif
    std::cout<<"neural contracts passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
