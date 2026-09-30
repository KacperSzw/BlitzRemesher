#include "neural_internal.hpp"
#include "neural_action_gpu.hpp"
#include "neural_vulkan.hpp"
#include "neural_vulkan_cuda.hpp"
#include "neural_cuda.cuh"
#include "neural_raster_pixel.cuh"
#include <iostream>
using namespace blitz;using namespace blitz::neural;
static void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
static Mesh fixture(){Mesh m;m.positions={{-1,-1,0},{1,-1,0},{1,1,0},{-1,1,0}};m.normals.assign(4,{0,0,1});m.uv={{-8,-8},{8,-8},{8,8},{-8,8}};m.tangents.assign(4,{1,0,0,-1});m.colors.assign(4,{31,127,255,47});m.indices={0,1,2,0,2,3};m.materials={65535,65535};return m;}
static void candidate_contracts(){
    Mesh mesh;for(unsigned y=0;y<4;++y)for(unsigned x=0;x<4;++x){mesh.positions.push_back({float(x),float(y),0});mesh.normals.push_back({0,0,1});}
    for(uint32_t y=0;y<3;++y)for(uint32_t x=0;x<3;++x){auto i=y*4+x;mesh.indices.insert(mesh.indices.end(),{i,i+1,i+4,i+1,i+5,i+4});}mesh.double_sided={1};
    NeuralOptions options;options.memory_mib=256;options.raster_backend=NeuralRasterBackend::Vulkan;options.view_batch=1;AuditSession session(options);GpuActionState state(mesh.view(),options,true);AuditCuda audit(options,mesh.view());
    auto rows=state.teacher_actions({},1,917);require(!rows.empty(),"candidate fixture has no legal edits");auto alternatives=state.teacher_proposals(rows[0].action);
    std::array<GpuActionState::Proposal,4> proposals{alternatives[0],alternatives[2],alternatives[4],alternatives[4]};proposals[3].placement.position.x=INFINITY;
    EvalSettings e;e.screen_size=24;e.limit=3;e.views={3,1,817};e.supersample=2;e.max_supersample=4;e.max_changed_area=.4;auto box=bounds(mesh.view());
    for(auto profile:{Profile::Coverage,Profile::Attributes})for(auto count:{1u,2u,4u})for(double cutoff:{std::numeric_limits<double>::infinity(),0.,.02}){
        e.profile=profile;std::array<CandidateAudit,4> expected;
        for(unsigned i=0;i<count;++i){DeviceMeshView candidate;auto& q=expected[i];q.valid=state.trial(rows[0].action,proposals[i].placement,candidate);if(q.valid){q.faces=candidate.faces;q.value=audit.certify(mesh.view(),candidate,box,e,nullptr,cutoff,std::isfinite(cutoff)?&q.pruned:nullptr);}}
        auto views=state.trial_batch(rows[0].action,std::span(proposals).first(count));auto actual=audit.certify_candidates(mesh.view(),views,box,e,nullptr,cutoff);
        for(unsigned i=0;i<count;++i){auto& a=actual[i];auto& b=expected[i];require(a.valid==b.valid,"GPU indirect validity differs");if(!a.valid)continue;
            require(a.faces==b.faces&&a.pruned==b.pruned&&a.value.verdict==b.value.verdict&&a.value.views==b.value.views&&a.value.changed_area==b.value.changed_area&&a.value.error_upper==b.value.error_upper,"candidate batch differs from serial audit");}
    }
    require(state.view().faces==mesh.view().triangles(),"speculative batch committed geometry");
}
int main(){try{if(!neural_available())return 77;NeuralOptions options;options.memory_mib=512;options.raster_backend=NeuralRasterBackend::Vulkan;
    candidate_contracts();
    auto mesh=fixture();auto original=mesh;auto b=bounds(mesh.view());Camera c{{1,0,0},{0,1,0},{0,0,1},b.radius*4,1,24/b.diameter(),false};
    for(auto storage:{NeuralVertexStorage::Float32,NeuralVertexStorage::Position16,NeuralVertexStorage::Packed}){
        options.vertex_storage=storage;auto image=raster_gpu(mesh.view(),b,c,24,2,false,options);unsigned visible=0,covered=0;
        for(auto p:image.pixels){covered+=p.covered;if(p.visible){++visible;require(p.covered&&p.material==65535,"hardware visibility/material contract");require(p.normal.z>.999,"hardware front normal");require(std::abs(p.color.x-31/255.f)<1e-5,"RGBA8 linear color");}}
        require(visible>100&&covered>=visible&&!image.clipped,"hardware empty or clipped front face");
        auto back=mesh;std::swap(back.indices[1],back.indices[2]);std::swap(back.indices[4],back.indices[5]);auto culled=raster_gpu(back.view(),b,c,24,2,false,options);
        for(auto p:culled.pixels)require(!p.covered&&!p.visible,"one-sided back face not culled");
        auto two=raster_gpu(back.view(),b,c,24,2,true,options);unsigned back_count=0;for(auto p:two.pixels)if(p.visible){++back_count;require(p.normal.z<-.999,"two-sided normal flip");}require(back_count==visible,"two-sided visibility differs");
        auto flat=mesh;flat.normals.clear();auto flat_image=raster_gpu(flat.view(),b,c,24,2,false,options);for(auto p:flat_image.pixels)if(p.visible)require(p.normal.z>.999,"missing-normal flat fallback");
        flat.normals.assign(4,{});auto zero_image=raster_gpu(flat.view(),b,c,24,2,false,options);for(auto p:zero_image.pixels)if(p.visible)require(p.normal.z>.999,"zero-normal flat fallback");
        auto outside=mesh;outside.positions[0].x-=100;require(raster_gpu(outside.view(),b,c,24,2,true,options).clipped,"out-of-frame hardware draw escaped audit");
    }
    options.vertex_storage=NeuralVertexStorage::Automatic;require(options.draw_storage()==NeuralVertexStorage::Packed,"hardware default must pack draw streams");auto bad=mesh;bad.uv[0].x=std::nextafter(8.f,INFINITY);bool rejected=false;try{(void)raster_gpu(bad.view(),b,c,24,2,true,options);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"default packed draw silently clamped out-of-range UV");
    EvalSettings e;e.screen_size=24;e.supersample=2;e.max_supersample=4;e.limit=3;e.views={2,1,51};e.profile=Profile::Attributes;
    auto identical=evaluate_gpu(mesh.view(),mesh.view(),b,e,options);require(identical.complete&&identical.passed&&identical.views_evaluated==3,"packed identity bypassed raster audit");
    // Same hardware images: threshold certificates must agree with the exact
    // metric, including limits immediately around the measured boundary.
    NeuralStats stats;AuditCuda audit(options,mesh.view());
    for(auto profile:{Profile::Coverage,Profile::Normals,Profile::Attributes})for(float displacement:{0.f,.03f,.4f}){
        auto changed=mesh;changed.positions[0].x+=displacement;changed.normals[0]=normalized({displacement,0,1});GpuActionState state(changed.view(),options,true);e.profile=profile;
        for(double limit:{.8,1.5,3.}){e.limit=limit;auto exact=audit.evaluate(mesh.view(),state.view(),b,e);auto predicate=audit.certify(mesh.view(),state.view(),b,e,&stats);
            require(predicate.verdict!=(exact.passed?AuditVerdict::Fail:AuditVerdict::Pass),"sparse certificate contradicts exact hardware metric");require(predicate.verdict!=AuditVerdict::Unknown,"bounded fixture returned unknown");}
    }
    require(stats.gpu_sparse_passes>0&&stats.gpu_sparse_queries>0,"sparse tests did not exercise witnesses");
    // Match exact metrics and certificate verdicts across independent switches.
    // Limits below the sparse certificate floor exercise exact fallback/refinement.
    for(auto profile:{Profile::Coverage,Profile::Normals,Profile::Attributes})for(float shift:{0.f,.13f,.6f})for(double limit:{.4,1.5,3.}){
        auto changed=mesh;changed.positions[0].x+=shift;changed.normals[0]=normalized({shift,0,1});e.profile=profile;e.limit=limit;
        auto serial=options;serial.view_batch=1;serial.direct_targets=false;serial.mask_only_coverage=false;AuditCuda control(serial,mesh.view());GpuActionState state(changed.view(),serial,true);
        auto expected=control.evaluate(mesh.view(),state.view(),b,e);auto certificate=control.certify(mesh.view(),state.view(),b,e);
        for(uint8_t batch:{2,4})for(bool direct:{false,true}){auto settings=options;settings.view_batch=batch;settings.direct_targets=direct;AuditCuda tested(settings,mesh.view());
            auto actual=tested.evaluate(mesh.view(),state.view(),b,e);auto predicate=tested.certify(mesh.view(),state.view(),b,e);
            require(actual.passed==expected.passed&&actual.complete==expected.complete&&actual.error==expected.error&&actual.coverage==expected.coverage&&actual.changed_area==expected.changed_area&&actual.normal_degrees==expected.normal_degrees&&actual.views_evaluated==expected.views_evaluated&&actual.supersample==expected.supersample,"batched exact audit differs from serial");
            require(predicate.verdict==certificate.verdict&&predicate.changed_area==certificate.changed_area&&predicate.views==certificate.views&&predicate.supersample==certificate.supersample,"direct/batched certificate differs from serial");}
        bool pruned=false;auto expected_pruning=control.certify(mesh.view(),state.view(),b,e,nullptr,.01,&pruned);AuditCuda batched(options,mesh.view());bool batch_pruned=false;auto actual_pruning=batched.certify(mesh.view(),state.view(),b,e,nullptr,.01,&batch_pruned);require(pruned==batch_pruned&&expected_pruning.verdict==actual_pruning.verdict&&expected_pruning.views==actual_pruning.views,"batched incumbent pruning differs from serial");
    }
    {
        auto precise=mesh;precise.positions[0].x=-.81234567f;precise.positions[2].y=.91234567f;precise.exact_position_bits={15};
        auto controls=options;controls.exact_position_bps=10000;GpuActionState exact_state(precise.view(),controls,true);gpu::Device device(controls);VulkanRaster renderer(controls);gpu::Buffer<AuditPixel> pixels(device,64*64);
        auto view=exact_state.view();renderer.render(view,b,c,24,2,false,NeuralVertexStorage::Packed,pixels.p,nullptr);auto packed=pixels.download();renderer.render(view,b,c,24,2,false,NeuralVertexStorage::Float32,pixels.p,nullptr);auto full=pixels.download();
        for(size_t i=0;i<full.size();++i)require(packed[i].covered==full[i].covered&&packed[i].visible==full[i].visible&&packed[i].normal.x==full[i].normal.x&&packed[i].normal.y==full[i].normal.y&&packed[i].normal.z==full[i].normal.z,"indexed sparse precision lookup changed exact positions");
        auto layout=draw_layout(view,NeuralVertexStorage::Packed);gpu::Buffer<char> draw(device,layout.bytes);pack_draw(view,NeuralVertexStorage::Packed,layout,draw.p);auto bytes=draw.download();for(size_t i=0;i<precise.positions.size();++i)require(std::memcmp(bytes.data()+layout.exact_positions+i*sizeof(Vec3),&precise.positions[i],sizeof(Vec3))==0,"sparse exact positions were rounded or reordered");
        view.exact_position_bps=500;bool rejected=false;try{renderer.render(view,b,c,24,2,false,NeuralVertexStorage::Packed,pixels.p,nullptr);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"renderer did not enforce precision cap");
    }
    e.cancelled=[] {return true;};GpuActionState state(mesh.view(),options,true);require(audit.certify(mesh.view(),state.view(),b,e).verdict==AuditVerdict::Unknown,"cancelled query produced label");
    require(same_mesh_data(mesh.view(),original.view()),"packing mutated source");
    {
        MemoryScope memory(options);gpu::Device device(options);GpuActionState state(mesh.view(),options,true);auto view=state.view();auto layout=draw_layout(view,NeuralVertexStorage::Packed);gpu::Buffer<char> draw(device,layout.bytes);pack_draw(view,NeuralVertexStorage::Packed,layout,draw.p);auto bytes=draw.download();
        auto read=[&]<class T>(size_t at){T v;std::memcpy(&v,bytes.data()+at,sizeof(v));return v;};
        require(layout.position_stride==6&&layout.normal_stride==4,"packed stream stride changed");
        for(size_t i=0;i<4;++i){auto p=mesh.positions[i];require(read.template operator()<uint16_t>(layout.position+6*i)==(p.x<0?0:65535),"GPU position code differs at bound");require(read.template operator()<uint16_t>(layout.uv+4*i)==(mesh.uv[i].x<0?0:65535),"GPU UV endpoint code");require(read.template operator()<uint32_t>(layout.normal+4*i)==(511u<<20),"GPU RGB10 normal layout");require(read.template operator()<uint32_t>(layout.tangent+4*i)==(511u|(3u<<30)),"GPU tangent alpha sign layout");auto color=read.template operator()<ColorRGBA8>(layout.color+4*i);require(color.r==31&&color.a==47,"GPU RGBA8 byte layout");}
        // Reuse the same attachments across removal, depth ties, material seams
        // and bounds changes. Changed revisions must repack and redraw all faces.
        Mesh layers;layers.positions={{-1,-1,0},{1,-1,0},{0,1,0},{-1,-1,0},{1,-1,0},{0,1,0}};layers.normals.assign(6,{0,0,1});layers.indices={0,1,2,3,4,5};layers.materials={7,13};
        auto positions=gpu::upload_stream(device,layers.view().positions);auto normals=gpu::upload_stream(device,layers.view().normals);gpu::Buffer<uint32_t> indices(device,layers.indices.size());indices.upload(layers.indices);gpu::Buffer<uint16_t> materials(device,layers.materials.size());materials.upload(layers.materials);
        DeviceMeshView dm{};dm.positions=positions.p;dm.normals=normals.p;dm.indices=indices.p;dm.materials=materials.p;dm.vertices=6;dm.faces=2;dm.identity=0xfeed;dm.revision=1;
        VulkanRaster raster(options);gpu::Buffer<AuditPixel> pixels(device,64*64);
        auto center=[&]{raster.render(dm,b,c,24,2,false,NeuralVertexStorage::Packed,pixels.p,nullptr);return pixels.download()[32*64+32];};
        require(center().material==7,"hardware depth tie does not keep first face");
        auto views=cameras(b,24,{2,2,171});std::array<gpu::Buffer<AuditPixel>,4> batched;std::array<gpu::Buffer<RasterDebugPixel>,4> debug;std::array<RasterOutput,4> outputs;
        for(size_t i=0;i<4;++i){batched[i]=gpu::Buffer<AuditPixel>(device,64*64);debug[i]=gpu::Buffer<RasterDebugPixel>(device,64*64);outputs[i]={batched[i].p,nullptr,debug[i].p};}
        raster.render_batch(dm,b,views,24,2,true,NeuralVertexStorage::Packed,outputs);
        std::array<std::vector<AuditPixel>,4> expected;std::array<std::vector<RasterDebugPixel>,4> witnesses;
        for(size_t i=0;i<4;++i){expected[i]=batched[i].download();witnesses[i]=debug[i].download();}
        for(size_t i=0;i<4;++i){bool clipped=raster.render(dm,b,views[i],24,2,true,NeuralVertexStorage::Packed,pixels.p,nullptr,debug[0].p);auto actual=pixels.download();auto actual_debug=debug[0].download();require(clipped==outputs[i].clipped,"batch clip flag differs");
            for(size_t j=0;j<actual.size();++j){auto x=actual[j],y=expected[i][j];require(x.covered==y.covered&&x.visible==y.visible&&x.material==y.material&&x.normal.x==y.normal.x&&x.normal.y==y.normal.y&&x.normal.z==y.normal.z,"batched draw changes pixels");require(actual_debug[j].face==witnesses[i][j].face&&actual_debug[j].depth==witnesses[i][j].depth,"debug attachment changes depth/ownership");if(x.visible)require(actual_debug[j].face<2&&actual_debug[j].depth>=0&&actual_debug[j].depth<=1,"debug face/depth range");}}
        gpu::Buffer<uint32_t> mask_bits(device,64*64/32);raster.trim_targets(1);
        auto full_bytes=raster.bytes();raster.render(dm,b,c,24,2,false,NeuralVertexStorage::Packed,pixels.p,nullptr,nullptr,true,mask_bits.p);
        auto mask_image=pixels.download();auto words=mask_bits.download();
        auto mask_bytes=raster.bytes();require(mask_bytes<full_bytes,"mask-only retained shading/depth targets");
        for(size_t i=0;i<mask_image.size();++i)require(mask_image[i].covered==((words[i/32]>>(i%32))&1)&&!mask_image[i].visible,"bit mask or absent visibility contract");
        auto mask_surface=raster.surfaces();require(mask_surface.mask&&!mask_surface.attributes&&!mask_surface.colors,"mask-only exported shading surfaces");
        center();auto full_image=pixels.download();for(size_t i=0;i<mask_image.size();++i)require(mask_image[i].covered==full_image[i].covered,"mask-only changed coverage");
        raster.trim_targets(1);require(center().material==7,"target reclamation invalidated surviving slot");
        // Domain changes are part of the geometry cache key even without a revision.
        dm.fixed_quantization=true;dm.quant_low={-2,-2,0};dm.quant_extent={4,4,0};require(center().material==7,"explicit quantization domain failed");dm.quant_extent.z=-1;bool invalid_domain=false;try{center();}catch(const std::invalid_argument&){invalid_domain=true;}require(invalid_domain,"invalid quantization domain reused cached geometry");dm.fixed_quantization=false;
        std::vector<uint32_t> removed={3,4,5,0,1,2};indices.upload(removed);std::vector<uint16_t> new_materials={13,7};materials.upload(new_materials);dm.faces=1;++dm.revision;require(center().material==13,"removed foreground left stale depth or material");
        for(unsigned i=3;i<6;++i)layers.positions[i].x+=10;positions.upload(layers.positions);++dm.revision;require(!center().visible,"changed bounds reused stale packed geometry");
    }
    std::cout<<"Vulkan draw/storage contracts passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
