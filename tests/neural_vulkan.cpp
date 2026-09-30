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
int main(){try{if(!neural_available())return 77;NeuralOptions options;options.memory_mib=512;options.raster_backend=NeuralRasterBackend::Vulkan;
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
    options.vertex_storage=NeuralVertexStorage::Packed;auto bad=mesh;bad.uv[0].x=std::nextafter(8.f,INFINITY);bool rejected=false;try{(void)raster_gpu(bad.view(),b,c,24,2,true,options);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"out-of-range UV silently clamped");
    EvalSettings e;e.screen_size=24;e.supersample=2;e.max_supersample=4;e.limit=3;e.views={2,1,51};e.profile=Profile::Attributes;
    auto identical=evaluate_gpu(mesh.view(),mesh.view(),b,e,options);require(identical.complete&&identical.passed&&identical.views_evaluated==3,"packed identity bypassed raster audit");
    // Same hardware images: threshold certificates must agree with the exact
    // metric, including limits immediately around the measured boundary.
    NeuralStats stats;AuditCuda audit(options,mesh.view());
    for(auto profile:{Profile::Coverage,Profile::Normals,Profile::Attributes})for(float displacement:{0.f,.03f,.4f}){
        auto changed=mesh;changed.positions[0].z+=displacement;changed.normals[0]=normalized({displacement,0,1});GpuActionState state(changed.view(),options,true);e.profile=profile;
        for(double limit:{.8,1.5,3.}){e.limit=limit;auto exact=audit.evaluate(mesh.view(),state.view(),b,e);auto predicate=audit.certify(mesh.view(),state.view(),b,e,&stats);
            require(predicate.verdict!=(exact.passed?AuditVerdict::Fail:AuditVerdict::Pass),"sparse certificate contradicts exact hardware metric");require(predicate.verdict!=AuditVerdict::Unknown,"bounded fixture returned unknown");}
    }
    require(stats.gpu_sparse_passes>0&&stats.gpu_sparse_queries>0,"sparse tests did not exercise witnesses");
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
        std::vector<uint32_t> removed={3,4,5,0,1,2};indices.upload(removed);std::vector<uint16_t> new_materials={13,7};materials.upload(new_materials);dm.faces=1;++dm.revision;require(center().material==13,"removed foreground left stale depth or material");
        for(unsigned i=3;i<6;++i)layers.positions[i].x+=10;positions.upload(layers.positions);++dm.revision;require(!center().visible,"changed bounds reused stale packed geometry");
    }
    std::cout<<"Vulkan draw/storage contracts passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
