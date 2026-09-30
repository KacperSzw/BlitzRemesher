#include "neural_action_data.hpp"
#include "neural_json.hpp"
#include <cuda_runtime.h>
#include <iostream>
using namespace blitz;using namespace blitz::neural;using namespace blitz::neural::training;
int main(int argc,char** argv){try{
    if(argc<3||argc>4)throw std::invalid_argument("blitz-neural-hardware-profile ASSET FRESH_OUTPUT_JSON [PIXELS]");if(!neural_available())return 77;
    fs::path path=argv[2];if(fs::exists(path))throw std::invalid_argument("choose a fresh report");auto [mesh,metadata]=training_mesh(argv[1]);auto b=bounds(mesh.view());double screen=argc>3?std::stod(argv[3]):128;
    NeuralOptions o;o.memory_mib=2048;json report={{"complete",false},{"score",nullptr},{"asset",metadata},{"pixels",screen},{"sampling",4},{"repetitions",12},{"timing_scope","reused workspace, changed revision each draw; packing + passes + interop + CUDA target conversion; host readback excluded"},{"rows",json::array()}};
    cudaDeviceProp gpu{};cudaGetDeviceProperties(&gpu,0);int driver=0;cudaDriverGetVersion(&driver);report["gpu"]={{"name",gpu.name},{"compute_capability",std::to_string(gpu.major)+"."+std::to_string(gpu.minor)},{"cuda_driver",driver}};report["binary_sha256"]=file_sha256("/proc/self/exe");report["protocol_sha256"]=file_sha256("research/PROTOCOL.md");
    auto views=cameras(b,screen,{2,2,719});
    for(uint32_t v=0;v<views.size();++v){Raster original,hardware,packed;
        for(unsigned format=0;format<5;++format){o.raster_backend=format?NeuralRasterBackend::Vulkan:NeuralRasterBackend::Cuda;o.vertex_storage=format<=1?NeuralVertexStorage::Float32:format==2?NeuralVertexStorage::Position16:NeuralVertexStorage::Packed;
            auto r=raster_benchmark(mesh.view(),b,views[v],screen,4,false,o,12,format==4);if(format==3)packed=r.raster;if(format==0)original=r.raster;if(format==1)hardware=r.raster;auto& reference=format==4?packed:format<2?original:hardware;
            uint64_t coverage=0,visibility=0,material=0;double normal=0,rgb=0;json worst;
            for(size_t i=0;i<reference.pixels.size();++i){auto a=reference.pixels[i],q=r.raster.pixels[i];coverage+=a.covered!=q.covered;visibility+=a.visible!=q.visible;if(a.visible&&q.visible){material+=a.material!=q.material;double angle=std::acos(std::clamp(dot(a.normal,q.normal)/std::max(1e-30,length(a.normal)*length(q.normal)),-1.,1.))*180/3.141592653589793;if(angle>normal){normal=angle;worst={{"pixel",i},{"reference",{a.normal.x,a.normal.y,a.normal.z}},{"actual",{q.normal.x,q.normal.y,q.normal.z}}};}rgb=std::max(rgb,length(Vec3{a.color.x-q.color.x,a.color.y-q.color.y,a.color.z-q.color.z}));}}
            report["rows"].push_back({{"view",v},{"coverage_only",format==4},{"raster",raster_name(o.raster_backend)},{"storage",storage_name(o.vertex_storage)},{"seconds",r.seconds},{"setup_seconds",r.setup_seconds},{"packing_gpu_seconds",r.packing_seconds},{"raster_gpu_seconds",r.render_seconds},{"unpack_gpu_seconds",r.unpack_seconds},{"draw_vertex_bytes",r.draw_bytes},{"gpu_bytes",r.gpu_bytes},{"coverage_changes",coverage},{"visibility_changes",visibility},{"material_changes",material},{"maximum_normal_delta_degrees",normal},{"maximum_rgb_delta",rgb},{"clipped",r.raster.clipped},{"comparison",format<2?"cuda FP32":"Vulkan FP32"}});write_json(path,report);
            report["rows"].back()["worst_normal"]=worst;if(format==4)report["rows"].back()["comparison"]="Vulkan packed full attachments; coverage only";
        }
    }report["complete"]=true;write_json(path,report);std::cout<<report.dump(2)<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
