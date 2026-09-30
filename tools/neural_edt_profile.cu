#include "neural_cuda.cuh"
#include <nppi_filtering_functions.h>
#include <nppcore.h>
#include <nlohmann/json.hpp>
#include <iostream>
using namespace blitz;using namespace blitz::neural;using namespace blitz::neural::gpu;
int main(){try{if(!neural_available())return 77;auto npp=[](NppStatus s){if(s!=NPP_SUCCESS)throw std::runtime_error("NPP error "+std::to_string(s));};nlohmann::json rows=nlohmann::json::array();NeuralOptions options;options.memory_mib=512;
    for(uint32_t size:{64u,65u,127u,160u,544u,1056u,2897u})for(unsigned pattern=0;pattern<3;++pattern){std::vector<uint8_t> sites(size_t(size)*size,255);std::vector<std::pair<uint32_t,uint32_t>> points;
        if(pattern==0)points={{size/3,size/2}};else if(pattern==1)points={{0,0},{size-1,0},{0,size-1},{size-1,size-1}};else for(unsigned i=0;i<43;++i)points.push_back({(i*71+13)%size,(i*113+31)%size});for(auto [x,y]:points)sites[size_t(y)*size+x]=0;
        auto legacy=benchmark_distance_field(sites,size,options,16);Device device(options,true);Buffer<uint8_t> mask(device,sites.size());mask.upload(sites);Buffer<short2> voronoi(device,sites.size());size_t scratch_bytes=0;NppiSize roi{int(size),int(size)};npp(nppiDistanceTransformPBAGetBufferSize(roi,&scratch_bytes));Buffer<uint8_t> scratch(device,scratch_bytes);NppStreamContext context{};npp(nppGetStreamContext(&context));
        auto perform=[&]{npp(nppiDistanceTransformPBA_8u32f_C1R_Ctx(mask.p,size,0,0,reinterpret_cast<Npp16s*>(voronoi.p),size*sizeof(short2),nullptr,0,nullptr,0,nullptr,0,roi,scratch.p,context));};perform();check(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();for(unsigned i=0;i<16;++i)perform();check(cudaDeviceSynchronize());double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/16;auto result=voronoi.download();uint64_t legacy_bad=0,relative_bad=0,absolute_bad=0;
        for(uint32_t y=0;y<size;++y)for(uint32_t x=0;x<size;++x){uint32_t expected=UINT32_MAX;for(auto [px,py]:points){int dx=int(px)-int(x),dy=int(py)-int(y);expected=std::min(expected,uint32_t(dx*dx+dy*dy));}auto v=result[size_t(y)*size+x];int dx=v.x-int(x),dy=v.y-int(y);relative_bad+=uint32_t(int(v.x)*v.x+int(v.y)*v.y)!=expected;absolute_bad+=uint32_t(dx*dx+dy*dy)!=expected;legacy_bad+=legacy.squared[size_t(y)*size+x]!=float(expected);}
        rows.push_back({{"size",size},{"pattern",pattern},{"legacy_seconds",legacy.seconds},{"npp_seconds",elapsed},{"legacy_bytes",legacy.bytes},{"npp_bytes",device.peak},{"legacy_mismatches",legacy_bad},{"relative_mismatches",relative_bad},{"absolute_mismatches",absolute_bad}});
    }std::cout<<rows.dump(2)<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
