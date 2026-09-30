#include "neural_internal.hpp"
#include <iostream>
using namespace blitz;using namespace blitz::neural;
static void check(bool good,const char* message){if(!good)throw std::runtime_error(message);}
static Pixel pixel(Vec3 normal={0,0,1}){Pixel p;p.covered=p.visible=1;p.normal=normal;p.color={1,1,1,1};return p;}
int main(){try{if(!neural_available())return 77;NeuralOptions options;options.memory_mib=256;
    Raster a{64,64,std::vector<Pixel>(4096),false},b=a;EvalSettings e;e.supersample=e.max_supersample=8;e.screen_size=16;e.limit=2;e.profile=Profile::Attributes;e.max_changed_area=1;
    a.pixels[20*64+20]=pixel();a.pixels[20*64+21]=pixel();b.pixels[20*64+20]=pixel();
    check(certify_rasters_cuda(a,b,e,options).verdict==AuditVerdict::Pass,"coverage witnesses excluded the mask intersection");
    auto compare=[&](const Raster& x,const Raster& y){
        double coverage=coverage_distance(x,y,e.supersample,true),upper=coverage+2*std::sqrt(2.)/e.supersample+1e-6;
        double error=upper<=e.limit?std::max(upper,attributed_distance(x,y,e,e.limit)):upper;if(x.clipped||y.clipped)error=INFINITY;
        for(auto backend:{NeuralRasterBackend::Cuda,NeuralRasterBackend::Vulkan}){options.raster_backend=backend;auto exact=measure_rasters_cuda(x,y,e,options);check((std::isinf(exact.error)&&std::isinf(error))||std::abs(exact.error-error)<2e-12,"GPU metric differs from independent raster oracle");
            auto p=certify_rasters_cuda(x,y,e,options);check(p.verdict!=AuditVerdict::Pass||exact.passed,"incorrect passing certificate");check(p.verdict!=AuditVerdict::Fail||!exact.passed,"incorrect failing certificate");}
    };
    for(auto profile:{Profile::Coverage,Profile::Normals,Profile::Attributes})for(uint8_t ss:{uint8_t(2),uint8_t(3),uint8_t(8)}){e.profile=profile;e.supersample=ss;e.limit=3;
        for(unsigned trial=0;trial<4;++trial){a.pixels.assign(4096,{});b.pixels.assign(4096,{});
            for(unsigned i=0;i<23;++i){auto at=(i*173+trial*91)%4096;a.pixels[at]=pixel(i%5?Vec3{0,0,1}:Vec3{});auto next=(at+trial)%4096;b.pixels[next]=pixel(i%7?normalized({.07f*trial,0,1}):Vec3{});b.pixels[next].material=uint16_t(i%2);b.pixels[next].color={1,.9f,1,1};}compare(a,b);}
    }
    e.profile=Profile::Normals;e.supersample=8;e.limit=.8;a.pixels.assign(4096,pixel());b.pixels.assign(4096,pixel(normalized({.1f,0,1})));
    auto exact=measure_rasters_cuda(a,b,e,options);check(exact.passed,"controlled normal fixture should pass");check(certify_rasters_cuda(a,b,e,options,1).verdict==AuditVerdict::Unknown,"queue overflow produced a certificate");
    for(double limit:{std::nextafter(exact.error,0.),exact.error,std::nextafter(exact.error,INFINITY)}){e.limit=limit;compare(a,b);}
    a.pixels.assign(4096,{});b.pixels.assign(4096,{});e.limit=2;compare(a,b);a.pixels[7]=pixel();compare(a,b);a.clipped=true;check(certify_rasters_cuda(a,b,e,options).verdict==AuditVerdict::Unknown,"clipping bypassed fallback");
    e.cancelled=[] {return true;};check(certify_rasters_cuda(a,b,e,options).verdict==AuditVerdict::Unknown,"cancelled predicate escaped unknown");
    std::cout<<"sparse/dense raster metric contracts passed\n";return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
