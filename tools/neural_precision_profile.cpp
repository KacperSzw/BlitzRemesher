#include "neural_action_data.hpp"
#include "neural_json.hpp"
#include <iostream>
using namespace blitz;using namespace blitz::neural;using namespace blitz::neural::training;
struct Format {const char* name;uint8_t attributes,depth,positions;};
constexpr Format formats[]={{"exact",32,32,32},{"attributes16",16,32,32},{"attributes8",8,32,32},{"depth16",32,16,32},{"positions16",32,32,16},{"all16",16,16,16}};
struct Metrics {double error{},area{},coverage{};};
Metrics metric(const Raster& a,const Raster& b,const EvalSettings& e){uint64_t changed=0,total=0;for(size_t i=0;i<a.pixels.size();++i){changed+=a.pixels[i].covered!=b.pixels[i].covered;total+=a.pixels[i].covered||b.pixels[i].covered;}
    double coverage=coverage_distance(a,b,e.supersample,true);return {a.clipped||b.clipped?INFINITY:std::max(coverage+2*std::sqrt(2.)/e.supersample+1e-6,attributed_distance(a,b,e,e.limit)),total?double(changed)/total:0,coverage};}
Mesh overlaps(){Mesh m;for(unsigned layer=0;layer<3;++layer){float z=layer==0?0:layer==1?1e-6f:1.f;uint32_t first=uint32_t(m.positions.size());
    for(auto p:{Vec3{0,0,z},Vec3{1,0,z},Vec3{1,1,z},Vec3{0,1,z}}){m.positions.push_back(p);m.normals.push_back(layer==1?normalized({.4f,0,1}):Vec3{0,0,1});m.colors.push_back(layer==1?ColorRGBA8{250,40,20,255}:ColorRGBA8{20,40,250,255});m.uv.push_back({p.x,p.y});}
    m.indices.insert(m.indices.end(),{first,first+1,first+2,first,first+2,first+3});m.materials.insert(m.materials.end(),2,uint16_t(layer));}m.double_sided={1,1,1};return m;}
int main(int argc,char** argv){try{if(argc<3)throw std::invalid_argument("blitz-neural-precision-profile ASSET|overlaps OUTPUT [PIXELS]");if(!neural_available())return 77;
    fs::path output=argv[2];if(fs::exists(output))throw std::invalid_argument("choose a fresh precision report");std::string asset=argv[1];Mesh mesh;json provenance;
    if(asset=="overlaps"){mesh=overlaps();provenance={{"fixture","three layers, a one-millionth separation, different materials and normals"}};}
    else {auto loaded=training_mesh(asset);mesh=std::move(loaded.first);provenance=std::move(loaded.second);}
    auto candidate=mesh;size_t vertex=mesh.positions.size()/2;candidate.positions[vertex].z+=float(bounds(mesh.view()).diameter()*.01);
    if(!candidate.normals.empty())candidate.normals[vertex]=normalized(candidate.normals[vertex]+Vec3{.17f,.11f,0});
    auto b=bounds(mesh.view());EvalSettings e;e.screen_size=argc>3?std::stod(argv[3]):128;e.profile=Profile::Attributes;e.supersample=e.max_supersample=4;e.limit=8;e.views={2,2,719};
    NeuralOptions options;options.memory_mib=256;json result={{"asset",asset},{"source",provenance},{"triangles",mesh.view().triangles()},{"vertices",mesh.positions.size()},{"pixels",e.screen_size},{"supersample",e.supersample},{"view_seed",e.views.rotation_seed},{"score",nullptr},{"research_only",true},{"rows",json::array()},{"position_storage",{{"fp32_bytes",mesh.positions.size()*12},{"unorm16_bytes",mesh.positions.size()*6+24},{"range","mesh AABB; three UNORM16 coordinates; FP32 low and scale per axis"}}}};
    auto views=cameras(b,e.screen_size,e.views);
    if(asset=="overlaps")views.push_back(Camera{{1,0,0},{0,1,0},{0,0,1},b.diameter()*4,1,e.screen_size/b.diameter(),false});
    result["complete"]=false;result["warmup_rasters_per_measurement"]=1;result["timing_scope"]="render and synchronize; excludes CPU packing, upload and readback; shared GPU";write_json(output,result);
    try{for(size_t view=0;view<views.size();++view){auto camera=views[view];auto original=raster_precision_cuda(mesh.view(),b,camera,e.screen_size,e.supersample,false,32,32,32,options);
        auto altered=raster_precision_cuda(candidate.view(),b,camera,e.screen_size,e.supersample,false,32,32,32,options);auto expected=metric(original.raster,altered.raster,e);
        for(auto format:formats){auto a=format.attributes==32&&format.depth==32&&format.positions==32?original:raster_precision_cuda(mesh.view(),b,camera,e.screen_size,e.supersample,false,format.attributes,format.depth,format.positions,options);
            auto c=format.attributes==32&&format.depth==32&&format.positions==32?altered:raster_precision_cuda(candidate.view(),b,camera,e.screen_size,e.supersample,false,format.attributes,format.depth,format.positions,options);
            uint64_t coverage_changed=0,visibility_changed=0,material_changed=0;double normal_delta=0,color_delta=0;
            for(size_t i=0;i<a.raster.pixels.size();++i){auto p=original.raster.pixels[i],q=a.raster.pixels[i];coverage_changed+=p.covered!=q.covered;visibility_changed+=p.visible!=q.visible;
                if(p.visible&&q.visible){material_changed+=p.material!=q.material;double cosine=dot(p.normal,q.normal)/std::max(1e-30,length(p.normal)*length(q.normal));normal_delta=std::max(normal_delta,std::acos(std::clamp(cosine,-1.,1.))*180/3.14159265358979323846);
                    color_delta=std::max(color_delta,length(Vec3{p.color.x-q.color.x,p.color.y-q.color.y,p.color.z-q.color.z}));}}
            auto actual=metric(a.raster,c.raster,e),drift=metric(original.raster,a.raster,e);unsigned changed_decisions=0,checks=0;std::vector<double> limits{.25,.5,1,2,3,4};
            if(std::isfinite(expected.error)){limits.push_back(std::nextafter(expected.error,0.));limits.push_back(expected.error);limits.push_back(std::nextafter(expected.error,INFINITY));}
            json threshold_checks=json::array();for(double limit:limits){++checks;changed_decisions+=(expected.error<=limit)!=(actual.error<=limit);threshold_checks.push_back({{"limit",limit},{"reference_passed",expected.error<=limit},{"quantized_passed",actual.error<=limit}});}
            result["rows"].push_back({{"view",view},{"format",format.name},{"bytes_per_pixel",a.bytes_per_pixel},{"raster_seconds",a.seconds+c.seconds},{"reference_error",expected.error},{"quantized_error",actual.error},{"reference_area",expected.area},{"quantized_area",actual.area},{"decision_checks",checks},{"decision_changes",changed_decisions},{"source_coverage_changes",coverage_changed},{"source_visibility_changes",visibility_changed},{"source_material_changes",material_changed},{"max_source_normal_delta_degrees",normal_delta},{"max_source_rgb_delta",color_delta}});
            result["rows"].back()["thresholds"]=threshold_checks;
            result["rows"].back()["source_quantization_error_px"]=drift.error;
            result["rows"].back()["source_quantization_area"]=drift.area;
            result["rows"].back()["source_quantization_passes_3px"]=drift.error<=3&&drift.area<=e.max_changed_area;
            write_json(output,result);
        }
    }}catch(const std::exception& error){result["error"]=error.what();write_json(output,result);throw;}
    result["complete"]=true;write_json(output,result);std::cout<<output<<'\n';return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
