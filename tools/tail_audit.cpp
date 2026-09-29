#include "blitz/io.hpp"
#include "coverage.hpp"
#include <openssl/sha.h>
#include <chrono>
#include <fstream>
#include <iostream>
using namespace blitz;
using json=nlohmann::json;
namespace fs=std::filesystem;
static json read(const fs::path& p){std::ifstream f(p);if(!f)throw std::runtime_error("cannot read "+p.string());json j;f>>j;return j;}
static std::string hash(const fs::path& p){
    std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("cannot hash "+p.string());
    std::string data{std::istreambuf_iterator<char>(f),{}};unsigned char d[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(data.data()),data.size(),d);std::string out;
    for(auto c:d){out+="0123456789abcdef"[c>>4];out+="0123456789abcdef"[c&15];}return out;
}
static json measurement(Measurement m,double pixel_limit,double area_limit) {
    return {{"passed",m.passed},{"complete",m.complete},{"changed_area",m.changed_area},{"error_px",std::isfinite(m.error)?json(m.error):json(nullptr)},
        {"nonfinite_error",!std::isfinite(m.error)},{"coverage_upper_px",std::isfinite(m.coverage_upper)?json(m.coverage_upper):json(nullptr)},
        {"distance_passed",m.error<=pixel_limit},{"area_passed",m.changed_area<=area_limit},
        {"views",m.views_evaluated},{"worst_view",m.worst_view},{"changed_area_worst_view",m.changed_area_worst_view},
        {"supersample",m.supersample},{"resource_limited",m.resource_limited}};
}
// Evaluator complete=false also means it stopped at a failing witness. That
// resolves this comparison, unlike cancellation or a resource interruption.
static bool decided(const json& m,double pixel_limit,double area_limit) {
    if(m.at("resource_limited")==true)return false;
    if(m.at("passed")==true)return m.at("complete")==true;
    return m.at("views").get<uint32_t>()>0&&(m.at("nonfinite_error")==true||
        m.at("error_px").get<double>()>pixel_limit||m.at("changed_area").get<double>()>area_limit);
}
int main(int argc,char** argv){
    try{
        if(argc<4||argc>7)throw std::invalid_argument("blitz-tail-audit RUN_DIRECTORY ASSET_ID OUTPUT.json [ROTATION_SEED [CACHE_MIB [FIRST_LEVEL]]]");
        auto seed=argc>=5?std::stoull(argv[4],nullptr,0):uint64_t(0xB1172026);if(seed>UINT32_MAX)throw std::invalid_argument("invalid rotation seed");
        auto cache_mib=argc>=6?std::stoull(argv[5]):0;if(cache_mib>256)throw std::invalid_argument("cache must be 0..256 MiB");
        fs::path run=argv[1],dir=run/"meshes"/argv[2];auto row=read(run/"rows"/(std::string(argv[2])+".json")),meta=read(run/"metadata.json");
        if(!row.at("complete").get<bool>()||row.value("failed",false)||row.at("run_sha256")!=meta.at("run_sha256"))throw std::runtime_error("invalid audit input");
        auto settings=settings_json(meta.at("config"),true);auto g=read(dir/"chain.gltf");auto levels=row.at("result").at("lods");
        const size_t first=argc==7?std::stoull(argv[6]):std::max<size_t>(1,levels.size()>3?levels.size()-3:1);
        if(!first||first>=levels.size())throw std::invalid_argument("invalid first audit level");
        auto source=load_gltf_mesh(dir/"chain.gltf",g.at("nodes").at(0).at("mesh"));auto reference=bounds(source.view());
        json report={{"version",3},{"id",argv[2]},{"run",run.filename().string()},{"run_sha256",meta.at("run_sha256")},
            {"chain_gltf_sha256",hash(dir/"chain.gltf")},{"chain_bin_sha256",hash(dir/"chain.bin")},
            {"row_sha256",hash(run/"rows"/(std::string(argv[2])+".json"))},{"profile",meta.at("config").at("profile")},
            {"purpose",first==1?"Dense independent full-chain check":"Dense independent tail check"},
            {"first_level",first},{"expected_levels",levels.size()-first},{"complete",false},
            {"audit",{{"orthographic",642},{"perspective",64},{"seed",uint32_t(seed)},{"supersample",8},{"max_supersample",32},
                {"max_changed_area",settings.max_changed_area}}},
            {"lods",json::array()},{"passed",true}};
#if defined(__linux__)
        report["binary_sha256"]=hash("/proc/self/exe");
#endif
        auto begin=std::chrono::steady_clock::now();PerformanceStats work;
        fs::path output=argv[3];if(!output.parent_path().empty())fs::create_directories(output.parent_path());
        json saved=json::array();
        if(fs::exists(output)) {
            auto previous=read(output);
            for(auto key:{"run_sha256","chain_gltf_sha256","chain_bin_sha256","row_sha256","audit","first_level"})
                if(previous.at(key)!=report.at(key))throw std::runtime_error("audit resume provenance mismatch");
            // Keep the old raw checkpoint, including an incomplete last level.
            auto archive=output;archive+="."+hash(output).substr(0,16)+".checkpoint";
            if(!fs::exists(archive))fs::copy_file(output,archive);
            report["prior_seconds"]=previous.value("prior_seconds",0.)+previous.value("seconds",0.);
            for(auto l:previous.at("lods")) {
                const auto i=l.at("level").get<size_t>();
                if(i!=first+saved.size()||i>=levels.size())throw std::runtime_error("audit resume level mismatch");
                if(!l.contains("binary_sha256"))l["binary_sha256"]=previous.at("binary_sha256");
                saved.push_back(std::move(l));
            }
        }
        auto checkpoint=[&]{auto temporary=output;temporary+=".part";std::ofstream out(temporary);out<<report.dump(2)<<'\n';out.close();if(!out)throw std::runtime_error("cannot checkpoint audit");fs::rename(temporary,output);};
        checkpoint();
        for(size_t i=first;i<levels.size();++i) {
            if(i-first<saved.size()) {
                const auto& l=saved[i-first];
                if(decided(l.at("source"),l.at("source_limit"),settings.max_changed_area)&&decided(l.at("adjacent"),l.at("transition_limit"),settings.max_changed_area)) {
                    if(l["source"]["passed"]==false||l["adjacent"]["passed"]==false)report["passed"]=false;
                    report["lods"].push_back(l);checkpoint();continue;
                }
            }
            auto current=load_gltf_mesh(dir/"chain.gltf",g.at("nodes").at(i).at("mesh"));
            auto previous=load_gltf_mesh(dir/"chain.gltf",g.at("nodes").at(i-1).at("mesh"));
            EvalSettings e;e.profile=settings.profile;e.weights=settings.weights;e.screen_size=levels[i].at("screen_pixels");e.views.rotation_seed=uint32_t(seed);
            e.max_changed_area=settings.max_changed_area;
            e.performance=&work;
            e.cancelled=[&]{return std::chrono::steady_clock::now()-begin>std::chrono::minutes(50);};
            double t=settings.levels==2?0:double(i-1)/(settings.levels-2);
            e.weights.normal*=settings.normal_importance.at(t);e.weights.color*=settings.attribute_importance.at(t);e.weights.material*=settings.attribute_importance.at(t);
            detail::CoverageCache cache(settings.profile==Profile::Coverage?uint32_t(cache_mib)*1024*1024:0,reference);
            e.limit=levels[i].at("source_limit");auto src=cache.evaluate(source.view(),current.view(),e,0,true);
            e.limit=levels[i].at("transition_limit");
            // LOD1 (and a source fallback predecessor) compares the exact same
            // imported mesh twice. Equal limits prove the same measurement.
            auto adj=levels[i].at("source_limit")==levels[i].at("transition_limit")&&same_mesh_data(source.view(),previous.view())?
                src:cache.evaluate(previous.view(),current.view(),e,1,true);
            if(!src.passed||!adj.passed)report["passed"]=false;
            report["lods"].push_back({{"level",i},{"binary_sha256",report.value("binary_sha256",std::string{})},{"triangles",current.view().triangles()},{"screen_pixels",e.screen_size},
                {"source_limit",levels[i].at("source_limit")},{"transition_limit",e.limit},{"max_changed_area",e.max_changed_area},
                {"source",measurement(src,levels[i].at("source_limit"),e.max_changed_area)},
                {"adjacent",measurement(adj,e.limit,e.max_changed_area)}});
            std::cerr<<argv[2]<<" LOD"<<i<<" source="<<src.passed<<" adjacent="<<adj.passed<<'\n';
            report["seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();checkpoint();
            // A large level can exceed the raster budget while smaller levels
            // remain auditable. Keep that interruption visible and attempt the
            // remaining scheduled levels until the batch deadline.
            if(e.cancelled())break;
        }
        report["complete"]=report["lods"].size()==levels.size()-first&&std::all_of(report["lods"].begin(),report["lods"].end(),[](const json& l){
            return decided(l["source"],l["source_limit"],l["max_changed_area"])&&decided(l["adjacent"],l["transition_limit"],l["max_changed_area"]);});
        report["passed"]=report["complete"]==true&&std::all_of(report["lods"].begin(),report["lods"].end(),[](const json& l){return l["source"]["passed"]==true&&l["adjacent"]["passed"]==true;});
        report["completion_scope"]="Every planned comparison passed all views or produced a failing witness; individual complete=false can mean early rejection.";
        report["seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
        report["coverage_cache"]={{"mib",cache_mib},{"rasters",work.coverage_rasters},{"fields",work.coverage_fields},
            {"mask_hits",work.coverage_mask_hits},{"field_hits",work.coverage_field_hits},{"bypasses",work.coverage_cache_bypasses},
            {"peak_bytes",work.coverage_cache_peak_bytes}};
        report["stage_seconds"]={{"raster",work.raster_ns*1e-9},{"distance",work.distance_ns*1e-9}};
        checkpoint();
        std::cout<<json{{"passed",report["passed"]},{"seconds",report["seconds"]}}.dump(2)<<'\n';
        return 0; // A failed visual recheck is a recorded research result, not a tool error.
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
