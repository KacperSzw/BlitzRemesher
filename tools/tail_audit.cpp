#include "blitz/io.hpp"
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
static json measurement(Measurement m) {
    return {{"passed",m.passed},{"complete",m.complete},{"error_px",std::isfinite(m.error)?json(m.error):json(nullptr)},
        {"nonfinite_error",!std::isfinite(m.error)},{"coverage_upper_px",std::isfinite(m.coverage_upper)?json(m.coverage_upper):json(nullptr)},
        {"views",m.views_evaluated},{"worst_view",m.worst_view},{"supersample",m.supersample},{"resource_limited",m.resource_limited}};
}
int main(int argc,char** argv){
    try{
        if(argc!=4)throw std::invalid_argument("blitz-tail-audit RUN_DIRECTORY ASSET_ID OUTPUT.json");
        fs::path run=argv[1],dir=run/"meshes"/argv[2];auto row=read(run/"rows"/(std::string(argv[2])+".json")),meta=read(run/"metadata.json");
        if(!row.at("complete").get<bool>()||row.value("failed",false)||row.at("run_sha256")!=meta.at("run_sha256"))throw std::runtime_error("invalid audit input");
        auto settings=settings_json(meta.at("config"));auto g=read(dir/"chain.gltf");auto levels=row.at("result").at("lods");
        auto source=load_gltf_mesh(dir/"chain.gltf",g.at("nodes").at(0).at("mesh"));auto reference=bounds(source.view());
        json report={{"version",1},{"id",argv[2]},{"run",run.filename().string()},{"run_sha256",meta.at("run_sha256")},
            {"chain_gltf_sha256",hash(dir/"chain.gltf")},{"chain_bin_sha256",hash(dir/"chain.bin")},
            {"row_sha256",hash(run/"rows"/(std::string(argv[2])+".json"))},{"profile",meta.at("config").at("profile")},
            {"purpose","Dense independent tail check; does not rewrite pilot SCORE or select replacement meshes"},
            {"audit",{{"orthographic",642},{"perspective",64},{"seed",uint32_t(0xB1172026)},{"supersample",8},{"max_supersample",32}}},
            {"lods",json::array()},{"passed",true}};
#if defined(__linux__)
        report["binary_sha256"]=hash("/proc/self/exe");
#endif
        auto begin=std::chrono::steady_clock::now();
        const size_t first=std::max<size_t>(1,levels.size()>3?levels.size()-3:1);
        for(size_t i=first;i<levels.size();++i) {
            auto current=load_gltf_mesh(dir/"chain.gltf",g.at("nodes").at(i).at("mesh"));
            auto previous=load_gltf_mesh(dir/"chain.gltf",g.at("nodes").at(i-1).at("mesh"));
            EvalSettings e;e.profile=settings.profile;e.weights=settings.weights;e.screen_size=levels[i].at("screen_pixels");
            double t=settings.levels==2?0:double(i-1)/(settings.levels-2);
            e.weights.normal*=settings.normal_importance.at(t);e.weights.color*=settings.attribute_importance.at(t);e.weights.material*=settings.attribute_importance.at(t);
            e.limit=levels[i].at("source_limit");auto src=evaluate(source.view(),current.view(),reference,e);
            e.limit=levels[i].at("transition_limit");auto adj=evaluate(previous.view(),current.view(),reference,e);
            if(!src.passed||!adj.passed)report["passed"]=false;
            report["lods"].push_back({{"level",i},{"triangles",current.view().triangles()},{"screen_pixels",e.screen_size},
                {"source_limit",levels[i].at("source_limit")},{"transition_limit",e.limit},{"source",measurement(src)},{"adjacent",measurement(adj)}});
            std::cerr<<argv[2]<<" LOD"<<i<<" source="<<src.passed<<" adjacent="<<adj.passed<<'\n';
        }
        report["seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
        fs::path output=argv[3];if(!output.parent_path().empty())fs::create_directories(output.parent_path());
        std::ofstream out(output);out<<report.dump(2)<<'\n';if(!out)throw std::runtime_error("cannot write tail audit");
        std::cout<<json{{"passed",report["passed"]},{"seconds",report["seconds"]}}.dump(2)<<'\n';
        return 0; // A failed visual recheck is a recorded research result, not a tool error.
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
