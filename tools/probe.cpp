#include "blitz/io.hpp"
#include <openssl/sha.h>
#include <chrono>
#include <fstream>
#include <iostream>
using namespace blitz;
using json=nlohmann::json;
namespace fs=std::filesystem;
static json read(const fs::path& p){std::ifstream f(p);if(!f)throw std::runtime_error("cannot open "+p.string());json j;f>>j;return j;}
static std::string hash(const fs::path& p){std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("cannot hash "+p.string());
    std::string s{std::istreambuf_iterator<char>(f),{}};unsigned char digest[SHA256_DIGEST_LENGTH];SHA256(reinterpret_cast<const unsigned char*>(s.data()),s.size(),digest);
    std::string out;for(auto c:digest){out+="0123456789abcdef"[c>>4];out+="0123456789abcdef"[c&15];}return out;}
int main(int argc,char** argv){
    try{
        if(argc!=3)throw std::invalid_argument("blitz-probe MANIFEST OUTPUT.json");
        auto manifest=read(argv[1]);json report={{"purpose","Unaudited reducer-floor diagnosis; not a visual score"},{"compiler",__VERSION__},
            {"manifest_sha256",hash(argv[1])},{"build",read("research/build.json")},{"assets",json::array()}};
#if defined(__linux__)
        report["binary_sha256"]=hash("/proc/self/exe");
#endif
        for(auto& a:manifest.at("assets")){
            for(auto& file:a.at("files"))if(hash(file.at("path").get<std::string>())!=file.at("sha256").get<std::string>())throw std::runtime_error("source hash mismatch");
            auto source=load_mesh(a.at("path").get<std::string>());
            json asset={{"id",a.at("id")},{"source_files",a.at("files")},{"proposals",json::array()}};
            for(auto objective:{Objective::Quadric,Objective::TopologyRelaxed})for(bool geometry_only:{false,true})for(size_t target:{1u,8u,16u,32u,64u,128u}){
                auto v=source.view();if(geometry_only){v.normals={};v.uv={};v.colors={};v.tangents={};}
                ReductionStats stats;ReduceSettings settings;settings.objective=objective;settings.coupled_wedges=true;settings.target_triangles=target;settings.statistics=&stats;
                auto begin=std::chrono::steady_clock::now();auto lod=reduce(v,settings);
                asset["proposals"].push_back({{"objective",objective==Objective::Quadric?"quadric":"topology_relaxed"},{"geometry_only",geometry_only},{"requested",target},{"achieved",lod.view(v).triangles()},
                    {"seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()},
                    {"passes",stats.passes},{"attempts",stats.attempts},{"collapsed",stats.collapsed},
                    {"geometry_rejections",stats.geometry_rejections},{"uv_rejections",stats.uv_rejections},{"link_rejections",stats.link_rejections},
                    {"last_candidates",stats.last_candidates},{"last_locked_edges",stats.last_locked_edges},{"first_locked_edges",stats.first_locked_edges}});
            }
            report["assets"].push_back(asset);std::cerr<<a.at("id")<<" complete\n";
        }
        std::ofstream out(argv[2]);out<<report.dump(2)<<'\n';if(!out)throw std::runtime_error("cannot write report");
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
