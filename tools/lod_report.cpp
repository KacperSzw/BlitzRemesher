#include "blitz/io.hpp"
#include "blitz/render_cost.hpp"
#include <openssl/sha.h>
#include <fstream>
#include <iostream>
#include <map>
using namespace blitz;
using json=nlohmann::json;
namespace fs=std::filesystem;
static std::string bytes(const fs::path& p) {
    std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("cannot read "+p.string());
    return {std::istreambuf_iterator<char>(f),{}};
}
static json read(const fs::path& p){return json::parse(bytes(p));}
static std::string hash(std::string_view s) {
    unsigned char digest[SHA256_DIGEST_LENGTH];SHA256(reinterpret_cast<const unsigned char*>(s.data()),s.size(),digest);
    std::string out;for(auto c:digest){out+="0123456789abcdef"[c>>4];out+="0123456789abcdef"[c&15];}return out;
}
static json ratio(uint64_t n,uint64_t d){return d?json(double(n)/double(d)):json(nullptr);}
static json cost_json(const RenderCost& c) {
    return {{"views",c.views},{"complete",c.complete},{"clipped",c.clipped},
      {"submitted",c.submitted},{"culled",c.culled},{"degenerate",c.degenerate},{"projected",c.projected},
      {"under_one_px2",c.under_one_px2},{"under_four_px2",c.under_four_px2},{"zero_samples",c.zero_samples},
      {"covered_samples",c.covered_samples},{"primitive_quads",c.primitive_quads},{"covered_pixels",c.covered_pixels},
      {"fraction_under_one_px2",ratio(c.under_one_px2,c.projected)},{"fraction_zero_samples",ratio(c.zero_samples,c.projected)},
      {"quad_lane_utilization",ratio(c.covered_samples,4*c.primitive_quads)},
      {"pre_depth_overlap",ratio(c.covered_samples,c.covered_pixels)}};
}
// Fingerprint raw export bytes, not re-normalized normals from the importer.
// This tool accepts only the tight float/u32 accessors written by save_chain.
static std::string mesh_hash(const json& g,const std::string& bin,size_t mesh) {
    auto accessor=[&](size_t index) {
        auto a=g.at("accessors").at(index);auto v=g.at("bufferViews").at(a.at("bufferView").get<size_t>());
        std::string shape=a.at("type");size_t components=shape=="SCALAR"?1:shape=="VEC2"?2:shape=="VEC3"?3:shape=="VEC4"?4:0;
        if(!components||a.contains("sparse")||v.contains("byteStride")||v.at("buffer")!=0||
           (a.at("componentType")!=5126&&a.at("componentType")!=5125))throw std::runtime_error("unsupported export accessor");
        size_t offset=v.value("byteOffset",size_t(0))+a.value("byteOffset",size_t(0));
        size_t count=a.at("count");if(count>bin.size()/(4*components))throw std::runtime_error("oversized accessor");
        size_t length=count*4*components;
        if(offset>bin.size()||length>bin.size()-offset||length>v.at("byteLength").get<size_t>())throw std::runtime_error("export buffer overrun");
        return json{{"shape",shape},{"component",a.at("componentType")},{"count",count},{"sha256",hash(std::string_view(bin).substr(offset,length))}};
    };
    json primitives=json::array();
    for(auto& p:g.at("meshes").at(mesh).at("primitives")) {
        json attributes=json::object();
        for(auto i=p.at("attributes").begin();i!=p.at("attributes").end();++i)attributes[i.key()]=accessor(i.value());
        primitives.push_back({{"attributes",attributes},{"indices",accessor(p.at("indices"))},
            {"mode",p.value("mode",4)},{"material",g.at("materials").at(p.at("material").get<size_t>())}});
    }
    return hash(primitives.dump());
}
int main(int argc,char** argv) {
    try {
        if(argc!=3)throw std::invalid_argument("blitz-lod-report RUN_DIRECTORY OUTPUT.json");
        fs::path run=argv[1],output=argv[2];auto meta=read(run/"metadata.json"),summary=read(run/"summary.json");
        if(!summary.at("complete").get<bool>())throw std::runtime_error("diagnostics require a complete batch");
        if(summary.at("run_sha256")!=meta.at("run_sha256"))throw std::runtime_error("summary hash mismatch");
        auto settings=settings_json(meta.at("config"),true);
        json report={{"version",1},{"run",run.filename().string()},{"run_sha256",meta.at("run_sha256")},
            {"summary_sha256",hash(bytes(run/"summary.json"))},{"config",meta.at("config")},{"score",summary.at("score")},
            {"proxy",{{"version",1},{"samples","pixel centers at (x+0.5,y+0.5); top-left fill; fixed even viewport; no MSAA"},
              {"quad","aligned 2x2; counted per primitive before depth testing"},
              {"culling","source winding and material double-sided flags"},
              {"interpretation","CPU geometry proxies, not GPU helper invocations or time"},
              {"views",meta.at("config").at("audit_views")}}},
            {"assets",json::array()},{"diagnostics_complete",true}};
#if defined(__linux__)
        report["binary_sha256"]=hash(bytes("/proc/self/exe"));
#endif
        struct Totals {size_t n{};double final{},tail{};};std::map<std::string,Totals> categories;
        size_t scheduled=0,runtime=0,failed=0;std::vector<fs::path> rows;
        for(auto& e:fs::directory_iterator(run/"rows"))if(e.path().extension()==".json")rows.push_back(e.path());
        std::sort(rows.begin(),rows.end());
        if(rows.size()!=summary.at("expected"))throw std::runtime_error("row count mismatch");
        for(auto& path:rows) {
            auto row=read(path);std::string id=row.at("id");
            if(!row.at("complete").get<bool>()||row.at("run_sha256")!=meta.at("run_sha256"))throw std::runtime_error("row hash or completion mismatch");
            json asset={{"id",id},{"category",row.at("category")},{"mean_ratio",row.at("ratio")},{"row_sha256",hash(bytes(path))},{"lods",json::array()}};
            auto& category=categories[row.at("category").get<std::string>()];++category.n;
            if(row.value("failed",false)) {
                ++failed;category.final+=1;category.tail+=1;asset["failure"]=row.at("failure");
                asset["final_ratio"]=1;asset["tail_ratio"]=1;report["diagnostics_complete"]=false;
                report["assets"].push_back(asset);continue;
            }
            auto levels=row.at("result").at("lods");const double source=levels.at(0).at("triangles");
            const size_t tail_start=std::max<size_t>(1,levels.size()>3?levels.size()-3:1);
            double tail=0;for(size_t i=tail_start;i<levels.size();++i)tail+=levels[i].at("triangles").get<double>()/source;
            tail/=levels.size()-tail_start;const double final=levels.back().at("triangles").get<double>()/source;
            category.final+=final;category.tail+=tail;asset["final_ratio"]=final;asset["tail_ratio"]=tail;
            unsigned repeated_counts=0;for(size_t i=1;i<levels.size();++i)repeated_counts+=levels[i].at("triangles")==levels[i-1].at("triangles");
            asset["same_count_transitions"]=repeated_counts;
            scheduled+=levels.size();
            try {
                auto dir=run/"meshes"/id;auto g=read(dir/"chain.gltf");auto bin=bytes(dir/"chain.bin");
                asset["chain_bin_sha256"]=hash(bin);asset["chain_gltf_sha256"]=hash(bytes(dir/"chain.gltf"));
                std::vector<Mesh> meshes;meshes.reserve(g.at("meshes").size());std::vector<std::string> hashes;
                for(size_t i=0;i<g.at("meshes").size();++i){meshes.push_back(load_gltf_mesh(dir/"chain.gltf",i));hashes.push_back(mesh_hash(g,bin,i));}
                const auto reference=bounds(meshes.at(g.at("nodes").at(0).at("mesh")).view());
                auto source_cost=render_cost(meshes.at(g.at("nodes").at(0).at("mesh")).view(),reference,settings.audit_views,levels.back().at("screen_pixels"));
                asset["source_render_cost_at_final"]=cost_json(source_cost);
                if(!source_cost.complete)report["diagnostics_complete"]=false;
                std::vector<uint8_t> active;std::string previous;
                for(size_t i=0;i<levels.size();++i) {
                    size_t mi=g.at("nodes").at(i).at("mesh");auto v=meshes.at(mi).view();
                    if(v.triangles()!=levels[i].at("triangles"))throw std::runtime_error("geometry/row count mismatch");
                    if(!i||hashes.at(mi)!=previous)active.push_back(uint8_t(i));previous=hashes.at(mi);
                    auto c=render_cost(v,reference,settings.audit_views,levels[i].at("screen_pixels"));
                    if(!c.complete)report["diagnostics_complete"]=false;
                    auto level=levels[i];level["render_cost"]=cost_json(c);level["export_mesh_sha256"]=hashes.at(mi);
                    asset["lods"].push_back(level);
                }
                if(row.at("result").contains("runtime_levels")&&row.at("result").at("runtime_levels")!=json(active))
                    throw std::runtime_error("runtime selection disagrees with exact export bytes");
                asset["runtime_levels"]=active;asset["runtime_lod_count"]=active.size();runtime+=active.size();
                if(row.at("result").contains("runtime_storage")) {
                    const auto& costs=row.at("result").at("runtime_storage");
                    if(costs.size()!=active.size())throw std::runtime_error("runtime storage count mismatch");
                    for(size_t i=0;i<active.size();++i)
                        if(costs.at(i).at("scheduled_index")!=active[i])throw std::runtime_error("runtime storage index mismatch");
                    asset["runtime_storage"]=costs;
                    asset["added_vertex_budget_bytes"]=row.at("result").value("added_vertex_budget_bytes",json(nullptr));
                }
                asset["final_pixel_coverage_ratio"]=ratio(asset.at("lods").back().at("render_cost").at("covered_pixels").get<uint64_t>(),source_cost.covered_pixels);
            }catch(const std::exception& e){asset["diagnostic_failure"]=e.what();report["diagnostics_complete"]=false;}
            report["assets"].push_back(asset);std::cerr<<id<<" analyzed\n";
        }
        double final=0,tail=0;
        for(auto& [name,c]:categories){final+=c.final/c.n;tail+=c.tail/c.n;report["categories"][name]={{"count",c.n},{"final_ratio",c.final/c.n},{"tail_ratio",c.tail/c.n}};}
        report["final_ratio"]=final/categories.size();report["tail_ratio"]=tail/categories.size();
        report["failed_assets"]=failed;report["scheduled_lods"]=scheduled;
        report["runtime_lods"]=report["diagnostics_complete"]==true?json(runtime):json(nullptr);
        if(!output.parent_path().empty())fs::create_directories(output.parent_path());
        std::ofstream out(output);out<<report.dump(2)<<'\n';if(!out)throw std::runtime_error("cannot write diagnostics");
        std::cout<<json{{"score",report["score"]},{"final_ratio",report["final_ratio"]},{"tail_ratio",report["tail_ratio"]},
            {"scheduled_lods",scheduled},{"runtime_lods",report["runtime_lods"]},{"complete",report["diagnostics_complete"]}}.dump(2)<<'\n';
        return report["diagnostics_complete"]==true?0:2;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
