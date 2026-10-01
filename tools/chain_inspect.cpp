// Validate the engine manifest against its exported geometry and provide the
// exact configured camera transforms for presentation. This does not re-audit.
#include "blitz/io.hpp"
#include <fstream>
#include <iostream>
using namespace blitz;
using json=nlohmann::json;
namespace fs=std::filesystem;
static json read(const fs::path& p){std::ifstream f(p);json j;f>>j;return j;}
static json vector(Vec3 p){return {p.x,p.y,p.z};}
int main(int argc,char** argv){try {
    if(argc!=3)throw std::invalid_argument("blitz-chain-inspect RUN_DIRECTORY OUTPUT.json");
    const fs::path run=argv[1];const auto meta=read(run/"metadata.json");const auto s=settings_json(meta.at("config"));
    json report={{"version",1},{"run_sha256",meta.at("run_sha256")},{"assets",json::object()}};
    std::vector<fs::path> rows;for(auto& e:fs::directory_iterator(run/"rows"))if(e.path().extension()==".json")rows.push_back(e.path());
    std::sort(rows.begin(),rows.end());
    for(auto path:rows) {
        const auto row=read(path);const std::string id=row.at("id");
        if(row.at("run_sha256")!=meta.at("run_sha256"))throw std::runtime_error("row provenance mismatch");
        if(row.value("failed",false)||!row.value("complete",false)){report["assets"][id]={{"valid",false}};continue;}
        const auto dir=run/"meshes"/id;
        const auto g=read(dir/"chain.gltf"),manifest=read(dir/"lods.json"),result=row.at("result");
        if(manifest.at("lods").size()!=result.at("lods").size()||g.at("nodes").size()!=result.at("lods").size())throw std::runtime_error("export level mismatch");
        if(fs::file_size(dir/"chain.bin")!=result.at("storage").at("total_bytes").get<uint64_t>())throw std::runtime_error("export byte mismatch");
        if(result.at("added_vertex_budget_bytes").is_number()&&result.at("storage").at("added_vertex_bytes")>result.at("added_vertex_budget_bytes"))throw std::runtime_error("vertex budget exceeded");
        auto source=load_gltf_mesh(dir/"chain.gltf",g.at("nodes").at(0).at("mesh"));const auto b=bounds(source.view());
        const auto expected=schedule(b,s);
        json asset={{"valid",true},{"center",vector(b.center)},{"radius",b.radius},{"levels",json::array()}};
        for(size_t i=0;i<result.at("lods").size();++i) {
            const auto& l=result.at("lods").at(i);const size_t mi=g.at("nodes").at(i).at("mesh");
            auto mesh=load_gltf_mesh(dir/"chain.gltf",mi);
            if(mesh.view().triangles()!=l.at("triangles")||manifest.at("lods").at(i).at("gltf_mesh")!=mi)throw std::runtime_error("export geometry mismatch");
            if(i>=expected.size()||l.at("screen_pixels")!=expected[i].pixels||l.at("source_limit")!=expected[i].source||l.at("transition_limit")!=expected[i].transition)
                throw std::runtime_error("export threshold mismatch");
            for(auto field:{"screen_pixels","source_limit","transition_limit"})if(manifest.at("lods").at(i).at(field)!=l.at(field))throw std::runtime_error("manifest threshold mismatch");
            if(i)for(auto kind:{"source","adjacent"}) {
                const auto& m=l.at(kind);double limit=l.at(std::string(kind)=="source"?"source_limit":"transition_limit");
                if(!m.at("passed").get<bool>()||!m.at("complete").get<bool>()||!m.at("error_px").is_number()||m.at("error_px").get<double>()>limit||m.at("changed_area").get<double>()>s.max_changed_area)
                    throw std::runtime_error("export has a failed configured audit");
            }
            json level={{"cameras",json::array()}};
            for(auto c:cameras(b,l.at("screen_pixels"),s.audit_views))level["cameras"].push_back({{"right",vector(c.right)},{"up",vector(c.up)},{"forward",vector(c.forward)},
                {"distance",c.distance},{"focal",c.focal},{"scale",c.scale},{"perspective",c.perspective}});
            asset["levels"].push_back(std::move(level));
        }
        const auto runtime=result.at("runtime_levels");uint64_t indices=0,vertices=0;
        if(runtime.size()!=g.at("meshes").size()||runtime.size()!=result.at("runtime_storage").size())throw std::runtime_error("runtime mesh mismatch");
        for(size_t i=0;i<runtime.size();++i) {
            auto cost=result.at("runtime_storage").at(i);
            if(cost.at("scheduled_index")!=runtime.at(i))throw std::runtime_error("runtime slot mismatch");
            indices+=cost.at("index_bytes").get<uint64_t>();vertices+=cost.at("added_vertex_bytes").get<uint64_t>();
            if(cost.at("cumulative_added_vertex_bytes")!=vertices)throw std::runtime_error("cumulative storage mismatch");
        }
        if(indices!=result.at("storage").at("index_bytes")||vertices!=result.at("storage").at("added_vertex_bytes"))throw std::runtime_error("runtime byte totals mismatch");
        report["assets"][id]=std::move(asset);
    }
    fs::path output=argv[2];if(!output.parent_path().empty())fs::create_directories(output.parent_path());std::ofstream out(output);out<<report.dump(2)<<'\n';if(!out)throw std::runtime_error("cannot write inspection");
    std::cout<<report["assets"].size()<<" chains inspected\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
