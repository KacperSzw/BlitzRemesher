// Research-only, attribute-preserving adapter for the opaque geometry derivatives.
#include "blitz/io.hpp"
#include <meshoptimizer.h>
#include <chrono>
#include <csignal>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <map>
using namespace blitz;
static volatile std::sig_atomic_t stopped=0;
static void stop(int){stopped=1;}
int main(int argc,char** argv){try {
    if(argc<3||std::string(argv[1])!="simplify")throw std::invalid_argument("simplify INPUT --config JSON --out DIR [--meshopt-options FLAGS]");
    std::signal(SIGTERM,stop);std::signal(SIGINT,stop);Settings s;std::filesystem::path out;unsigned options=0;
    for(int i=3;i<argc;i+=2){if(i+1==argc)throw std::invalid_argument("option requires value");std::string k=argv[i];
        if(k=="--config"){std::ifstream f(argv[i+1]);nlohmann::json j;f>>j;s=settings_json(j);}else if(k=="--out")out=argv[i+1];else if(k=="--meshopt-options")options=unsigned(std::stoul(argv[i+1]));else throw std::invalid_argument("unknown option");}
    if(options&~unsigned(meshopt_SimplifyPrune|meshopt_SimplifyPermissive))throw std::invalid_argument("unsupported meshoptimizer flags");
    auto source=load_mesh(argv[2]);if(s.profile!=Profile::Coverage||!source.normals.empty()||!source.colors.empty()||!source.tangents.empty())throw std::invalid_argument("adapter supports geometry/UV coverage derivatives only");
    s.cancelled=[]{return stopped!=0;};PerformanceStats work;s.performance=&work;
    auto proposer=[&](MeshView input,const ReduceSettings& rs) {
        Lod l;l.shared_vertices=rs.output==OutputMode::Reuse;l.data=copy_mesh(input);l.data.indices.clear();l.data.materials.clear();
        std::map<uint16_t,std::vector<uint32_t>> parts;std::vector<uint16_t> material(input.positions.count);std::vector<uint8_t> seen(input.positions.count),lock(input.positions.count,meshopt_SimplifyVertex_Protect);
        for(size_t f=0;f<input.triangles();++f){auto mat=input.material(f);for(unsigned k=0;k<3;++k){auto v=input.indices[3*f+k];parts[mat].push_back(v);if(seen[v]&&material[v]!=mat)lock[v]|=meshopt_SimplifyVertex_Lock;seen[v]=1;material[v]=mat;}}
        const float weights[]={1,1};
        for(auto& [mat,indices]:parts) {
            size_t target=std::max<size_t>(1,rs.target_triangles*(indices.size()/3)/input.triangles())*3;
            auto attrs=l.data.uv.empty()?nullptr:&l.data.uv[0].x;auto count=l.data.uv.empty()?0u:2u;
            size_t n=l.shared_vertices?
                meshopt_simplifyWithAttributes(indices.data(),indices.data(),indices.size(),&l.data.positions[0].x,l.data.positions.size(),sizeof(Vec3),attrs,sizeof(Vec2),weights,count,lock.data(),target,FLT_MAX,options,nullptr):
                meshopt_simplifyWithUpdate(indices.data(),indices.size(),&l.data.positions[0].x,l.data.positions.size(),sizeof(Vec3),attrs,sizeof(Vec2),weights,count,lock.data(),target,FLT_MAX,options,nullptr);
            l.data.indices.insert(l.data.indices.end(),indices.begin(),indices.begin()+n);l.data.materials.insert(l.data.materials.end(),n/3,mat);
        }
        if(l.shared_vertices){l.data.positions.clear();l.data.uv.clear();}else compact(l.data);return l;
    };
    auto begin=std::chrono::steady_clock::now();auto r=generate(source.view(),s,proposer);auto generated=std::chrono::steady_clock::now();save_chain(r,out);auto exported=std::chrono::steady_clock::now();
    auto j=result_json(r);j["generation_seconds"]=std::chrono::duration<double>(generated-begin).count();j["export_seconds"]=std::chrono::duration<double>(exported-generated).count();j["seconds"]=std::chrono::duration<double>(exported-begin).count();j["meshopt_options"]=options;j["output"]=out.string();
    std::cout<<j.dump(2)<<'\n';return r.status==Status::Complete?0:2;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
