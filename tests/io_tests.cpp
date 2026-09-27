#include "blitz/io.hpp"
#include <fstream>
#include <iostream>
using namespace blitz;
#define CHECK(x) do{if(!(x))throw std::runtime_error("line "+std::to_string(__LINE__)+": " #x);}while(0)
int main() {
    try {
        auto dir=std::filesystem::current_path()/"io-test-output";std::filesystem::create_directories(dir);
        Mesh m;m.positions={{0,0,0},{1,0,0},{0,1,0}};m.normals={{0,0,1},{0,0,1},{0,0,1}};m.uv={{0,0},{1,0},{0,1}};m.indices={0,1,2};
        save_ply(m.view(),dir/"triangle.ply");auto p=load_mesh(dir/"triangle.ply");CHECK(p.indices==m.indices);CHECK(p.uv[1].x==1);CHECK(p.normals[0].z==1);
        Result r;r.source=m.view();r.reference_bounds=bounds(m.view());
        for(int i=0;i<3;++i){Lod l;l.data.indices=m.indices;r.lods.push_back(std::move(l));}
        save_chain(r,dir/"gltf");auto g=load_mesh(dir/"gltf/chain.gltf");CHECK(g.indices.size()==3);CHECK(g.positions.size()==3);
        nlohmann::json j;std::ifstream(dir/"gltf/chain.gltf")>>j;
        CHECK(j["meshes"][0]["primitives"][0]["attributes"]==j["meshes"][2]["primitives"][0]["attributes"]);
        j["nodes"][0]["scale"]={-1,2,3};std::ofstream(dir/"gltf/chain.gltf")<<j.dump();
        g=load_mesh(dir/"gltf/chain.gltf");CHECK(g.positions[1].x==-1);CHECK(g.positions[2].y==2);CHECK(g.indices[1]==2);
        auto n=cross(g.positions[g.indices[1]]-g.positions[0],g.positions[g.indices[2]]-g.positions[0]);CHECK(dot(n,g.normals[0])>0);
        std::ofstream(dir/"triangle.obj")<<"v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nf 1/1 2/2 3/3\n";
        auto o=load_mesh(dir/"triangle.obj");CHECK(o.indices.size()==3&&o.uv.size()==3);
        std::ofstream(dir/"triangle.stl")<<"solid t\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid t\n";
        CHECK(load_mesh(dir/"triangle.stl").indices.size()==3);
        auto config=settings_json(Settings{});CHECK(settings_json(settings_json(config))==config);
        bool failed=false;try{settings_json({{"levels",258}});}catch(...){failed=true;}CHECK(failed);
        std::cout<<"Import, mirror transform, shared accessors and settings round trips passed\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
