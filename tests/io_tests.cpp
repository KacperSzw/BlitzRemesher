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
        nlohmann::json manifest;std::ifstream(dir/"gltf/lods.json")>>manifest;CHECK(manifest["lods"][2]["gltf_mesh"]==0);
        CHECK(manifest["lods"].size()==3&&manifest["runtime_levels"]==nlohmann::json::array({0}));
        nlohmann::json j;std::ifstream(dir/"gltf/chain.gltf")>>j;
        CHECK(j["meshes"].size()==1&&j["nodes"].size()==3&&j["nodes"][2]["mesh"]==0);
        j["nodes"][0]["scale"]={-1,2,3};std::ofstream(dir/"gltf/chain.gltf")<<j.dump();
        g=load_mesh(dir/"gltf/chain.gltf");CHECK(g.positions[1].x==-1);CHECK(g.positions[2].y==2);CHECK(g.indices[1]==2);
        auto n=cross(g.positions[g.indices[1]]-g.positions[0],g.positions[g.indices[2]]-g.positions[0]);CHECK(dot(n,g.normals[0])>0);
        g=load_gltf_mesh(dir/"gltf/chain.gltf",0);CHECK(g.positions[1].x==1&&g.positions[2].y==1&&g.indices==m.indices);
        bool bad_index=false;try{load_gltf_mesh(dir/"gltf/chain.gltf",1);}catch(...){bad_index=true;}CHECK(bad_index);
        // Equal triangle counts do not imply equal render data. A winding change
        // keeps the borrowed vertex accessors but requires a separate mesh.
        r.lods[2].data.indices={0,2,1};save_chain(r,dir/"distinct");
        std::ifstream(dir/"distinct/chain.gltf")>>j;CHECK(j["meshes"].size()==2&&j["nodes"][2]["mesh"]==1);
        CHECK(j["meshes"][0]["primitives"][0]["attributes"]==j["meshes"][1]["primitives"][0]["attributes"]);
        g=load_gltf_mesh(dir/"distinct/chain.gltf",1);CHECK(g.indices==r.lods[2].data.indices);
        std::ofstream(dir/"triangle.obj")<<"v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nf 1/1 2/2 3/3\n";
        auto o=load_mesh(dir/"triangle.obj");CHECK(o.indices.size()==3&&o.uv.size()==3);
        std::ofstream(dir/"triangle.stl")<<"solid t\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid t\n";
        CHECK(load_mesh(dir/"triangle.stl").indices.size()==3);
        auto config=settings_json(Settings{});CHECK(settings_json(settings_json(config))==config);
        bool failed=false;try{settings_json({{"levels",258}});}catch(...){failed=true;}CHECK(failed);
        std::cout<<"Import, transforms, runtime mesh sharing, shared accessors and settings round trips passed\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
