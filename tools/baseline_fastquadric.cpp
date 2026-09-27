#include "blitz/io.hpp"
#include "Simplify.h"
#include <iostream>
int main(int argc,char** argv) {
    try {
        if(argc!=5||std::string(argv[4])!="rebuild")throw std::invalid_argument("Fast Quadric adapter supports rebuild geometry only");
        auto m=blitz::load_mesh(argv[1]);Simplify::vertices.resize(m.positions.size());Simplify::triangles.resize(m.indices.size()/3);
        for(size_t i=0;i<m.positions.size();++i){auto p=m.positions[i];Simplify::vertices[i].p=vec3f(p.x,p.y,p.z);}
        for(size_t i=0;i<Simplify::triangles.size();++i){auto& t=Simplify::triangles[i];t={};for(int k=0;k<3;++k)t.v[k]=int(m.indices[i*3+k]);}
        Simplify::simplify_mesh(std::stoi(argv[3]),7,false);
        blitz::Mesh out;for(auto& v:Simplify::vertices)out.positions.push_back({float(v.p.x),float(v.p.y),float(v.p.z)});
        for(auto& t:Simplify::triangles)if(!t.deleted)for(int k=0;k<3;++k)out.indices.push_back(uint32_t(t.v[k]));
        blitz::save_ply(out.view(),argv[2]);return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
