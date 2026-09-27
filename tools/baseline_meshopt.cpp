#include "blitz/io.hpp"
#include <meshoptimizer.h>
#include <cfloat>
#include <iostream>
int main(int argc,char** argv) {
    try {
        if(argc!=5)throw std::invalid_argument("meshopt INPUT OUTPUT TARGET reuse|rebuild");
        auto m=blitz::load_mesh(argv[1]);size_t target=std::stoull(argv[3])*3;bool reuse=std::string(argv[4])=="reuse";
        size_t n;
        if(reuse)n=meshopt_simplify(m.indices.data(),m.indices.data(),m.indices.size(),&m.positions[0].x,m.positions.size(),sizeof(blitz::Vec3),target,FLT_MAX,0,nullptr);
        else n=meshopt_simplifyWithUpdate(m.indices.data(),m.indices.size(),&m.positions[0].x,m.positions.size(),sizeof(blitz::Vec3),nullptr,0,nullptr,0,nullptr,target,FLT_MAX,0,nullptr);
        m.indices.resize(n);m.normals.clear();m.uv.clear();m.colors.clear();m.tangents.clear();m.materials.clear();
        if(!reuse)blitz::compact(m);blitz::save_ply(m.view(),argv[2]);return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
