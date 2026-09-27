// SPDX-License-Identifier: GPL-3.0-or-later
// This optional research executable uses CGAL's GPL Surface_mesh_simplification package.
#include "blitz/io.hpp"
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Surface_mesh_simplification/edge_collapse.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Edge_count_stop_predicate.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/GarlandHeckbert_plane_policies.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/GarlandHeckbert_probabilistic_plane_policies.h>
#include <iostream>
namespace SMS=CGAL::Surface_mesh_simplification;
using K=CGAL::Simple_cartesian<double>;using Surface=CGAL::Surface_mesh<K::Point_3>;
int main(int argc,char** argv) {
    try {
        if(argc!=6||std::string(argv[4])!="rebuild")throw std::invalid_argument("CGAL adapter INPUT OUTPUT TARGET rebuild lt|qem|probabilistic");
        auto input=blitz::load_mesh(argv[1]);Surface m;std::vector<Surface::Vertex_index> vertices;
        for(auto p:input.positions)vertices.push_back(m.add_vertex({p.x,p.y,p.z}));
        for(size_t f=0;f<input.indices.size();f+=3)if(m.add_face(vertices[input.indices[f]],vertices[input.indices[f+1]],vertices[input.indices[f+2]])==Surface::null_face())
            throw std::invalid_argument("CGAL requires oriented manifold connectivity; no silent repair");
        size_t target=std::stoull(argv[3]);
        auto stop=[&](auto const&,auto const&,size_t,size_t){return m.number_of_faces()<=target;};
        std::string policy=argv[5];
        if(policy=="qem"){SMS::GarlandHeckbert_plane_policies<Surface,K> q(m);SMS::edge_collapse(m,stop,CGAL::parameters::get_cost(q.get_cost()).get_placement(q.get_placement()));}
        else if(policy=="probabilistic"){SMS::GarlandHeckbert_probabilistic_plane_policies<Surface,K> q(m);SMS::edge_collapse(m,stop,CGAL::parameters::get_cost(q.get_cost()).get_placement(q.get_placement()));}
        else if(policy=="lt")SMS::edge_collapse(m,stop);else throw std::invalid_argument("unknown CGAL policy");
        m.collect_garbage();blitz::Mesh out;
        for(auto v:m.vertices()){auto p=m.point(v);out.positions.push_back({float(p.x()),float(p.y()),float(p.z())});}
        for(auto f:m.faces())for(auto v:CGAL::vertices_around_face(m.halfedge(f),m))out.indices.push_back(uint32_t(v));
        blitz::save_ply(out.view(),argv[2]);return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
