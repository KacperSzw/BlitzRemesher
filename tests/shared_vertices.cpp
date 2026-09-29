#include "shared_vertices.hpp"
#include <iostream>
#include <stdexcept>
using namespace blitz;
#define CHECK(x) do{if(!(x))throw std::runtime_error(std::string(__FILE__)+":"+std::to_string(__LINE__)+": " #x);}while(0)
static Mesh source_mesh() {
    Mesh m;m.positions={{-1,-1,0},{1,-1,0},{1,1,0},{-1,1,0},{0,0,0},{.2f,0,0}};m.indices={0,1,4,1,2,5,2,3,5,3,0,4,4,1,5,4,5,3};m.double_sided={1};
    for(auto p:m.positions){m.normals.push_back({0,0,1});m.uv.push_back({(p.x+1)*.5f,(p.y+1)*.5f});m.colors.push_back({71,133,211,77});m.tangents.push_back({1,0,0,-1});}return m;
}
static Lod candidate(const Mesh& source,float x) {
    Lod l;l.shared_vertices=false;l.data=source;l.data.indices={0,1,4,1,2,4,2,3,4,3,0,4};
    l.data.positions[4]={x,0,0};l.data.uv[4]={(x+1)*.5f,.5f};compact(l.data);return l;
}
int main(){try {
    const auto source=source_mesh(),original=source;detail::SourceVertices table(source.view(),true);
    auto a=candidate(source,.1f);auto expected=a.data;table.share(a);
    CHECK(a.source_prefix_vertices==6&&a.vertex_pool&&a.vertex_pool->positions.size()==7);
    CHECK(added_vertex_bytes(a,source.view())==52&&a.data.positions.capacity()==0);
    auto compacted=copy_mesh(a.view(source.view()));compact(compacted);CHECK(same_mesh_data(compacted.view(),expected.view()));
    CHECK(same_mesh_data(source.view(),original.view()));
    auto shifted=source;for(auto& p:shifted.positions){p.x=p.x*.12731f+.432019f;p.y=p.y*.27131f+.315379f;}
    ReduceSettings rs;rs.target_triangles=4;rs.preserve_positions=true;auto endpoints=reduce(shifted.view(),rs);CHECK(endpoints.view(shifted.view()).triangles()<shifted.view().triangles());
    for(auto p:endpoints.data.positions)CHECK(std::any_of(shifted.positions.begin(),shifted.positions.end(),[&](auto q){return std::memcmp(&p,&q,sizeof(p))==0;}));
    for(unsigned field=0;field<6;++field){Lod l;l.shared_vertices=false;l.data=source;l.data.indices={0,1,2};compact(l.data);
        switch(field){case 0:l.data.positions[0].x+=.1f;break;case 1:l.data.normals[0].x=.1f;break;case 2:l.data.uv[0].x+=.1f;break;case 3:l.data.colors[0].r^=1;break;case 4:l.data.tangents[0].x+=.1f;break;default:l.data.colors[0].a^=1;}
        auto before=l.data;table.share(l);CHECK(added_vertex_bytes(l,source.view())==52);
        auto after=copy_mesh(l.view(source.view()));compact(after);CHECK(same_mesh_data(after.view(),before.view()));}
    Lod exact;exact.shared_vertices=false;exact.data=source;exact.data.indices={0,1,2};compact(exact.data);table.share(exact);
    CHECK(exact.shared_vertices&&!exact.vertex_pool&&added_vertex_bytes(exact,source.view())==0);
    struct Padded {Vec3 value;uint32_t padding;};std::vector<Padded> padded;
    for(auto p:source.positions)padded.push_back({p,0xfedcba98u});auto strided=source.view();strided.positions={};
    strided.positions.data=reinterpret_cast<const std::byte*>(padded.data());strided.positions.count=padded.size();strided.positions.stride=sizeof(Padded);
    detail::SourceVertices strided_table(strided,true);auto strided_lod=candidate(source,.1f);strided_table.share(strided_lod);
    CHECK(added_vertex_bytes(strided_lod,strided)==52&&strided_lod.source_prefix_vertices==6);
    for(auto p:padded)CHECK(p.padding==0xfedcba98u);
    // Independent buffers combine into one shared source prefix and two changed
    // vertices. Runtime index buffers stay distinct, repeated levels stay shared.
    auto b=candidate(source,.3f);table.share(b);Result packed;packed.source=source.view();Lod root;root.data.indices=source.indices;
    packed.lods={root,a,b,a};CHECK(runtime_levels(packed).size()==4);
    CHECK(storage_stats(packed).added_vertex_bytes==104&&runtime_storage(packed).back().added_vertex_bytes==0);
    packed.lods={root,a,b,b};const auto before=storage_stats(packed);detail::share_result_vertices(packed);
    CHECK(storage_stats(packed).total()==before.total()&&storage_stats(packed).added_vertex_bytes==104);
    CHECK(runtime_levels(packed).size()==3&&runtime_storage(packed)[1].added_vertex_bytes==104&&runtime_storage(packed)[2].added_vertex_bytes==0);
    CHECK(packed.lods[1].vertex_pool==packed.lods[2].vertex_pool&&packed.source.positions.data==packed.lods[1].view(packed.source).positions.data);
    CHECK(packed.lods[1].vertex_pool->positions.size()==8&&packed.lods[1].data.indices[2]!=packed.lods[2].data.indices[2]);
    auto saved=packed.lods[1];auto view=saved.view(packed.source);packed={};CHECK(validate(view).empty()&&view.positions.count==8);
    // Equal-cost paths can retain different pools. After converging at a
    // source-only mesh, only the path owning B can afford another B index mesh.
    Mesh grid;grid.double_sided={1};
    for(unsigned y=0;y<3;++y)for(unsigned x=0;x<3;++x)grid.positions.push_back({float(x)-1,float(y)-1,0});
    for(unsigned y=0;y<2;++y)for(unsigned x=0;x<2;++x){auto i=y*3+x;grid.indices.insert(grid.indices.end(),{i,i+1,i+4,i,i+4,i+3});}
    detail::SourceVertices grid_table(grid.view(),true);Lod pa,pb;pa.shared_vertices=pb.shared_vertices=false;pa.data=pb.data=grid;
    pa.data.positions[4].x=.1f;pb.data.positions[4].y=.1f;pa.data.indices.resize(18);pb.data.indices.resize(18);
    compact(pa.data);compact(pb.data);grid_table.share(pa);grid_table.share(pb);Lod end=pb;end.data.indices.resize(12);
    CHECK(added_vertex_bytes(pa,grid.view())==12&&added_vertex_bytes(pb,grid.view())==12);
    for(uint32_t cap:{1112,2000}) {
        Settings s;s.levels=4;s.base_pixels=s.last_pixels=16;s.profile=Profile::Coverage;s.transition={{{0,100},{1,100}}};s.max_lod0_delta_px=100;
        s.search_views=s.audit_views={6,0,47};s.search_supersample=s.audit_supersample=s.max_supersample=4;s.max_changed_area=1;
        s.candidate_budget=2;s.beam_width=2;s.max_added_vertex_bytes_bps=cap;s.research.graph_passes=1;s.research.shared_rebuild=true;
        unsigned calls=0;auto r=generate(grid.view(),s,[&](MeshView input,const ReduceSettings&){Lod l;
            if(++calls<=6){l.data.indices.assign(input.indices.begin(),input.indices.end());return l;}
            if(calls==7)return pa;if(calls==8)return pb;if(calls<=10){l.data.indices.assign(grid.indices.begin(),grid.indices.begin()+15);return l;}return end;});
        CHECK(r.status==Status::Complete&&r.lods[1].data.indices.size()==18&&r.lods[2].data.indices.size()==15&&r.lods[3].data.indices.size()==12);
        CHECK(storage_stats(r).added_vertex_bytes==12&&runtime_storage(r).back().added_vertex_bytes==0);
    }
    // The same moderate reduction is unaffordable as a complete rebuilt buffer,
    // and affordable when only its one changed vertex is newly stored.
    for(bool graph:{false,true})for(uint32_t cap:{1000,2000,4000})for(bool sharing:{false,true}) {
        Settings s;s.levels=2;s.base_pixels=s.last_pixels=16;s.profile=Profile::Attributes;s.transition={{{0,2},{1,2}}};s.max_lod0_delta_px=2;s.max_changed_area=.1;
        s.search_views=s.audit_views={6,0,43};s.search_supersample=s.audit_supersample=s.max_supersample=8;s.candidate_budget=2;s.beam_width=2;s.max_added_vertex_bytes_bps=cap;
        s.research.graph_passes=graph;s.research.shared_rebuild=sharing;
        auto r=generate(source.view(),s,[&](MeshView input,const ReduceSettings& rs){if(rs.output==OutputMode::Rebuild)return candidate(source,.1f);Lod l;l.data.indices.assign(input.indices.begin(),input.indices.end());return l;});
        CHECK(r.status==Status::Complete&&storage_stats(r).added_vertex_bytes<=*r.added_vertex_budget_bytes);
        const bool fits=sharing&&cap>=2000;CHECK(r.lods.back().view(r.source).triangles()==(fits?4:6));
        CHECK(storage_stats(r).added_vertex_bytes==(fits?52:0));CHECK(same_mesh_data(r.source,source.view()));
        CHECK(r.lods.back().source_error.passed&&r.lods.back().adjacent.passed);
    }
    std::cout<<"Exact source sharing, ownership, mixed storage and capped generation contracts passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
