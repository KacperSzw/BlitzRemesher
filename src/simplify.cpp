#include "blitz/remesher.hpp"
#include <bit>
#include <numeric>
#include <stdexcept>
namespace blitz {
namespace {
struct Quadric {
    double a[10]{};
    Quadric& operator+=(const Quadric& b) {for(int i=0;i<10;++i)a[i]+=b.a[i];return *this;}
    void plane(double x,double y,double z,double w,double weight) {
        double v[4]={x,y,z,w};int k=0;
        for(int i=0;i<4;++i)for(int j=i;j<4;++j)a[k++]+=v[i]*v[j]*weight;
    }
    double cost(Vec3 p) const {
        double x=p.x,y=p.y,z=p.z;
        return std::max(0.0,a[0]*x*x+2.0*a[1]*x*y+2.0*a[2]*x*z+2.0*a[3]*x+a[4]*y*y+2.0*a[5]*y*z+2.0*a[6]*y+a[7]*z*z+2.0*a[8]*z+a[9]);
    }
    bool solve(Vec3& p,ReductionStats* stats) const {
        if(stats)++stats->solve_attempts;
        double a00=a[0],a01=a[1],a02=a[2],a11=a[4],a12=a[5],a22=a[7];
        double c00=a11*a22-a12*a12,c01=a02*a12-a01*a22,c02=a01*a12-a02*a11;
        double c11=a00*a22-a02*a02,c12=a01*a02-a00*a12,c22=a00*a11-a01*a01;
        double det=a00*c00+a01*c01+a02*c02,scale=std::max({std::abs(a00),std::abs(a11),std::abs(a22)});
        if(!std::isfinite(det)||!std::isfinite(scale)){if(stats)++stats->nonfinite_solves;return false;}
        if(!(std::abs(det)>1e-12*scale*scale*scale)){if(stats)++stats->singular_solves;return false;}
        p={float(-(c00*a[3]+c01*a[6]+c02*a[8])/det),float(-(c01*a[3]+c11*a[6]+c12*a[8])/det),float(-(c02*a[3]+c12*a[6]+c22*a[8])/det)};
        if(!finite(p)){if(stats)++stats->nonfinite_solves;return false;}return true;
    }
};
static_assert(sizeof(Quadric)==80);
struct PositionKey {uint32_t x,y,z;bool operator==(const PositionKey&)const=default;};
struct PositionEntry {PositionKey key;uint32_t index;};
static_assert(sizeof(PositionEntry)==16);
struct Candidate {double cost;uint32_t u,v;Vec3 point;};
static_assert(sizeof(Candidate)==32);
struct Trace {std::vector<Vec3> positions;std::vector<uint32_t> faces;};
constexpr uint8_t Locked=1,Boundary=2,Used=4,MaterialSeen=8,LinkNeighbor=16,LinkOpposite=32;
}
ReductionStorage reduction_storage() {return {uint8_t(sizeof(Quadric)),uint8_t(sizeof(Candidate))};}
static Lod reduce_impl(MeshView source,const ReduceSettings& settings,Trace* trace) {
    if(auto error=validate(source);!error.empty())throw std::invalid_argument(error);
    if(unsigned(settings.output)>1||unsigned(settings.objective)>3||!std::isfinite(settings.normal_weight)||settings.normal_weight<0
       ||!std::isfinite(settings.regularization)||settings.regularization<0)throw std::invalid_argument("invalid reduction settings");
    Lod result;result.shared_vertices=settings.output==OutputMode::Reuse;
    auto stats=settings.statistics;
    if(stats){*stats={};stats->initial_triangles=uint32_t(source.triangles());stats->final_triangles=stats->initial_triangles;}
    Mesh mesh=copy_mesh(source);const size_t n=mesh.positions.size();
    std::vector<uint32_t> history;
    if(trace){history.resize(n);std::iota(history.begin(),history.end(),0);trace->faces.resize(source.triangles());std::iota(trace->faces.begin(),trace->faces.end(),0);trace->positions=mesh.positions;}
    auto b=bounds(source);const double scale=b.diameter();
    std::vector<Vec3> p(n);std::vector<Quadric> q(n);
    std::vector<uint8_t> flags(n);
    std::vector<uint16_t> mat(n);
    std::vector<PositionEntry> positions;positions.reserve(n);
    for(uint32_t i=0;i<n;++i) {
        auto v=mesh.positions[i];
        p[i]={float((double(v.x)-b.center.x)/scale),float((double(v.y)-b.center.y)/scale),float((double(v.z)-b.center.z)/scale)};
        PositionKey key{std::bit_cast<uint32_t>(v.x==0?0.f:v.x),std::bit_cast<uint32_t>(v.y==0?0.f:v.y),std::bit_cast<uint32_t>(v.z==0?0.f:v.z)};
        positions.push_back({key,i});
        if(settings.objective==Objective::Regularized) {
            q[i].plane(1,0,0,-p[i].x,settings.regularization);
            q[i].plane(0,1,0,-p[i].y,settings.regularization);
            q[i].plane(0,0,1,-p[i].z,settings.regularization);
        }
    }
    std::sort(positions.begin(),positions.end(),[](auto& a,auto& z){
        return std::array{a.key.x,a.key.y,a.key.z}<std::array{z.key.x,z.key.y,z.key.z};});
    for(size_t i=1;i<positions.size();++i)if(positions[i-1].key==positions[i].key)
        flags[positions[i-1].index]=flags[positions[i].index]=Locked; // Preserve attribute wedges without per-vertex allocations.
    for(size_t f=0;f<source.triangles();++f) {
        uint32_t ids[3]={mesh.indices[f*3],mesh.indices[f*3+1],mesh.indices[f*3+2]};
        Vec3 normal=cross(p[ids[1]]-p[ids[0]],p[ids[2]]-p[ids[0]]);
        double area=length(normal);if(area<=1e-24)continue;
        normal=normal*(1/area);double w=-dot(normal,p[ids[0]]);
        for(auto i:ids) {
            q[i].plane(normal.x,normal.y,normal.z,w,area);
            auto m=source.material(f);if((flags[i]&MaterialSeen)&&mat[i]!=m)flags[i]|=Locked;mat[i]=m;flags[i]|=MaterialSeen;
        }
    }
    std::vector<PositionEntry>().swap(positions);
    std::vector<uint16_t>().swap(mat);
    if(mesh.indices.size()/3>settings.target_triangles) {
        // Degenerate faces do not contribute surface area or the live collapse budget.
        // If there is no surface at all, preserve the exact input as the only incumbent.
        size_t dst=0;
        for(size_t f=0;f<mesh.indices.size()/3;++f) {
            auto a=mesh.indices[f*3],b1=mesh.indices[f*3+1],c=mesh.indices[f*3+2];
            if(a==b1||a==c||b1==c||length(cross(p[b1]-p[a],p[c]-p[a]))==0)continue;
            for(int k=0;k<3;++k)mesh.indices[dst*3+k]=mesh.indices[f*3+k];
            if(trace)trace->faces[dst]=trace->faces[f];
            if(!mesh.materials.empty())mesh.materials[dst]=mesh.materials[f];++dst;
        }
        if(!dst) {
            result.shared_vertices=true;result.data.indices.assign(source.indices.begin(),source.indices.end());
            result.data.materials.assign(source.materials.begin(),source.materials.end());return result;
        }
        mesh.indices.resize(dst*3);if(!mesh.materials.empty())mesh.materials.resize(dst);
        if(trace)trace->faces.resize(dst);
    }
    size_t target=std::max<size_t>(1,settings.target_triangles);
    if(settings.prune&&mesh.indices.size()/3>target) {
        std::vector<uint32_t> component(n);std::iota(component.begin(),component.end(),0);
        auto root=[&](uint32_t v){while(component[v]!=v){component[v]=component[component[v]];v=component[v];}return v;};
        for(size_t f=0;f<mesh.indices.size();f+=3)for(int k=1;k<3;++k){auto a=root(mesh.indices[f]),b1=root(mesh.indices[f+k]);component[b1]=a;}
        struct Part {uint32_t id;size_t faces;double area;};
        std::vector<Part> parts(n);double total=0;
        for(size_t f=0;f<mesh.indices.size();f+=3){auto a=mesh.indices[f],b1=mesh.indices[f+1],c=mesh.indices[f+2];auto id=root(a);
            double area=length(cross(p[b1]-p[a],p[c]-p[a]));parts[id].id=id;++parts[id].faces;parts[id].area+=area;total+=area;}
        std::sort(parts.begin(),parts.end(),[](auto& a,auto& b1){return a.area<b1.area;});
        std::vector<uint8_t> removed(n);size_t remaining=mesh.indices.size()/3;
        // This only proposes removals. The independent camera audit must still accept them.
        for(auto part:parts)if(part.faces&&part.area<total*.002&&remaining>=target+part.faces){removed[part.id]=1;remaining-=part.faces;}
        size_t dst=0;
        for(size_t f=0;f<mesh.indices.size()/3;++f)if(!removed[root(mesh.indices[f*3])]) {
            for(int k=0;k<3;++k)mesh.indices[dst*3+k]=mesh.indices[f*3+k];
            if(trace)trace->faces[dst]=trace->faces[f];
            if(!mesh.materials.empty())mesh.materials[dst]=mesh.materials[f];++dst;
        }
        mesh.indices.resize(dst*3);if(!mesh.materials.empty())mesh.materials.resize(dst);
        if(trace)trace->faces.resize(dst);
    }
    std::vector<uint32_t> offsets(n+1),adj,map(n);
    std::vector<uint64_t> edges;std::vector<Candidate> candidates;
    for(unsigned pass=0;pass<128 && mesh.indices.size()/3>target;++pass) {
        if(settings.cancelled&&settings.cancelled())break;
        if(stats){++stats->passes;stats->last_candidates=0;stats->last_locked_edges=0;}
        std::fill(offsets.begin(),offsets.end(),0);
        for(auto i:mesh.indices)++offsets[i+1];
        std::partial_sum(offsets.begin(),offsets.end(),offsets.begin());
        adj.resize(mesh.indices.size());auto cursor=offsets;
        edges.clear();edges.reserve(mesh.indices.size());
        for(uint32_t f=0;f<mesh.indices.size()/3;++f) {
            for(int k=0;k<3;++k) {
                auto u=mesh.indices[f*3+k],v=mesh.indices[f*3+(k+1)%3];
                adj[cursor[u]++]=f;if(u>v)std::swap(u,v);
                if(u!=v)edges.push_back((uint64_t(u)<<32)|v);
            }
        }
        std::sort(edges.begin(),edges.end());
        for(auto& f:flags)f&=Locked;candidates.clear();
        for(size_t i=0;i<edges.size();) {
            size_t j=i+1;while(j<edges.size()&&edges[j]==edges[i])++j;
            auto u=uint32_t(edges[i]>>32),v=uint32_t(edges[i]);
            if(j-i==1){flags[u]|=Boundary;flags[v]|=Boundary;}
            // An optional topology-changing proposal. Attribute/material locks,
            // face orientation checks and all generate() visual gates still apply.
            if(j-i>2&&settings.objective!=Objective::TopologyRelaxed){flags[u]|=Locked;flags[v]|=Locked;}
            i=j;
        }
        for(size_t i=0;i<edges.size();) {
            size_t j=i+1;while(j<edges.size()&&edges[j]==edges[i])++j;
            auto u=uint32_t(edges[i]>>32),v=uint32_t(edges[i]);i=j;
            if((flags[u]|flags[v])&Locked){if(stats)++stats->last_locked_edges;continue;}
            Quadric sum=q[u];sum+=q[v];
            if((flags[v]&Boundary)&&!(flags[u]&Boundary))std::swap(u,v);
            Vec3 point=p[u];double cost=sum.cost(point);
            if(!(flags[u]&Boundary) && sum.cost(p[v])<cost) {std::swap(u,v);point=p[u];cost=sum.cost(point);}
            if(settings.output==OutputMode::Rebuild&&!((flags[u]|flags[v])&Boundary)) {
                Vec3 x;
                if(sum.solve(x,stats)&&length(x-(p[u]+p[v])*.5)<=2*length(p[u]-p[v])&&sum.cost(x)<cost){point=x;cost=sum.cost(x);}
                else if(stats)++stats->position_fallbacks;
                x=(p[u]+p[v])*.5;if(sum.cost(x)<cost){point=x;cost=sum.cost(x);}
            }
            if(!mesh.normals.empty()) {
                double bend=std::max(0.0,1-dot(normalized(mesh.normals[u]),normalized(mesh.normals[v])));
                cost+=settings.normal_weight*bend*dot(p[u]-p[v],p[u]-p[v])*1e-3;
                if(settings.objective==Objective::Visual)cost+=bend*bend*dot(p[u]-p[v],p[u]-p[v])*.05;
            }
            if(stats&&!std::isfinite(cost))++stats->nonfinite_costs;
            candidates.push_back({cost,u,v,point});
        }
        if(stats){stats->last_candidates=uint32_t(candidates.size());if(pass==0)stats->first_locked_edges=stats->last_locked_edges;}
        std::sort(candidates.begin(),candidates.end(),[](auto& a,auto& b){if(a.cost!=b.cost)return a.cost<b.cost;return std::pair(a.u,a.v)<std::pair(b.u,b.v);});
        std::iota(map.begin(),map.end(),0);
        size_t remaining=mesh.indices.size()/3,collapsed=0;
        for(auto& c:candidates) {
            if(remaining<=target)break;
            auto u=c.u,v=c.v;if((flags[u]|flags[v])&Used)continue;
            if(stats)++stats->attempts;
            if(settings.objective!=Objective::TopologyRelaxed) {
                // Reject contractions that would create a new edge with >2
                // incident faces. Otherwise these edges become permanent locks
                // and prevent useful coarse LODs later in the reduction.
                unsigned edge_faces=0;
                for(uint32_t k=offsets[u];k<offsets[u+1];++k) {
                    auto f=adj[k];bool opposite=false;
                    for(int j=0;j<3;++j)opposite|=mesh.indices[f*3+j]==v;
                    edge_faces+=opposite;
                    for(int j=0;j<3;++j){auto a=mesh.indices[f*3+j];if(a!=u&&a!=v)flags[a]|=LinkNeighbor|(opposite?LinkOpposite:0);}
                }
                bool link_ok=!(flags[u]&Boundary)||!(flags[v]&Boundary)||edge_faces==1;
                for(uint32_t k=offsets[v];k<offsets[v+1]&&link_ok;++k)for(int j=0;j<3;++j) {
                    auto a=mesh.indices[adj[k]*3+j];
                    if(a!=u&&a!=v&&(flags[a]&LinkNeighbor)&&!(flags[a]&LinkOpposite))link_ok=false;
                }
                for(uint32_t k=offsets[u];k<offsets[u+1];++k)for(int j=0;j<3;++j)
                    flags[mesh.indices[adj[k]*3+j]]&=uint8_t(~(LinkNeighbor|LinkOpposite));
                if(!link_ok){if(stats)++stats->link_rejections;continue;}
            }
            bool valid=true,bad_geometry=false,bad_uv=false;size_t removed=0;
            for(auto vertex:{u,v}) for(uint32_t k=offsets[vertex];k<offsets[vertex+1]&&valid;++k) {
                auto f=adj[k];uint32_t a[3]={map[mesh.indices[3*f]],map[mesh.indices[3*f+1]],map[mesh.indices[3*f+2]]};
                bool hasu=a[0]==u||a[1]==u||a[2]==u,hasv=a[0]==v||a[1]==v||a[2]==v;
                if(hasu&&hasv) {if(vertex==u)++removed;continue;}
                Vec3 oldn=cross(p[a[1]]-p[a[0]],p[a[2]]-p[a[0]]);
                Vec3 np[3];for(int j=0;j<3;++j)np[j]=(a[j]==u||a[j]==v)?c.point:p[a[j]];
                Vec3 newn=cross(np[1]-np[0],np[2]-np[0]);
                if(length(newn)<1e-15||dot(oldn,newn)<=.05*length(oldn)*length(newn)){valid=false;bad_geometry=true;}
                if(!mesh.uv.empty()&&settings.output==OutputMode::Rebuild) {
                    auto edge=p[v]-p[u];double len2=dot(edge,edge),t=len2?std::clamp(dot(c.point-p[u],edge)/len2,0.0,1.0):.5;
                    auto ua=mesh.uv[u],va=mesh.uv[v];Vec2 replacement{float(ua.x*(1-t)+va.x*t),float(ua.y*(1-t)+va.y*t)};
                    Vec2 old[3],now[3];for(int j=0;j<3;++j){old[j]=mesh.uv[a[j]];now[j]=(a[j]==u||a[j]==v)?replacement:old[j];}
                    auto area=[](Vec2* t){return (double(t[1].x)-t[0].x)*(double(t[2].y)-t[0].y)-(double(t[1].y)-t[0].y)*(double(t[2].x)-t[0].x);};
                    double before=area(old),after=area(now);if(std::abs(before)>1e-20&&before*after<=0){valid=false;bad_uv=true;}
                }
            }
            if(stats){stats->geometry_rejections+=bad_geometry;stats->uv_rejections+=bad_uv;}
            if(!valid||!removed||remaining<target+removed)continue;
            if(stats)++stats->collapsed;
            if(settings.output==OutputMode::Rebuild) {
                auto edge=p[v]-p[u];double len2=dot(edge,edge);
                double t=len2?std::clamp(dot(c.point-p[u],edge)/len2,0.0,1.0):.5;
                if(!mesh.normals.empty())mesh.normals[u]=normalized(mesh.normals[u]*(1-t)+mesh.normals[v]*t);
                if(!mesh.uv.empty()) {auto a=mesh.uv[u],z=mesh.uv[v];mesh.uv[u]={float(a.x*(1-t)+z.x*t),float(a.y*(1-t)+z.y*t)};}
                if(!mesh.colors.empty()) {
                    auto a=mesh.colors[u],z=mesh.colors[v];
                    auto channel=[&](uint8_t x,uint8_t y){return uint8_t(std::floor(x*(1-t)+y*t+.5));};
                    mesh.colors[u]={channel(a.r,z.r),channel(a.g,z.g),channel(a.b,z.b),a.a};
                }
                if(!mesh.tangents.empty()) {
                    auto a=mesh.tangents[u],z=mesh.tangents[v];
                    mesh.tangents[u]={float(a.x*(1-t)+z.x*t),float(a.y*(1-t)+z.y*t),float(a.z*(1-t)+z.z*t),a.w};
                }
            }
            p[u]=c.point;q[u]+=q[v];map[v]=u;remaining-=removed;++collapsed;
            if(trace)history[v]=u;
            flags[u]|=Used;flags[v]|=Used;
            for(auto vertex:{u,v})for(uint32_t k=offsets[vertex];k<offsets[vertex+1];++k)
                for(int j=0;j<3;++j)flags[map[mesh.indices[adj[k]*3+j]]]|=Used;
        }
        if(!collapsed)break;
        size_t dst=0;
        for(size_t f=0;f<mesh.indices.size()/3;++f) {
            uint32_t a=map[mesh.indices[f*3]],b1=map[mesh.indices[f*3+1]],c=map[mesh.indices[f*3+2]];
            if(a==b1||a==c||b1==c)continue;
            mesh.indices[dst*3]=a;mesh.indices[dst*3+1]=b1;mesh.indices[dst*3+2]=c;
            if(trace)trace->faces[dst]=trace->faces[f];
            if(!mesh.materials.empty())mesh.materials[dst]=mesh.materials[f];++dst;
        }
        mesh.indices.resize(dst*3);if(!mesh.materials.empty())mesh.materials.resize(dst);
        if(trace)trace->faces.resize(dst);
    }
    if(trace)for(uint32_t i=0;i<n;++i) {
        auto root=i;while(history[root]!=root){history[root]=history[history[root]];root=history[root];}
        trace->positions[i]={float(double(p[root].x)*scale+b.center.x),float(double(p[root].y)*scale+b.center.y),float(double(p[root].z)*scale+b.center.z)};
    }
    if(std::equal(mesh.indices.begin(),mesh.indices.end(),source.indices.begin(),source.indices.end()))result.shared_vertices=true;
    if(!result.shared_vertices) {
        for(size_t i=0;i<n;++i)mesh.positions[i]={float(double(p[i].x)*scale+b.center.x),float(double(p[i].y)*scale+b.center.y),float(double(p[i].z)*scale+b.center.z)};
        for(auto& t:mesh.tangents){auto a=normalized({t.x,t.y,t.z});t.x=a.x;t.y=a.y;t.z=a.z;}
        compact(mesh);
    } else {
        mesh.positions.clear();mesh.normals.clear();mesh.uv.clear();mesh.colors.clear();mesh.tangents.clear();
    }
    if(stats)stats->final_triangles=uint32_t(mesh.indices.size()/3);
    result.data=std::move(mesh);return result;
}
Lod reduce(MeshView source,const ReduceSettings& settings) {
    if(!settings.coupled_wedges||settings.output==OutputMode::Reuse)return reduce_impl(source,settings,nullptr);
    if(auto e=validate(source);!e.empty())throw std::invalid_argument(e);
    // Work on positional topology while retaining each original attribute wedge.
    // Surviving source faces keep their UVs, normals, colors and material IDs.
    std::vector<PositionEntry> entries;entries.reserve(source.positions.count);
    for(uint32_t i=0;i<source.positions.count;++i){auto p=source.positions[i];entries.push_back({
        {std::bit_cast<uint32_t>(p.x==0?0.f:p.x),std::bit_cast<uint32_t>(p.y==0?0.f:p.y),std::bit_cast<uint32_t>(p.z==0?0.f:p.z)},i});}
    std::sort(entries.begin(),entries.end(),[](auto& a,auto& b){return std::array{a.key.x,a.key.y,a.key.z,a.index}<std::array{b.key.x,b.key.y,b.key.z,b.index};});
    Mesh geometric;std::vector<uint32_t> map(source.positions.count);PositionKey last{};
    for(auto e:entries){if(geometric.positions.empty()||!(e.key==last)){geometric.positions.push_back(source.positions[e.index]);last=e.key;}map[e.index]=uint32_t(geometric.positions.size()-1);}
    // Already welded input takes the ordinary path, preserving its interpolation behavior.
    if(geometric.positions.size()==source.positions.count)return reduce_impl(source,settings,nullptr);
    geometric.indices.reserve(source.indices.size());for(auto i:source.indices)geometric.indices.push_back(map[i]);
    geometric.materials.assign(source.materials.begin(),source.materials.end());geometric.double_sided.assign(source.double_sided.begin(),source.double_sided.end());
    Trace trace;auto simplified=reduce_impl(geometric.view(),settings,&trace);
    if(simplified.shared_vertices) {
        Lod l;l.data.indices.assign(source.indices.begin(),source.indices.end());l.data.materials.assign(source.materials.begin(),source.materials.end());return l;
    }
    Lod result;result.shared_vertices=false;result.data=copy_mesh(source);
    for(size_t i=0;i<source.positions.count;++i)result.data.positions[i]=trace.positions[map[i]];
    result.data.indices.clear();result.data.materials.clear();
    for(auto face:trace.faces) {
        for(int k=0;k<3;++k)result.data.indices.push_back(source.indices[face*3+k]);
        if(!source.materials.empty())result.data.materials.push_back(source.material(face));
    }
    compact(result.data);return result;
}
}
