#include "neural_internal.hpp"
#include <numeric>
#include <bit>
namespace blitz::neural {
uint8_t bounded_refinement(const EvalSettings& e) {
    if(!std::isfinite(e.screen_size)||e.screen_size<=0||e.screen_size>16384||!e.supersample||e.max_supersample<e.supersample)throw std::invalid_argument("invalid bounded refinement settings");
    uint64_t side=uint64_t(std::ceil(e.screen_size+8));unsigned ss=e.supersample;
    while(ss<e.max_supersample){auto next=std::min<unsigned>(ss*2,e.max_supersample);
        if(side*side*next*next>max_raster_samples)break;ss=next;}
    return uint8_t(ss);
}
namespace {
template<class T> int compare(Stream<T> s,uint32_t a,uint32_t b) {
    if(!s)return 0;auto x=s[a],y=s[b];return std::memcmp(&x,&y,sizeof(T));
}
auto position_key(Vec3 p) {return std::array{std::bit_cast<uint32_t>(p.x==0?0.f:p.x),std::bit_cast<uint32_t>(p.y==0?0.f:p.y),std::bit_cast<uint32_t>(p.z==0?0.f:p.z)};}
double uv_area(Vec2 a,Vec2 b,Vec2 c) {return (double(b.x)-a.x)*(double(c.y)-a.y)-(double(b.y)-a.y)*(double(c.x)-a.x);}
}
Graph graph(MeshView m) {
    if(auto e=validate(m);!e.empty())throw std::invalid_argument(e);
    const uint32_t n=uint32_t(m.positions.count);Graph g;g.flags.resize(n);g.canonical.resize(n);
    std::vector<uint32_t> order(n);std::iota(order.begin(),order.end(),0);
    auto attributes=[&](uint32_t a,uint32_t b) {
        int c=compare(m.normals,a,b);if(!c)c=compare(m.uv,a,b);if(!c)c=compare(m.colors,a,b);if(!c)c=compare(m.tangents,a,b);return c;
    };
    std::sort(order.begin(),order.end(),[&](uint32_t a,uint32_t b){
        auto x=position_key(m.positions[a]),y=position_key(m.positions[b]);
        if(x!=y)return x<y;int c=attributes(a,b);return c?c<0:a<b;
    });
    for(size_t i=0;i<n;) {
        size_t end=i+1;while(end<n&&position_key(m.positions[order[i]])==position_key(m.positions[order[end]]))++end;
        uint32_t representative=order[i];bool seam=false;
        for(size_t j=i;j<end;++j) {
            if(j>i&&attributes(order[j-1],order[j])){representative=order[j];seam=true;}
            g.canonical[order[j]]=representative;
        }
        if(seam)for(size_t j=i;j<end;++j)g.flags[order[j]]|=Locked|Seam;
        i=end;
    }
    std::vector<uint64_t> edges;edges.reserve(m.indices.size());
    std::vector<uint16_t> material(n);std::vector<Vec3> normals(n);
    auto b=bounds(m);double scale=b.diameter();
    auto position=[&](uint32_t i){auto p=m.positions[i];return Vec3{float((double(p.x)-b.center.x)/scale),float((double(p.y)-b.center.y)/scale),float((double(p.z)-b.center.z)/scale)};};
    for(size_t f=0;f<m.triangles();++f) {
        uint32_t id[3];for(unsigned k=0;k<3;++k)id[k]=g.canonical[m.indices[f*3+k]];
        auto vnormal=normalized(cross(position(id[1])-position(id[0]),position(id[2])-position(id[0])));
        for(unsigned k=0;k<3;++k) {
            auto u=id[k],v=id[(k+1)%3];
            if((g.flags[u]&Used)&&material[u]!=m.material(f))g.flags[u]|=Locked|Material;
            material[u]=m.material(f);g.flags[u]|=Used;normals[u]=normals[u]+vnormal;
            if(u!=v){if(u>v)std::swap(u,v);edges.push_back((uint64_t(u)<<32)|v);}
        }
    }
    std::sort(edges.begin(),edges.end());
    std::vector<uint64_t> unique;unique.reserve(edges.size());g.offsets.resize(size_t(n)+1);
    for(size_t i=0;i<edges.size();) {
        size_t j=i+1;while(j<edges.size()&&edges[j]==edges[i])++j;
        auto u=uint32_t(edges[i]>>32),v=uint32_t(edges[i]);
        // Boundary vertices are retained. This also protects chart boundary shape.
        if(j-i==1){g.flags[u]|=Locked|Boundary;g.flags[v]|=Locked|Boundary;}
        if(j-i>2){g.flags[u]|=Locked;g.flags[v]|=Locked;}
        unique.push_back(edges[i]);++g.offsets[u+1];++g.offsets[v+1];i=j;
    }
    if(unique.size()>UINT32_MAX/2)throw std::length_error("neural adjacency exceeds u32 offsets");
    std::partial_sum(g.offsets.begin(),g.offsets.end(),g.offsets.begin());
    g.neighbors.resize(unique.size()*2);auto cursor=g.offsets;
    for(auto edge:unique){auto u=uint32_t(edge>>32),v=uint32_t(edge);g.neighbors[cursor[u]++]=v;g.neighbors[cursor[v]++]=u;}
    g.x.resize(size_t(n)*features);
    for(uint32_t i=0;i<n;++i) {
        std::sort(g.neighbors.begin()+g.offsets[i],g.neighbors.begin()+g.offsets[i+1]);
        auto p=position(i),normal=normalized(m.normals?m.normals[i]:normals[i]);
        auto color=m.colors?linear_color(m.colors[i]):Vec4{1,1,1,1};auto uv=m.uv?m.uv[i]:Vec2{};
        auto tangent=m.tangents?m.tangents[i]:Vec4{};auto direction=normalized({tangent.x,tangent.y,tangent.z});tangent.x=direction.x;tangent.y=direction.y;tangent.z=direction.z;double edge=0;
        for(auto k=g.offsets[i];k<g.offsets[i+1];++k)edge+=length(position(g.neighbors[k])-p);
        uint32_t degree=g.offsets[i+1]-g.offsets[i];edge/=std::max(1u,degree);
        const float x[features]={p.x,p.y,p.z,normal.x,normal.y,normal.z,color.x,color.y,color.z,
            std::clamp(uv.x,-16.f,16.f)/16,std::clamp(uv.y,-16.f,16.f)/16,tangent.x,tangent.y,tangent.z,tangent.w,
            float(bool(m.normals)),float(bool(m.uv)),float(bool(m.colors)),float(bool(m.tangents)),
            float(bool(g.flags[i]&Seam)),float(bool(g.flags[i]&Boundary)),float(bool(g.flags[i]&Material)),
            float(std::log2(1+degree)/8),float(std::log2(1+edge*1024)/10)};
        std::copy_n(x,features,g.x.data()+size_t(i)*features);
    }
    return g;
}
Patch patch(const Graph& g,std::span<const uint32_t> core,uint32_t maximum) {
    Patch p;if(core.empty()||core.size()>maximum)throw std::invalid_argument("invalid patch core");
    std::vector<uint32_t> local(g.size(),UINT32_MAX);p.ids.reserve(std::min<size_t>(maximum,core.size()*4));
    auto add=[&](uint32_t id){if(id>=g.size())throw std::invalid_argument("patch vertex outside graph");if(local[id]==UINT32_MAX){
        if(p.ids.size()==maximum)throw std::length_error("three-hop patch exceeds vertex cap");local[id]=uint32_t(p.ids.size());p.ids.push_back(id);}};
    for(auto id:core)add(id);p.core=uint32_t(p.ids.size());size_t start=0,end=p.ids.size();
    for(unsigned hop=0;hop<3;++hop){for(size_t i=start;i<end;++i)for(auto k=g.offsets[p.ids[i]];k<g.offsets[p.ids[i]+1];++k)add(g.neighbors[k]);start=end;end=p.ids.size();}
    p.graph.offsets.push_back(0);
    for(auto id:p.ids) {
        p.graph.flags.push_back(g.flags[id]);p.graph.canonical.push_back(uint32_t(p.graph.canonical.size()));
        p.graph.x.insert(p.graph.x.end(),g.x.begin()+size_t(id)*features,g.x.begin()+size_t(id+1)*features);
        for(auto k=g.offsets[id];k<g.offsets[id+1];++k)if(local[g.neighbors[k]]!=UINT32_MAX)p.graph.neighbors.push_back(local[g.neighbors[k]]);
        p.graph.offsets.push_back(uint32_t(p.graph.neighbors.size()));
    }
    return p;
}
std::array<float,conditions> condition(const EvalSettings& e,double adjacent,double fraction) {
    return {float(std::log2(e.screen_size)/10),float(adjacent/16),float(e.limit/16),
        float(e.profile==Profile::Coverage?0:e.weights.normal/16),float(e.profile==Profile::Attributes?e.weights.color/16:0),
        float(e.profile==Profile::Attributes?e.weights.material/16:0),float(e.max_changed_area),float(fraction)};
}
Lod decode(MeshView m,const Graph& g,const Prediction& prediction,size_t target,OutputMode mode,DecodeStats* stats,const std::function<bool()>& cancel,std::vector<uint32_t>* representatives) {
    if(g.size()!=m.positions.count||prediction.values.size()!=g.size()*outputs)throw std::invalid_argument("prediction shape differs from mesh");
    for(float v:prediction.values)if(!std::isfinite(v))throw std::invalid_argument("nonfinite neural prediction");
    if(stats)*stats={};Lod out;out.data.indices.assign(m.indices.begin(),m.indices.end());out.data.materials.assign(m.materials.begin(),m.materials.end());
    if(representatives){representatives->resize(g.size());std::iota(representatives->begin(),representatives->end(),0);}
    if(target>=m.triangles())return out;target=std::max<size_t>(1,target);
    for(auto& i:out.data.indices)i=g.canonical[i];
    const uint32_t n=uint32_t(g.size());
    auto position=[&](uint32_t i){auto* x=g.x.data()+size_t(i)*features;return Vec3{x[0],x[1],x[2]};};
    std::vector<uint32_t> representative(n),order;std::iota(representative.begin(),representative.end(),0);
    auto higher=[&](uint32_t a,uint32_t b){float x=prediction.values[size_t(a)*outputs],y=prediction.values[size_t(b)*outputs];return x!=y?x>y:a>b;};
    for(uint32_t u=0;u<n;++u)if((g.flags[u]&Used)&&!(g.flags[u]&Locked)) {
        auto p=position(u);auto* pred=prediction.values.data()+size_t(u)*outputs;double best=std::numeric_limits<double>::infinity();
        for(auto k=g.offsets[u];k<g.offsets[u+1];++k){auto v=g.neighbors[k];if(!higher(v,u))continue;
            auto d=position(v)-p;double dx=d.x-pred[1],dy=d.y-pred[2],dz=d.z-pred[3],cost=dx*dx+dy*dy+dz*dz;
            if(cost<best||(cost==best&&higher(v,representative[u]))){best=cost;representative[u]=v;}}
        if(representative[u]!=u)order.push_back(u);
    }
    std::stable_sort(order.begin(),order.end(),[&](uint32_t a,uint32_t b){return higher(b,a);});
    std::vector<uint32_t> parent(n),offsets(n+1),incident,neighbors,opposite;
    std::vector<uint8_t> used(n),marks(n);std::iota(parent.begin(),parent.end(),0);
    auto root=[&](uint32_t v){while(parent[v]!=v){parent[v]=parent[parent[v]];v=parent[v];}return v;};
    // Batched non-overlapping contractions keep each pass linear in mesh storage.
    // Predictions/forest are fixed; rejected contractions cannot rewrite the learned choices.
    for(unsigned pass=0;pass<128&&out.data.indices.size()/3>target;++pass) {
        if(cancel&&cancel())break;std::fill(offsets.begin(),offsets.end(),0);std::fill(used.begin(),used.end(),0);
        for(auto v:out.data.indices)++offsets[v+1];std::partial_sum(offsets.begin(),offsets.end(),offsets.begin());
        incident.resize(out.data.indices.size());auto cursor=offsets;
        for(uint32_t f=0;f<out.data.indices.size()/3;++f)for(unsigned j=0;j<3;++j)incident[cursor[out.data.indices[f*3+j]]++]=f;
        size_t remaining=out.data.indices.size()/3,accepted=0;
        for(auto original:order) {
            auto u=root(original),v=root(representative[original]);
            if(u!=original||u==v||used[u]||used[v]||!offsets[u+1]||offsets[u]==offsets[u+1])continue;
            uint32_t edge_faces=0;neighbors.clear();opposite.clear();
            for(auto k=offsets[u];k<offsets[u+1];++k){auto f=incident[k];bool both=false;
                for(unsigned j=0;j<3;++j)both|=out.data.indices[f*3+j]==v;edge_faces+=both;
                for(unsigned j=0;j<3;++j){auto a=out.data.indices[f*3+j];if(a==u||a==v)continue;marks[a]|=1;if(both)marks[a]|=2;neighbors.push_back(a);}}
            bool valid=edge_faces>=1&&edge_faces<=2&&remaining>=target+edge_faces;
            for(auto k=offsets[v];k<offsets[v+1]&&valid;++k)for(unsigned j=0;j<3;++j){auto a=out.data.indices[incident[k]*3+j];if(a!=u&&a!=v&&(marks[a]&1)&&!(marks[a]&2))valid=false;}
            for(auto a:neighbors)marks[a]=0;
            for(auto k=offsets[u];k<offsets[u+1]&&valid;++k) {
                auto f=incident[k];uint32_t a[3];bool both=false;
                for(unsigned j=0;j<3;++j){a[j]=out.data.indices[f*3+j];both|=a[j]==v;}if(both)continue;
                // Link vertices alone do not detect the tetrahedron case: reject
                // any surviving triangle that would duplicate a face at the target.
                std::array<uint32_t,3> proposed{a[0]==u?v:a[0],a[1]==u?v:a[1],a[2]==u?v:a[2]};std::sort(proposed.begin(),proposed.end());
                for(auto k2=offsets[v];k2<offsets[v+1]&&valid;++k2){auto other=incident[k2];std::array<uint32_t,3> existing{out.data.indices[other*3],out.data.indices[other*3+1],out.data.indices[other*3+2]};std::sort(existing.begin(),existing.end());if(existing==proposed)valid=false;}
                Vec3 before[3],after[3];for(unsigned j=0;j<3;++j){before[j]=position(a[j]);after[j]=position(a[j]==u?v:a[j]);}
                auto oldn=cross(before[1]-before[0],before[2]-before[0]),newn=cross(after[1]-after[0],after[2]-after[0]);
                if(length(newn)<1e-15||dot(oldn,newn)<=.05*length(oldn)*length(newn))valid=false;
                if(m.uv){auto x=uv_area(m.uv[a[0]],m.uv[a[1]],m.uv[a[2]]);for(auto& id:a)if(id==u)id=v;
                    double y=uv_area(m.uv[a[0]],m.uv[a[1]],m.uv[a[2]]);if(std::abs(x)>1e-20&&x*y<=0)valid=false;}
            }
            if(!valid){if(stats)++stats->rejected;continue;}
            parent[u]=v;remaining-=edge_faces;++accepted;if(stats)++stats->accepted;
            for(auto vertex:{u,v})for(auto k=offsets[vertex];k<offsets[vertex+1];++k)for(unsigned j=0;j<3;++j)used[out.data.indices[incident[k]*3+j]]=1;
            if(remaining<=target)break;
        }
        if(!accepted)break;
        size_t dst=0;for(size_t f=0;f<out.data.indices.size()/3;++f){uint32_t a[3];for(unsigned j=0;j<3;++j)a[j]=root(out.data.indices[f*3+j]);
            if(a[0]==a[1]||a[0]==a[2]||a[1]==a[2])continue;for(unsigned j=0;j<3;++j)out.data.indices[dst*3+j]=a[j];
            if(!out.data.materials.empty())out.data.materials[dst]=out.data.materials[f];++dst;}
        out.data.indices.resize(dst*3);if(!out.data.materials.empty())out.data.materials.resize(dst);
    }
    if(representatives)for(uint32_t i=0;i<g.size();++i)(*representatives)[i]=root(g.canonical[i]);
    if(mode==OutputMode::Rebuild){out.data=copy_mesh(out.view(m));compact(out.data);out.shared_vertices=false;}
    return out;
}
}
