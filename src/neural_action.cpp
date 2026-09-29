#include "neural_action.hpp"
#include <bit>
#include <numeric>

namespace blitz::neural {
namespace {
auto key(Vec3 p) {return std::array{std::bit_cast<uint32_t>(p.x==0?0.f:p.x),std::bit_cast<uint32_t>(p.y==0?0.f:p.y),std::bit_cast<uint32_t>(p.z==0?0.f:p.z)};}
double area(Vec2 a,Vec2 b,Vec2 c) {return (double(b.x)-a.x)*(double(c.y)-a.y)-(double(b.y)-a.y)*(double(c.x)-a.x);}
template<class T> void unique(std::vector<T>& a) {std::sort(a.begin(),a.end());a.erase(std::unique(a.begin(),a.end()),a.end());}
}
ActionState::ActionState(MeshView m):input_(m) {
    if(auto error=validate(m);!error.empty())throw std::invalid_argument(error);
    diameter_=std::max(1e-20,bounds(m).diameter());
    current_.data.indices.assign(m.indices.begin(),m.indices.end());current_.data.materials.assign(m.materials.begin(),m.materials.end());
    std::vector<uint32_t> order(m.positions.count);std::iota(order.begin(),order.end(),0);
    std::sort(order.begin(),order.end(),[&](auto a,auto b){return key(m.positions[a])!=key(m.positions[b])?key(m.positions[a])<key(m.positions[b]):a<b;});
    geometry_.resize(order.size());uint32_t root=order[0];
    for(auto i:order){if(key(m.positions[i])!=key(m.positions[root]))root=i;geometry_[i]=root;}
    rebuild();
}
void ActionState::rebuild() {
    auto m=view();const auto n=geometry_.size();graph_=graph(m);
    offsets_.assign(n+1,0);face_offsets_.assign(n+1,0);boundary_.assign(n,0);invalid_.assign(n,0);edges_.clear();
    std::vector<uint64_t> pairs;pairs.reserve(m.indices.size());
    for(uint32_t f=0;f<m.triangles();++f){uint32_t v[3];for(unsigned j=0;j<3;++j)v[j]=geometry_[m.indices[f*3+j]];
        for(unsigned j=0;j<3;++j){auto a=v[j],b=v[(j+1)%3];++face_offsets_[a+1];
            if(a==b){for(auto i:v)invalid_[i]=1;continue;}if(a>b)std::swap(a,b);pairs.push_back(uint64_t(a)<<32|b);}}
    std::sort(pairs.begin(),pairs.end());
    for(size_t i=0;i<pairs.size();){size_t end=i+1;while(end<pairs.size()&&pairs[end]==pairs[i])++end;
        auto a=uint32_t(pairs[i]>>32),b=uint32_t(pairs[i]);auto count=end-i;
        if(count==1)boundary_[a]=boundary_[b]=1;if(count>2)invalid_[a]=invalid_[b]=1;
        edges_.push_back({a,b,uint8_t(std::min<size_t>(3,count))});++offsets_[a+1];++offsets_[b+1];i=end;}
    if(edges_.size()>UINT32_MAX/2)throw std::length_error("action adjacency exceeds u32");
    std::partial_sum(offsets_.begin(),offsets_.end(),offsets_.begin());std::partial_sum(face_offsets_.begin(),face_offsets_.end(),face_offsets_.begin());
    neighbors_.resize(edges_.size()*2);faces_.resize(m.indices.size());auto next=offsets_,face_next=face_offsets_;
    for(auto e:edges_){neighbors_[next[e.a]++]=e.b;neighbors_[next[e.b]++]=e.a;}
    for(uint32_t f=0;f<m.triangles();++f)for(unsigned j=0;j<3;++j)faces_[face_next[geometry_[m.indices[f*3+j]]]++]=f;
    for(uint32_t i=0;i<n;++i)std::sort(neighbors_.begin()+offsets_[i],neighbors_.begin()+offsets_[i+1]);
    // Coincident disconnected layers and bow-tie fans are ambiguous geometric
    // vertices, even when every edge has at most two incident faces.
    std::vector<std::pair<uint32_t,uint32_t>> links;std::vector<uint32_t> reached;
    for(uint32_t u=0;u<n;++u){if(face_offsets_[u]==face_offsets_[u+1]||invalid_[u])continue;links.clear();
        for(auto k=face_offsets_[u];k<face_offsets_[u+1];++k){auto f=faces_[k];uint32_t other[2];unsigned count=0;
            for(unsigned j=0;j<3;++j){auto v=geometry_[m.indices[f*3+j]];if(v!=u&&count<2)other[count++]=v;}
            if(count!=2){invalid_[u]=1;break;}links.emplace_back(other[0],other[1]);}
        if(invalid_[u])continue;reached.assign(1,links[0].first);
        for(size_t i=0;i<reached.size();++i)for(auto [a,b]:links){uint32_t v=a==reached[i]?b:b==reached[i]?a:UINT32_MAX;if(v!=UINT32_MAX&&std::find(reached.begin(),reached.end(),v)==reached.end())reached.push_back(v);}
        if(reached.size()!=offsets_[u+1]-offsets_[u])invalid_[u]=1;
    }
}
bool ActionState::mapping(Action a,std::vector<std::pair<uint32_t,uint32_t>>& remap) const {
    remap.clear();const auto n=geometry_.size();auto m=view();
    if(a.revision!=revision_||a.from>=n||a.to>=n||a.from==a.to||invalid_[a.from]||invalid_[a.to])return false;
    auto begin=neighbors_.begin()+offsets_[a.from],end=neighbors_.begin()+offsets_[a.from+1];
    if(!std::binary_search(begin,end,a.to))return false;
    std::vector<uint32_t> opposite,common,from_wedges;std::vector<uint16_t> from_materials,to_materials;
    uint32_t edge_faces=0;
    for(auto k=face_offsets_[a.from];k<face_offsets_[a.from+1];++k){auto f=faces_[k];uint32_t u=UINT32_MAX,v=UINT32_MAX,w=UINT32_MAX;
        from_materials.push_back(m.material(f));
        for(unsigned j=0;j<3;++j){auto id=m.indices[f*3+j],g=geometry_[id];if(g==a.from)u=id;else if(g==a.to)v=id;else w=g;}
        if(u==UINT32_MAX)return false;from_wedges.push_back(u);
        if(v!=UINT32_MAX){++edge_faces;if(w==UINT32_MAX)return false;opposite.push_back(w);remap.emplace_back(u,v);}}
    if(edge_faces<1||edge_faces>2||edge_faces>=m.triangles())return false;
    // Boundary vertices can disappear only along their geometric boundary.
    if(boundary_[a.from]&&(!boundary_[a.to]||edge_faces!=1))return false;
    std::set_intersection(begin,end,neighbors_.begin()+offsets_[a.to],neighbors_.begin()+offsets_[a.to+1],std::back_inserter(common));
    unique(opposite);if(common!=opposite||opposite.size()!=edge_faces)return false;
    unique(remap);unique(from_wedges);
    // Each source chart must have exactly one destination in the same edge fan.
    // Missing or multiple destinations are ambiguous; never weld their UVs/normals.
    if(remap.size()!=from_wedges.size())return false;
    for(size_t i=0;i<remap.size();++i)if(remap[i].first!=from_wedges[i])return false;
    for(auto k=face_offsets_[a.to];k<face_offsets_[a.to+1];++k)to_materials.push_back(m.material(faces_[k]));
    unique(from_materials);unique(to_materials);if(from_materials.size()>1&&from_materials!=to_materials)return false;
    auto map=[&](uint32_t id){for(auto [u,v]:remap)if(id==u)return v;return id;};
    std::vector<std::array<uint32_t,3>> destination_faces;
    for(auto k=face_offsets_[a.to];k<face_offsets_[a.to+1];++k){auto f=faces_[k];std::array<uint32_t,3> ids;
        for(unsigned j=0;j<3;++j)ids[j]=geometry_[m.indices[f*3+j]];std::sort(ids.begin(),ids.end());destination_faces.push_back(ids);}
    for(auto k=face_offsets_[a.from];k<face_offsets_[a.from+1];++k){auto f=faces_[k];uint32_t old[3],next[3];bool removed=false;
        for(unsigned j=0;j<3;++j){old[j]=m.indices[f*3+j];next[j]=map(old[j]);removed|=geometry_[old[j]]==a.to;}
        if(removed)continue;
        std::array<uint32_t,3> ids{geometry_[next[0]],geometry_[next[1]],geometry_[next[2]]};std::sort(ids.begin(),ids.end());
        if(ids[0]==ids[1]||ids[1]==ids[2]||std::find(destination_faces.begin(),destination_faces.end(),ids)!=destination_faces.end())return false;
        destination_faces.push_back(ids);
        auto before=cross(m.positions[old[1]]-m.positions[old[0]],m.positions[old[2]]-m.positions[old[0]]);
        auto after=cross(m.positions[next[1]]-m.positions[next[0]],m.positions[next[2]]-m.positions[next[0]]);
        if(length(after)<=1e-15||dot(before,after)<=.05*length(before)*length(after))return false;
        if(m.uv){double x=area(m.uv[old[0]],m.uv[old[1]],m.uv[old[2]]),y=area(m.uv[next[0]],m.uv[next[1]],m.uv[next[2]]);if(std::abs(x)>1e-20&&x*y<=0)return false;}
    }
    return true;
}
bool ActionState::legal(Action a) const {std::vector<std::pair<uint32_t,uint32_t>> map;return mapping(a,map);}
Lod ActionState::trial(Action a) const {
    std::vector<std::pair<uint32_t,uint32_t>> remap;if(!mapping(a,remap))throw std::invalid_argument("illegal or stale endpoint action");
    auto m=view();Lod candidate;candidate.data.indices.reserve(m.indices.size());candidate.data.materials.reserve(m.materials.size());
    for(uint32_t f=0;f<m.triangles();++f){uint32_t ids[3];bool changed=false;for(unsigned j=0;j<3;++j){ids[j]=m.indices[f*3+j];for(auto [u,v]:remap)if(ids[j]==u){ids[j]=v;changed=true;break;}}
        if(changed&&(geometry_[ids[0]]==geometry_[ids[1]]||geometry_[ids[1]]==geometry_[ids[2]]||geometry_[ids[0]]==geometry_[ids[2]]))continue;
        candidate.data.indices.insert(candidate.data.indices.end(),ids,ids+3);if(!m.materials.empty())candidate.data.materials.push_back(m.material(f));}
    return candidate;
}
void ActionState::commit(Action a) {auto candidate=trial(a);current_=std::move(candidate);++revision_;rebuild();}
double ActionState::teacher_cost(Action a) const {
    if(a.revision!=revision_||a.from>=geometry_.size()||a.to>=geometry_.size())throw std::invalid_argument("stale teacher action");
    auto m=view();double cost=0,scale=diameter_;
    for(auto k=face_offsets_[a.from];k<face_offsets_[a.from+1];++k){auto f=faces_[k];auto p=m.positions[m.indices[f*3]],q=m.positions[m.indices[f*3+1]],r=m.positions[m.indices[f*3+2]];
        auto normal=cross(q-p,r-p);double weight=length(normal);double distance=dot(normalized(normal),m.positions[a.to]-p)/scale;cost+=distance*distance*weight/(scale*scale);}
    return cost;
}
std::vector<ActionRecord> ActionState::actions(const std::array<float,conditions>& c) const {
    for(float x:c)if(!std::isfinite(x))throw std::invalid_argument("nonfinite action condition");
    std::vector<ActionRecord> result;auto m=view();auto scale=diameter_;
    for(auto edge:edges_)for(auto [u,v]:{std::pair{edge.a,edge.b},std::pair{edge.b,edge.a}}){Action a{u,v,revision_};std::vector<std::pair<uint32_t,uint32_t>> map;if(!mapping(a,map))continue;
        ActionRecord r;r.action=a;auto source_id=map.front().first,target_id=map.front().second;
        std::copy_n(graph_.x.data()+size_t(source_id)*features,features,r.x.data());std::copy_n(graph_.x.data()+size_t(target_id)*features,features,r.x.data()+features);
        std::copy(c.begin(),c.end(),r.x.begin()+48);auto d=(m.positions[v]-m.positions[u])*(1/scale);
        auto* x=r.x.data();x[56]=d.x;x[57]=d.y;x[58]=d.z;x[59]=float(length(d));
        for(unsigned j=0;j<3;++j)x[60]+=x[3+j]*x[27+j];
        if(m.uv){x[61]=std::clamp(m.uv[target_id].x-m.uv[source_id].x,-16.f,16.f)/16;x[62]=std::clamp(m.uv[target_id].y-m.uv[source_id].y,-16.f,16.f)/16;}
        x[63]=float(offsets_[u+1]-offsets_[u])/64;x[64]=float(offsets_[v+1]-offsets_[v])/64;x[65]=edge.count/2.f;
        x[66]=float(double(m.triangles())/input_.triangles());x[67]=boundary_[u];x[68]=boundary_[v];x[69]=float(map.size())/8;
        x[70]=float(face_offsets_[u+1]-face_offsets_[u])/64;x[71]=float(face_offsets_[v+1]-face_offsets_[v])/64;
        Vec3 mean{},target_mean{};double minimum=1,maximum=0;
        for(auto k=face_offsets_[u];k<face_offsets_[u+1];++k){auto f=faces_[k];auto p=m.positions[m.indices[f*3]],q=m.positions[m.indices[f*3+1]],s=m.positions[m.indices[f*3+2]];auto normal=normalized(cross(q-p,s-p));mean=mean+normal;}
        for(auto k=face_offsets_[v];k<face_offsets_[v+1];++k){auto f=faces_[k];target_mean=target_mean+normalized(cross(m.positions[m.indices[f*3+1]]-m.positions[m.indices[f*3]],m.positions[m.indices[f*3+2]]-m.positions[m.indices[f*3]]));}
        mean=normalized(mean);target_mean=normalized(target_mean);
        for(auto k=face_offsets_[u];k<face_offsets_[u+1];++k){auto f=faces_[k];auto p=m.positions[m.indices[f*3]],q=m.positions[m.indices[f*3+1]],s=m.positions[m.indices[f*3+2]];minimum=std::min(minimum,dot(mean,normalized(cross(q-p,s-p))));maximum=std::max(maximum,length(cross(q-p,s-p))/(scale*scale));}
        x[72]=float(minimum);x[73]=float(dot(mean,target_mean));x[74]=float(maximum);x[75]=float(map.size()>1);x[76]=float(bool(m.materials.size()));
        if(m.colors){auto p=linear_color(m.colors[source_id]),q=linear_color(m.colors[target_id]);x[77]=float(length({p.x-q.x,p.y-q.y,p.z-q.z}));}
        x[78]=graph_.x[size_t(source_id)*features+23];x[79]=graph_.x[size_t(target_id)*features+23];result.push_back(r);
    }
    return result;
}
Lod execute_actions(MeshView input,const std::array<float,conditions>& c,size_t target,uint32_t budget,
    const ActionRanker& rank,const ActionGate& gate,ActionStats* statistics,const std::function<bool()>& cancel) {
    ActionStats local;auto& stats=statistics?*statistics:local;stats={};ActionState state(input);target=std::max<size_t>(1,target);
    while(state.view().triangles()>target&&stats.trials<budget){if(cancel&&cancel())break;auto actions=state.actions(c);if(actions.empty())break;
        auto scores=rank(state,actions);stats.ranked+=actions.size();if(scores.size()!=actions.size())throw std::invalid_argument("action rank shape mismatch");
        for(auto x:scores)if(!std::isfinite(x))throw std::invalid_argument("nonfinite action rank");
        std::vector<uint32_t> order(actions.size());std::iota(order.begin(),order.end(),0);std::stable_sort(order.begin(),order.end(),[&](auto a,auto b){return scores[a]>scores[b];});bool accepted=false;
        for(auto i:order){if(stats.trials==budget||(cancel&&cancel()))break;auto candidate=state.trial(actions[i].action);
            if(candidate.data.indices.size()/3<target)continue;++stats.trials;
            if(gate(candidate.view(input))){state.commit(actions[i].action);++stats.accepted;accepted=true;break;}++stats.rejected;}
        if(!accepted)break;
    }
    return state.lod();
}
}
