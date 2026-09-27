#include "blitz/remesher.hpp"
#include <memory>
#include <stdexcept>
namespace blitz {
double Curve::at(double t) const {
    if(points.empty())throw std::invalid_argument("empty curve");
    if(t<=points.front().x)return points.front().y;
    for(size_t i=1;i<points.size();++i)if(t<=points[i].x) {
        double u=(t-points[i-1].x)/(points[i].x-points[i-1].x);
        return points[i-1].y*(1-u)+points[i].y*u;
    }
    return points.back().y;
}
std::string validate(const Settings& s) {
    if(s.levels<2||s.levels>32)return "level count must be 2..32";
    if(unsigned(s.output)>1||unsigned(s.chain)>2||unsigned(s.objective)>2||unsigned(s.profile)>2)return "unknown mode";
    for(double v:{s.pixels_per_meter,s.meters_per_unit,s.last_pixels})
        if(!std::isfinite(v)||v<=0)return "screen scale must be positive and finite";
    for(auto v:{s.base_pixels,s.max_lod0_delta_px})if(v&&(!std::isfinite(*v)||*v<=0))return "invalid optional pixel limit";
    for(auto c:{&s.transition,&s.normal_importance,&s.attribute_importance}) {
        if(c->points.empty()||c->points.front().x!=0||c->points.back().x!=1)return "curves must span [0,1]";
        double last=-1;
        for(auto p:c->points) {
            if(!std::isfinite(p.x)||!std::isfinite(p.y)||p.x<=last||p.x>1||p.y<0||p.y>1e6)return "curve points must be ordered, finite and in 0..1e6";
            last=p.x;
        }
    }
    for(double v:{s.weights.normal,s.weights.color,s.weights.material})if(!std::isfinite(v)||v<0||v>1e6)return "appearance weights must be in 0..1e6";
    if(!s.candidate_budget||!s.beam_width||s.beam_width>32)return "invalid work budget or beam width";
    for(auto n:{s.search_supersample,s.audit_supersample,s.max_supersample})
        if(!n||n>32||(n&(n-1)))return "supersampling must be a power of two in 1..32";
    if(s.max_supersample<std::max(s.search_supersample,s.audit_supersample))return "maximum supersampling is too small";
    if(!s.search_views.orthographic&&!s.search_views.perspective)return "empty search cameras";
    if(!s.audit_views.orthographic&&!s.audit_views.perspective)return "empty audit cameras";
    return {};
}
std::vector<ScheduleEntry> schedule(const Bounds& b,const Settings& s) {
    if(auto e=validate(s);!e.empty())throw std::invalid_argument(e);
    double first=s.base_pixels.value_or(b.diameter()*s.meters_per_unit*s.pixels_per_meter);
    if(!std::isfinite(first)||first<s.last_pixels||first>16384)throw std::invalid_argument("base pixels must be >= last pixels and <=16384");
    std::vector<ScheduleEntry> out(s.levels);out[0].pixels=first;
    double cumulative=0;
    for(unsigned i=1;i<s.levels;++i) {
        double pixels=first*std::pow(s.last_pixels/first,double(i)/(s.levels-1));
        double delta=s.transition.at(s.levels==2?0:double(i-1)/(s.levels-2));
        cumulative=cumulative*pixels/out[i-1].pixels+delta;
        out[i]={pixels,delta,s.max_lod0_delta_px?std::min(cumulative,*s.max_lod0_delta_px):cumulative};
    }
    return out;
}
namespace {
Lod unchanged(MeshView source,ScheduleEntry step) {
    Lod l;l.schedule=step;l.data.indices.assign(source.indices.begin(),source.indices.end());
    l.data.materials.assign(source.materials.begin(),source.materials.end());return l;
}
struct Node {
    std::shared_ptr<Node> parent;
    Lod lod;
    uint64_t triangles{};
};
template<class T> bool equal_vector(const std::vector<T>& a,const std::vector<T>& b) {
    return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0);
}
bool same_lod(const Lod& a,const Lod& b) {
    return a.shared_vertices==b.shared_vertices&&a.data.indices==b.data.indices&&a.data.materials==b.data.materials
      &&equal_vector(a.data.positions,b.data.positions)&&equal_vector(a.data.normals,b.data.normals)
      &&equal_vector(a.data.colors,b.data.colors)&&equal_vector(a.data.uv,b.data.uv)&&equal_vector(a.data.tangents,b.data.tangents);
}
EvalSettings eval_config(const Settings& s,ScheduleEntry step,unsigned level,bool audit,double limit) {
    EvalSettings e;e.profile=s.profile;e.weights=s.weights;e.screen_size=step.pixels;e.limit=limit;
    double t=s.levels==2?0:double(level-1)/(s.levels-2);
    e.weights.normal*=s.normal_importance.at(t);
    e.weights.color*=s.attribute_importance.at(t);e.weights.material*=s.attribute_importance.at(t);
    e.views=audit?s.audit_views:s.search_views;e.supersample=audit?s.audit_supersample:s.search_supersample;
    e.max_supersample=s.max_supersample;e.cancelled=s.cancelled;e.force_scalar=s.force_scalar;return e;
}
}
Result generate(MeshView source,const Settings& s,const Proposer& proposer) {
    if(auto e=validate(source);!e.empty())throw std::invalid_argument(e);
    Result result;result.source=source;result.reference_bounds=bounds(source);
    auto steps=schedule(result.reference_bounds,s);
    // An exact source chain is a valid incumbent even if cancellation precedes the first audit.
    for(auto step:steps)result.lods.push_back(unchanged(source,step));
    auto root=std::make_shared<Node>();root->lod=unchanged(source,steps[0]);
    auto source_path=root;
    std::vector<std::shared_ptr<Node>> beam{root};
    auto cancelled=[&]{return s.cancelled&&s.cancelled();};
    for(unsigned level=1;level<s.levels;++level) {
        if(cancelled()){result.status=Status::Cancelled;return result;}
        std::vector<std::shared_ptr<Node>> next;
        auto search_source=eval_config(s,steps[level],level,false,steps[level].source);
        auto search_adj=eval_config(s,steps[level],level,false,steps[level].transition);
        auto audit_source=eval_config(s,steps[level],level,true,steps[level].source);
        auto audit_adj=eval_config(s,steps[level],level,true,steps[level].transition);
        auto accepted=[&](const Measurement& m,size_t stage) {
            if(m.resource_limited)result.status=Status::BudgetLimited;
            if(m.passed)return true;
            ++result.rejected_gates[stage];
            if(m.error>=result.worst_rejected[stage].error)result.worst_rejected[stage]=m;
            return false;
        };
        auto offer=[&](Lod candidate,const std::shared_ptr<Node>& parent) {
            auto view=candidate.view(source),previous=parent->lod.view(source);
            if(!validate(view).empty())return;
            if(view.triangles()>previous.triangles())return;
            if(!accepted(evaluate(source,view,result.reference_bounds,search_source),0))return;
            if(!accepted(evaluate(previous,view,result.reference_bounds,search_adj),1))return;
            candidate.source_error=evaluate(source,view,result.reference_bounds,audit_source);
            if(!accepted(candidate.source_error,2))return;
            candidate.adjacent=evaluate(previous,view,result.reference_bounds,audit_adj);
            if(!accepted(candidate.adjacent,3))return;
            candidate.schedule=steps[level];
            auto node=std::make_shared<Node>();node->parent=parent;node->triangles=parent->triangles+view.triangles();
            node->lod=std::move(candidate);next.push_back(std::move(node));
        };
        // Retaining each incumbent is important: camera-space errors need not decrease monotonically.
        for(auto& parent:beam)offer(parent->lod,parent);
        unsigned origins=s.chain==ChainMode::Hybrid?2:1;
        unsigned slots=unsigned(beam.size())*origins;
        unsigned rounds=std::max(1u,unsigned(s.candidate_budget)/slots);
        unsigned proposals=0;
        for(unsigned r=0;r<rounds&&proposals<s.candidate_budget;++r) {
            for(auto& parent:beam)for(unsigned o=0;o<origins&&proposals<s.candidate_budget;++o) {
                if(cancelled()){result.status=Status::Cancelled;return result;}
                bool direct=s.chain==ChainMode::Direct||(s.chain==ChainMode::Hybrid&&o==0);
                auto input=direct?source:parent->lod.view(source);
                // Logarithmic ladder spends proposals on both aggressive reductions and near-incumbents.
                double fraction=rounds==1?.5:.02*std::pow(45.0,double(r)/(rounds-1));
                size_t target=std::max<size_t>(1,size_t(parent->lod.view(source).triangles()*fraction));
                ReduceSettings rs;rs.output=s.output;rs.objective=s.objective;rs.target_triangles=target;
                rs.normal_weight=search_source.weights.normal;rs.cancelled=s.cancelled;
                rs.prune=s.prune&&r%2==0;
                rs.coupled_wedges=s.coupled_wedges;
                auto candidate=proposer?proposer(input,rs):reduce(input,rs);
                ++proposals;++result.candidate_evaluations;
                offer(std::move(candidate),parent);
            }
        }
        if(cancelled()){result.status=Status::Cancelled;return result;}
        // Always keep an exact source path; it also handles zero-error and unusual nonmonotonic views.
        auto fallback=std::make_shared<Node>();fallback->parent=source_path;fallback->lod=unchanged(source,steps[level]);
        fallback->triangles=uint64_t(level)*source.triangles();source_path=fallback;next.push_back(fallback);
        std::stable_sort(next.begin(),next.end(),[](auto& a,auto& b){return a->triangles<b->triangles;});
        std::vector<std::shared_ptr<Node>> distinct;
        for(auto& node:next)if(std::none_of(distinct.begin(),distinct.end(),[&](auto& n){return same_lod(node->lod,n->lod);}))distinct.push_back(node);
        next=std::move(distinct);
        if(next.empty()) {result.status=Status::BudgetLimited;return result;}
        if(next.size()>s.beam_width)next.resize(s.beam_width);
        if(std::none_of(next.begin(),next.end(),[&](auto& n){return same_lod(n->lod,source_path->lod);}))next.push_back(source_path);
        beam=std::move(next);
    }
    auto best=*std::min_element(beam.begin(),beam.end(),[](auto& a,auto& b){return a->triangles<b->triangles;});
    for(size_t i=result.lods.size();i-->0;) {result.lods[i]=std::move(best->lod);best=best->parent;}
    return result;
}
}
