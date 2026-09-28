#include "blitz/remesher.hpp"
#include "timing.hpp"
#include "components.hpp"
#include <chrono>
#include <memory>
#include <stdexcept>
namespace blitz {
std::vector<uint8_t> runtime_levels(const Result& r) {
    if(r.lods.size()>32)throw std::invalid_argument("runtime chain exceeds 32 scheduled levels");
    std::vector<uint8_t> levels;
    for(size_t i=0;i<r.lods.size();++i)
        if(!i||!same_mesh_data(r.lods[i-1].view(r.source),r.lods[i].view(r.source)))levels.push_back(uint8_t(i));
    return levels;
}
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
    if((s.research.output&&unsigned(*s.research.output)>1)||unsigned(s.research.chain)>2||unsigned(s.objective)>3||unsigned(s.profile)>2)return "unknown mode";
    if(s.triangle_overhead_bps>10000)return "triangle overhead must be 0..10000 basis points";
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
    if(!std::isfinite(s.research.boundary_weight)||s.research.boundary_weight<0||s.research.boundary_weight>1e6)return "invalid boundary weight";
    if(s.research.component_candidates&&s.profile!=Profile::Coverage)return "component proposals require coverage profile";
    if(s.research.independent_seams&&s.profile!=Profile::Coverage)return "independent seam proposals require coverage profile";
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
    StorageStats storage;
};
ChainCost cost(const std::shared_ptr<Node>& node) {
    ChainCost c;c.storage=node->storage;
    for(auto p=node;p;p=p->parent)c.triangles.push_back(uint32_t(p->lod.data.indices.size()/3));
    std::reverse(c.triangles.begin(),c.triangles.end());return c;
}
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
    e.max_supersample=s.max_supersample;e.cancelled=s.cancelled;e.force_scalar=s.force_scalar;e.performance=s.performance;return e;
}
}
Result generate(MeshView source,const Settings& s,const Proposer& proposer) {
    if(s.performance)*s.performance={};
    if(auto e=validate(source);!e.empty())throw std::invalid_argument(e);
    Result result;result.source=source;result.reference_bounds=bounds(source);result.triangle_overhead_bps=s.triangle_overhead_bps;
    auto steps=schedule(result.reference_bounds,s);
    // An exact source chain is a valid incumbent even if cancellation precedes the first audit.
    for(auto step:steps)result.lods.push_back(unchanged(source,step));
    result.candidates.push_back({std::vector<uint32_t>(steps.size(),uint32_t(source.triangles())),storage_stats(result)});
    auto root=std::make_shared<Node>();root->lod=unchanged(source,steps[0]);
    root->storage={vertex_bytes(source),0,uint64_t(source.indices.size())*4};
    auto source_path=root;
    std::vector<std::shared_ptr<Node>> beam{root},finalists;
    const bool automatic=!s.research.output;
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
        struct Seen {const Node* parent;Lod lod;uint8_t gate;};std::vector<Seen> seen;size_t seen_bytes=0;
        auto offer=[&](Lod candidate,const std::shared_ptr<Node>& parent)->uint8_t {
            auto view=candidate.view(source),previous=parent->lod.view(source);
            if(!validate(view).empty())return 5;
            if(view.triangles()>previous.triangles())return 6;
            bool cached=false;uint8_t gate=0;
            if(automatic||s.research.adaptive_targets) {
                for(auto& item:seen)if((item.parent==parent.get()||(automatic&&same_mesh_data(item.parent->lod.view(source),previous)))&&same_mesh_data(item.lod.view(source),view)) {
                    ++result.duplicate_proposals;
                    if(item.parent==parent.get()&&same_lod(item.lod,candidate))return 7;
                    candidate.source_error=item.lod.source_error;candidate.adjacent=item.lod.adjacent;
                    gate=item.gate;cached=true;break;
                }
            }
            if(!cached) {
                if(!accepted(evaluate(source,view,result.reference_bounds,search_source),0))gate=1;
                else if(!accepted(evaluate(previous,view,result.reference_bounds,search_adj),1))gate=2;
                else {
                    candidate.source_error=evaluate(source,view,result.reference_bounds,audit_source);
                    if(!accepted(candidate.source_error,2))gate=3;
                    else {candidate.adjacent=evaluate(previous,view,result.reference_bounds,audit_adj);if(!accepted(candidate.adjacent,3))gate=4;}
                }
                if(automatic||s.research.adaptive_targets) {
                    size_t bytes=view.indices.size()*4+view.materials.size()*2+(candidate.shared_vertices?0:vertex_bytes(view));
                    if(bytes<=64u*1024u*1024u-seen_bytes){seen.push_back({parent.get(),candidate,gate});seen_bytes+=bytes;}
                }
            }
            if(gate)return gate;
            candidate.schedule=steps[level];
            auto node=std::make_shared<Node>();node->parent=parent;node->triangles=parent->triangles+view.triangles();
            node->storage=parent->storage;
            if(!same_mesh_data(view,previous)) {
                node->storage.index_bytes+=uint64_t(view.indices.size())*4;
                if(!candidate.shared_vertices)node->storage.added_vertex_bytes+=vertex_bytes(view);
            }
            node->lod=std::move(candidate);next.push_back(std::move(node));
            return 0;
        };
        // Retaining each incumbent is important: camera-space errors need not decrease monotonically.
        for(auto& parent:beam)offer(parent->lod,parent);
        struct Slot { std::shared_ptr<Node> parent; bool direct;OutputMode output;std::vector<std::pair<size_t,bool>> trials; };
        std::vector<Slot> slots;
        struct Trials {const Node* parent;std::vector<std::pair<size_t,bool>> values;};std::vector<Trials> parent_trials;
        // Strategy-major traversal gives both placements/origins a turn even at small budgets.
        for(auto output:{OutputMode::Rebuild,OutputMode::Reuse}) {
            if(!automatic&&output!=*s.research.output)continue;
            for(bool direct:{true,false})for(auto& parent:beam) {
                if(automatic&&beam.size()>1&&parent==source_path)continue;
                if(direct&&s.research.chain==ChainMode::Progressive)continue;
                if(!direct&&(s.research.chain==ChainMode::Direct ||
                    (s.research.chain==ChainMode::Hybrid&&same_mesh_data(source,parent->lod.view(source)))))continue;
                if(automatic&&std::any_of(slots.begin(),slots.end(),[&](auto& slot){return slot.direct==direct&&slot.output==output&&same_mesh_data(slot.parent->lod.view(source),parent->lod.view(source));}))continue;
                slots.push_back({parent,direct,output,{}});
            }
        }
        // Alternate placement priority across levels; work remains deterministic.
        if(automatic&&level%2==0)std::rotate(slots.begin(),slots.begin()+slots.size()/2,slots.end());
        const unsigned rounds=std::max(1u,unsigned(s.candidate_budget)/unsigned(slots.size()));
        struct Components {MeshView input;detail::ComponentOrder order;};std::vector<Components> components;
        unsigned proposals=0;
        for(unsigned r=0;proposals<s.candidate_budget;++r) {
            for(auto& slot:slots) {
                if(proposals==s.candidate_budget)break;
                if(cancelled()){result.status=Status::Cancelled;return result;}
                auto& parent=slot.parent;
                auto input=slot.direct?source:parent->lod.view(source);
                auto found_trials=std::find_if(parent_trials.begin(),parent_trials.end(),[&](auto& t){return t.parent==parent.get();});
                if(found_trials==parent_trials.end()){parent_trials.push_back({parent.get(),{}});found_trials=std::prev(parent_trials.end());}
                auto& trials=automatic?found_trials->values:slot.trials;
                // Logarithmic ladder spends proposals on both aggressive reductions and near-incumbents.
                // A partial final round probes a deeper reduction instead of discarding the remaining budget.
                double fraction=r==rounds?.1:rounds==1?.5:.02*std::pow(45.0,double(r)/(rounds-1));
                size_t target=std::max<size_t>(1,size_t(parent->lod.view(source).triangles()*fraction));
                if(s.research.adaptive_targets) {
                    size_t high=parent->lod.view(source).triangles(),low=0;
                    for(auto [n,ok]:trials)if(ok)high=std::min(high,n);
                    for(auto [n,ok]:trials)if(!ok&&n<high)low=std::max(low,n);
                    target=std::max<size_t>(1,(low+high)/2);
                    // Retain broad exploration; a rejected target is not a proof
                    // that every smaller target is invalid.
                    if((automatic?trials.size():r)%4==3)target=std::max<size_t>(1,size_t(parent->lod.view(source).triangles()*.02));
                    if(std::any_of(trials.begin(),trials.end(),[&](auto x){return x.first==target;}))
                        target=std::max<size_t>(1,size_t(parent->lod.view(source).triangles()*fraction));
                    // Test a newly accepted count with the other placement too:
                    // this can keep the same triangle contract without buying a vertex buffer.
                    if(automatic&&slot.direct&&slot.output==OutputMode::Reuse&&high<parent->lod.view(source).triangles()&&
                       std::none_of(slot.trials.begin(),slot.trials.end(),[&](auto x){return x.first==high;}))target=high;
                }
                ReduceSettings rs;rs.output=slot.output;rs.objective=s.objective;rs.target_triangles=target;
                rs.normal_weight=search_source.weights.normal;rs.cancelled=s.cancelled;
                rs.prune=s.prune&&r%2==0;
                rs.coupled_wedges=s.coupled_wedges;
                rs.boundary_weight=s.research.boundary_weight;rs.boundary_placement=s.research.boundary_placement;
                rs.independent_seams=s.research.independent_seams;
                ReductionStats stats;rs.statistics=(s.performance||s.research.trace)?&stats:nullptr;
                ProposalTrace trace;trace.level=uint8_t(level);trace.origin=!slot.direct;trace.strategy=rs.output==OutputMode::Reuse?1:0;
                trace.input_triangles=uint32_t(input.triangles());trace.parent_triangles=uint32_t(parent->lod.view(source).triangles());trace.requested=uint32_t(target);
                if(!automatic&&!proposer&&*s.research.output==OutputMode::Rebuild&&rs.boundary_placement&&proposals%3==1){rs.output=OutputMode::Reuse;trace.strategy=1;}
                auto begin=std::chrono::steady_clock::now();
                Lod candidate;
                {detail::ScopedTime timer(s.performance?&s.performance->reduction_ns:nullptr);
                 if(!proposer&&s.research.component_candidates&&steps[level].pixels<=128&&proposals%3==2) {
                     trace.strategy=2;auto found=std::find_if(components.begin(),components.end(),[&](auto& c){return same_mesh_data(c.input,input);});
                     if(found==components.end()) {components.push_back({input,detail::component_order(input,result.reference_bounds,search_source)});found=std::prev(components.end());++result.component_builds;}
                     if(found->order.available)candidate=found->order.select(input,target);
                     else {++result.component_unavailable;trace.gate=8;candidate=reduce(input,rs);}
                 }else candidate=proposer?proposer(input,rs):reduce(input,rs);}
                if(s.performance) {
                    s.performance->solve_attempts+=stats.solve_attempts;s.performance->singular_solves+=stats.singular_solves;
                    s.performance->nonfinite_solves+=stats.nonfinite_solves;s.performance->position_fallbacks+=stats.position_fallbacks;
                    s.performance->nonfinite_costs+=stats.nonfinite_costs;
                }
                ++proposals;++result.candidate_evaluations;
                trace.achieved=uint32_t(candidate.data.indices.size()/3);
                trace.attempts=stats.attempts;trace.collapsed=stats.collapsed;trace.geometry_rejections=stats.geometry_rejections;
                trace.uv_rejections=stats.uv_rejections;trace.link_rejections=stats.link_rejections;
                if(candidate.shared_vertices&&!slot.direct&&!parent->lod.shared_vertices) {
                    // A reducer borrows from its actual input. An owned previous
                    // LOD has different vertex IDs/storage from the original source.
                    auto local=candidate.view(input);
                    if(same_mesh_data(local,input)) {trace.gate=7;++result.duplicate_proposals;trace.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();if(s.research.trace)result.proposals.push_back(trace);continue;}
                    candidate.data=copy_mesh(local);compact(candidate.data);candidate.shared_vertices=false;
                }
                auto gate=offer(std::move(candidate),parent);
                trials.push_back({target,gate==0});
                if(automatic)slot.trials.push_back({target,gate==0});
                if(trace.gate!=8)trace.gate=gate;
                trace.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
                if(s.research.trace)result.proposals.push_back(trace);
            }
        }
        if(cancelled()){result.status=Status::Cancelled;return result;}
        if(automatic&&!next.empty()) {
            // Reconnect the current triangle leader to cheaper histories. Sharing a
            // prefix does not require its descendants to keep the same reduction path.
            auto leader=*std::min_element(next.begin(),next.end(),[](auto& a,auto& b){return a->triangles<b->triangles;});
            for(auto& parent:beam)if(parent!=leader->parent&&parent!=source_path) {
                if(cancelled()){result.status=Status::Cancelled;return result;}
                ++result.transition_reconnections;offer(leader->lod,parent);
            }
        }
        // Always keep an exact source path; it also handles zero-error and unusual nonmonotonic views.
        auto fallback=std::make_shared<Node>();fallback->parent=source_path;fallback->lod=unchanged(source,steps[level]);
        fallback->triangles=uint64_t(level)*source.triangles();fallback->storage=root->storage;source_path=fallback;next.push_back(fallback);
        std::stable_sort(next.begin(),next.end(),[&](auto& a,auto& b){return a->triangles!=b->triangles?a->triangles<b->triangles:automatic&&a->storage.total()<b->storage.total();});
        std::vector<std::shared_ptr<Node>> distinct;
        for(auto& node:next)if(std::none_of(distinct.begin(),distinct.end(),[&](auto& n){
            return (!automatic||n->parent==node->parent)&&same_lod(node->lod,n->lod);
        }))distinct.push_back(node);
        next=std::move(distinct);
        for(auto& node:next)if(node->parent==source_path->parent&&same_lod(node->lod,source_path->lod)){source_path=node;break;}
        if(level+1==s.levels)finalists=next;
        if(automatic) {
            auto all=next;const auto prefix_reference=cost(next.front()).triangles;
            if(next.size()>(s.beam_width+1u)/2)next.resize((s.beam_width+1u)/2);
            std::stable_sort(all.begin(),all.end(),[](auto& a,auto& b){return a->storage.total()!=b->storage.total()?a->storage.total()<b->storage.total():a->triangles<b->triangles;});
            for(auto& node:all) {
                if(next.size()>=s.beam_width)break;
                auto counts=cost(node).triangles;bool near=true;
                // Fixed search envelope, independent of the user's selection allowance.
                // Histories far outside it cannot buy useful memory savings at the tested 0..10% settings.
                for(size_t i=1;i<counts.size();++i)if(uint64_t(counts[i])*10000>uint64_t(prefix_reference[i])*11000){near=false;break;}
                if(near&&node!=source_path&&std::find(next.begin(),next.end(),node)==next.end())next.push_back(node);
            }
            std::stable_sort(all.begin(),all.end(),[](auto& a,auto& b){return a->triangles<b->triangles;});
            for(auto& node:all){if(next.size()>=s.beam_width)break;if(node!=source_path&&std::find(next.begin(),next.end(),node)==next.end())next.push_back(node);}
        }else if(next.size()>s.beam_width)next.resize(s.beam_width);
        // The exact source history remains available without occupying a beam slot.
        if(std::find(next.begin(),next.end(),source_path)==next.end())next.push_back(source_path);
        beam=std::move(next);
    }
    result.candidates.clear();for(auto& node:finalists)result.candidates.push_back(cost(node));
    result.selection=select_chain(result.candidates,automatic?s.triangle_overhead_bps:0);
    if(!automatic)result.selection={0,0}; // Historical triangle-first research controls.
    auto best=finalists[result.selection.selected];
    for(size_t i=result.lods.size();i-->0;) {result.lods[i]=std::move(best->lod);best=best->parent;}
    return result;
}
}
