#include "blitz/remesher.hpp"
#include "timing.hpp"
#include "components.hpp"
#include "coverage.hpp"
#include "density.hpp"
#include "shared_vertices.hpp"
#include <chrono>
#include <iterator>
#include <memory>
#include <stdexcept>
namespace blitz {
namespace detail { Result improve_chain(Result,const Settings&,const Proposer&,double); }
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
    if(s.max_added_vertex_bytes_bps&&*s.max_added_vertex_bytes_bps==UINT32_MAX)return "added vertex budget must be below UINT32_MAX basis points";
    if(s.research.coverage_cache_mib>256)return "coverage cache must be 0..256 MiB";
    if(unsigned(s.research.appearance_stage)>3)return "appearance stage must be 0..3";
    if(s.research.appearance_stage!=AppearanceStage::Off&&s.profile==Profile::Coverage)return "appearance proposals require an appearance profile";
    if(s.research.graph_passes>3)return "graph passes must be 0..3";
    if(s.research.graph_passes&&(s.research.output||s.research.chain!=ChainMode::Hybrid))return "graph search requires automatic hybrid output";
    if(!std::isfinite(s.max_changed_area)||s.max_changed_area<0||s.max_changed_area>1)
        return "maximum changed area must be in [0,1]";
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
    if(s.research.topology_fallback&&s.objective!=Objective::Quadric)return "topology fallback requires quadric objective";
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
struct SearchPass {
    Result result;
    std::vector<std::shared_ptr<Node>> finalists;
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
    return a.shared_vertices==b.shared_vertices&&a.vertex_pool==b.vertex_pool&&a.source_prefix_vertices==b.source_prefix_vertices&&a.data.indices==b.data.indices&&a.data.materials==b.data.materials
      &&equal_vector(a.data.positions,b.data.positions)&&equal_vector(a.data.normals,b.data.normals)
      &&equal_vector(a.data.colors,b.data.colors)&&equal_vector(a.data.uv,b.data.uv)&&equal_vector(a.data.tangents,b.data.tangents);
}
EvalSettings eval_config(const Settings& s,ScheduleEntry step,unsigned level,bool audit,double limit) {
    EvalSettings e;e.profile=s.profile;e.weights=s.weights;e.screen_size=step.pixels;e.limit=limit;
    e.conservative_screen=!audit&&s.research.conservative_screen;
    e.max_changed_area=audit?s.max_changed_area:1.0;
    double t=s.levels==2?0:double(level-1)/(s.levels-2);
    e.weights.normal*=s.normal_importance.at(t);
    e.weights.color*=s.attribute_importance.at(t);e.weights.material*=s.attribute_importance.at(t);
    e.views=audit?s.audit_views:s.search_views;e.supersample=audit?s.audit_supersample:s.search_supersample;
    e.max_supersample=s.max_supersample;e.cancelled=s.cancelled;e.force_scalar=s.force_scalar;e.performance=s.performance;return e;
}
SearchPass search_pass(MeshView source,const Settings& s,const Proposer& proposer) {
    if(auto e=validate(source);!e.empty())throw std::invalid_argument(e);
    const bool automatic=!s.research.output;
    detail::SourceVertices source_vertices(source,automatic&&s.research.shared_rebuild);
    Result result;result.source=source;result.reference_bounds=bounds(source);result.triangle_overhead_bps=s.triangle_overhead_bps;
    result.audit={s.profile,s.search_views,s.audit_views,s.weights,s.normal_importance,s.attribute_importance,
        s.search_supersample,s.audit_supersample,s.max_supersample};
    result.max_added_vertex_bytes_bps=automatic?s.max_added_vertex_bytes_bps:std::nullopt;
    if(result.max_added_vertex_bytes_bps) {
        auto bytes=vertex_bytes(source),bps=uint64_t(*result.max_added_vertex_bytes_bps);
        auto whole=bytes/10000,part=(bytes%10000)*bps/10000;
        result.added_vertex_budget_bytes=bps&&whole>(UINT64_MAX-part)/bps?UINT64_MAX:whole*bps+part;
    }
    result.chain_objective=result.added_vertex_budget_bytes?ChainObjective::TailFirst:ChainObjective::WholeChain;
    result.max_changed_area=s.max_changed_area;
    auto steps=schedule(result.reference_bounds,s);
    // An exact source chain is a valid incumbent even if cancellation precedes the first audit.
    for(auto step:steps)result.lods.push_back(unchanged(source,step));
    result.candidates.push_back({std::vector<uint32_t>(steps.size(),uint32_t(source.triangles())),storage_stats(result)});
    auto root=std::make_shared<Node>();root->lod=unchanged(source,steps[0]);
    root->storage={vertex_bytes(source),0,uint64_t(source.indices.size())*4};
    auto source_path=root;
    std::vector<std::shared_ptr<Node>> beam{root},finalists;
    auto cancelled=[&]{return result.status==Status::Cancelled||(s.cancelled&&s.cancelled());};
    std::optional<Lod> tail_seed;
    uint64_t tail_reserved=0;
    if(automatic&&result.added_vertex_budget_bytes&&*result.added_vertex_budget_bytes&&!proposer) {
        // Find an audited compact tail first. Earlier levels may use only the
        // remaining bytes; the final level can connect this mesh to any prefix.
        const size_t last=s.levels-1,base=std::max<size_t>(1,source.triangles()/100);
        const std::array<size_t,8> targets{1,std::max<size_t>(1,base/16),std::max<size_t>(1,base/4),
            std::max<size_t>(1,base/2),base,std::min(source.triangles(),base*2),
            std::min(source.triangles(),base*4),std::min(source.triangles(),base*8)};
        auto search=eval_config(s,steps[last],last,false,std::min(steps[last].source,steps[last].transition));
        auto audit=eval_config(s,steps[last],last,true,std::min(steps[last].source,steps[last].transition));
        detail::CoverageCache coverage(s.profile==Profile::Coverage?uint32_t(s.research.coverage_cache_mib)*1024*1024:0,
            result.reference_bounds);
        size_t previous_target=0;detail::DensityTargets density;
        for(auto nominal:targets) {
            auto target=s.research.density_targets?density.next(source,source.triangles(),*result.added_vertex_budget_bytes,nominal):nominal;
            if(cancelled()){result.status=Status::Cancelled;return {std::move(result),{}};}
            if(target==previous_target)continue;
            previous_target=target;
            ReduceSettings rs;rs.output=OutputMode::Rebuild;rs.objective=s.objective;rs.target_triangles=target;
            rs.appearance_stage=s.research.appearance_stage;rs.appearance_weights=search.weights;if(s.profile!=Profile::Attributes)rs.appearance_weights.color=0;rs.screen_size=search.screen_size;
            rs.normal_weight=search.weights.normal;rs.cancelled=s.cancelled;rs.prune=s.prune;
            rs.coupled_wedges=s.coupled_wedges;rs.merge_wedges=s.research.merge_wedges;rs.preserve_positions=s.research.shared_rebuild;rs.boundary_weight=s.research.boundary_weight;
            rs.boundary_placement=s.research.boundary_placement;rs.independent_seams=s.research.independent_seams;
            ReductionStats stats;rs.statistics=(s.performance||s.research.trace)?&stats:nullptr;
            const auto begin=std::chrono::steady_clock::now();
            Lod candidate;
            {detail::ScopedTime timer(s.performance?&s.performance->reduction_ns:nullptr);candidate=reduce(source,rs);}
            if(s.performance) {
                s.performance->solve_attempts+=stats.solve_attempts;s.performance->singular_solves+=stats.singular_solves;
                s.performance->nonfinite_solves+=stats.nonfinite_solves;s.performance->position_fallbacks+=stats.position_fallbacks;
                s.performance->nonfinite_costs+=stats.nonfinite_costs;s.performance->appearance_peak_bytes=std::max(s.performance->appearance_peak_bytes,stats.appearance_bytes);
            }
            ++result.candidate_evaluations;++result.tail_probe_evaluations;
            source_vertices.share(candidate);auto view=candidate.view(source);
            const auto achieved=view.triangles();
            uint8_t gate=0;
            const auto bytes=added_vertex_bytes(candidate,source);
            if(!validate(view).empty())gate=5;
            else if(same_mesh_data(view,source))gate=7;
            else if(bytes>*result.added_vertex_budget_bytes){++result.vertex_budget_rejections;gate=9;}
            else {
                coverage.begin_candidate();
                ++result.audit_evaluations[0];auto measured=coverage.evaluate(source,view,search,0,false);
                if(measured.resource_limited)result.status=Status::BudgetLimited;
                if(measured.cancelled)result.status=Status::Cancelled;
                if(!measured.passed)gate=1;
                else {
                    ++result.audit_evaluations[2];measured=coverage.evaluate(source,view,audit,0,true);
                    if(measured.resource_limited)result.status=Status::BudgetLimited;
                    if(measured.cancelled)result.status=Status::Cancelled;
                    if(!measured.passed)gate=3;
                }
                coverage.begin_candidate();
            }
            if(s.research.density_targets)density.observe(target,achieved,bytes,gate);
            if(!gate&&(!tail_seed||view.triangles()<tail_seed->view(source).triangles()||
                (view.triangles()==tail_seed->view(source).triangles()&&bytes<tail_reserved))) {
                tail_reserved=bytes;tail_seed=std::move(candidate);
            }
            if(s.research.trace) {
                ProposalTrace trace;trace.level=uint8_t(last);trace.origin=2;trace.strategy=0;
                trace.input_triangles=trace.parent_triangles=uint32_t(source.triangles());
                trace.requested=uint32_t(target);trace.achieved=uint32_t(achieved);trace.gate=gate;
                trace.attempts=stats.attempts;trace.collapsed=stats.collapsed;trace.geometry_rejections=stats.geometry_rejections;
                trace.uv_rejections=stats.uv_rejections;trace.link_rejections=stats.link_rejections;
                trace.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
                result.proposals.push_back(trace);
            }
        }
        result.tail_reserved_vertex_bytes=tail_reserved;
    }
    for(unsigned level=1;level<s.levels;++level) {
        if(cancelled()){result.status=Status::Cancelled;return {std::move(result),{}};}
        std::vector<std::shared_ptr<Node>> next;
        auto search_source=eval_config(s,steps[level],level,false,steps[level].source);
        auto search_adj=eval_config(s,steps[level],level,false,steps[level].transition);
        auto audit_source=eval_config(s,steps[level],level,true,steps[level].source);
        auto audit_adj=eval_config(s,steps[level],level,true,steps[level].transition);
        detail::CoverageCache coverage(s.profile==Profile::Coverage?uint32_t(s.research.coverage_cache_mib)*1024*1024:0,
            result.reference_bounds);
        // IDs expire with this level. The source and exact source parents share ID 0.
        std::array<uint8_t,33> reference_ids{};
        for(size_t i=0;i<beam.size();++i)
            reference_ids[i]=same_mesh_data(source,beam[i]->lod.view(source))?0:uint8_t(i+1);
        auto accepted=[&](const Measurement& m,size_t stage) {
            ++result.audit_evaluations[stage];
            if(m.resource_limited)result.status=Status::BudgetLimited;
            if(m.cancelled){result.status=Status::Cancelled;return false;}
            if(m.passed)return true;
            ++result.rejected_gates[stage];
            if(stage>=2&&m.error<=(stage==2?audit_source.limit:audit_adj.limit)
              &&m.changed_area>s.max_changed_area)++result.area_rejected_gates[stage];
            if(m.error>=result.worst_rejected[stage].error)result.worst_rejected[stage]=m;
            return false;
        };
        struct Seen {const Node* parent;Lod lod;uint8_t gate;};std::vector<Seen> seen;size_t seen_bytes=0;
        auto offer=[&](Lod candidate,const std::shared_ptr<Node>& parent)->uint8_t {
            source_vertices.share(candidate);
            auto view=candidate.view(source),previous=parent->lod.view(source);
            if(!validate(view).empty())return 5;
            if(view.triangles()>previous.triangles())return 6;
            const bool changed=!same_mesh_data(view,previous);
            auto added=added_vertex_bytes(candidate,source);
            if(candidate.source_prefix_vertices)for(auto p=parent;p;p=p->parent)
                if(p->lod.source_prefix_vertices&&p->lod.vertex_pool==candidate.vertex_pool){added=0;break;}
            if(result.added_vertex_budget_bytes&&changed&&!candidate.shared_vertices) {
                auto allowed=*result.added_vertex_budget_bytes-(level+1<s.levels?tail_reserved:0);
                if(parent->storage.added_vertex_bytes>allowed||added>allowed-parent->storage.added_vertex_bytes) {
                    ++result.vertex_budget_rejections;return 9;
                }
            }
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
                coverage.begin_candidate();
                const auto parent_index=size_t(std::find(beam.begin(),beam.end(),parent)-beam.begin());
                const auto parent_id=reference_ids[parent_index];
                if(!accepted(coverage.evaluate(source,view,search_source,0,false),0))gate=1;
                else if(!accepted(coverage.evaluate(previous,view,search_adj,parent_id,false),1))gate=2;
                else {
                    candidate.source_error=coverage.evaluate(source,view,audit_source,0,true);
                    if(!accepted(candidate.source_error,2))gate=3;
                    else {candidate.adjacent=coverage.evaluate(previous,view,audit_adj,parent_id,true);if(!accepted(candidate.adjacent,3))gate=4;}
                }
                coverage.begin_candidate();
                if(automatic||s.research.adaptive_targets) {
                    size_t bytes=view.indices.size()*4+view.materials.size()*2+(candidate.shared_vertices?0:vertex_bytes(view));
                    if(bytes<=64u*1024u*1024u-seen_bytes){seen.push_back({parent.get(),candidate,gate});seen_bytes+=bytes;}
                }
            }
            if(gate)return gate;
            candidate.schedule=steps[level];
            auto node=std::make_shared<Node>();node->parent=parent;node->triangles=parent->triangles+view.triangles();
            node->storage=parent->storage;
            if(changed) {
                node->storage.index_bytes+=uint64_t(view.indices.size())*4;
                node->storage.added_vertex_bytes+=added;
            }
            node->lod=std::move(candidate);next.push_back(std::move(node));
            return 0;
        };
        // Retaining each incumbent is important: camera-space errors need not decrease monotonically.
        for(auto& parent:beam)offer(parent->lod,parent);
        struct Slot {
            std::shared_ptr<Node> parent;bool direct;OutputMode output;
            std::vector<std::pair<size_t,bool>> trials,requested; // requested: target and prune flag.
            detail::DensityTargets density;
        };
        std::vector<Slot> slots;
        struct Trials {const Node* parent;std::vector<std::pair<size_t,bool>> values;};std::vector<Trials> parent_trials;
        // Strategy-major traversal gives both placements/origins a turn even at small budgets.
        for(auto output:{OutputMode::Rebuild,OutputMode::Reuse}) {
            if(!automatic&&output!=*s.research.output)continue;
            for(bool direct:{true,false})for(auto& parent:beam) {
                // Under a vertex cap, a direct compact tail may only fit after
                // retaining the source through earlier scheduled levels.
                if(automatic&&!result.added_vertex_budget_bytes&&beam.size()>1&&parent==source_path)continue;
                if(direct&&s.research.chain==ChainMode::Progressive)continue;
                if(!direct&&(s.research.chain==ChainMode::Direct ||
                    (s.research.chain==ChainMode::Hybrid&&same_mesh_data(source,parent->lod.view(source)))))continue;
                if(automatic&&std::any_of(slots.begin(),slots.end(),[&](auto& slot){return slot.direct==direct&&slot.output==output&&same_mesh_data(slot.parent->lod.view(source),parent->lod.view(source));}))continue;
                slots.push_back({parent,direct,output,{},{},{}});
            }
        }
        // Alternate placement priority across levels; work remains deterministic.
        if(automatic&&level%2==0)std::rotate(slots.begin(),slots.begin()+slots.size()/2,slots.end());
        const unsigned rounds=std::max(1u,unsigned(s.candidate_budget)/unsigned(slots.size()));
        struct Components {MeshView input;detail::ComponentOrder order;};std::vector<Components> components;
        unsigned proposals=0;bool topology_fallback_attempted=false;
        for(unsigned r=0;proposals<s.candidate_budget;++r) {
            for(auto& slot:slots) {
                if(proposals==s.candidate_budget)break;
                if(cancelled()){result.status=Status::Cancelled;return {std::move(result),{}};}
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
                if(automatic&&result.added_vertex_budget_bytes&&slot.output==OutputMode::Rebuild&&!s.research.shared_rebuild) {
                    const auto allowed=*result.added_vertex_budget_bytes-(level+1<s.levels?tail_reserved:0);
                    const auto remaining=parent->storage.added_vertex_bytes<=allowed?
                        allowed-parent->storage.added_vertex_bytes:0;
                    const auto stride=vertex_bytes(source)/source.positions.count;
                    if(s.research.density_targets)target=slot.density.next(input,parent->lod.view(source).triangles(),remaining,target);
                    else if(stride&&remaining/stride>=3)target=std::min(target,size_t(remaining/stride/3));
                }
                ReduceSettings rs;rs.output=slot.output;rs.objective=s.objective;rs.target_triangles=target;
                rs.appearance_stage=s.research.appearance_stage;rs.appearance_weights=search_source.weights;if(s.profile!=Profile::Attributes)rs.appearance_weights.color=0;rs.screen_size=search_source.screen_size*bounds(input).diameter()/result.reference_bounds.diameter();
                rs.normal_weight=search_source.weights.normal;rs.cancelled=s.cancelled;
                rs.prune=s.prune&&r%2==0;
                if(result.added_vertex_budget_bytes) {
                    auto request=std::pair{target,rs.prune};
                    if(std::find(slot.requested.begin(),slot.requested.end(),request)!=slot.requested.end()) {
                        ++proposals;continue;
                    }
                    slot.requested.push_back(request);
                }
                rs.coupled_wedges=s.coupled_wedges;rs.merge_wedges=s.research.merge_wedges;rs.preserve_positions=s.research.shared_rebuild;
                rs.boundary_weight=s.research.boundary_weight;rs.boundary_placement=s.research.boundary_placement;
                rs.independent_seams=s.research.independent_seams;
                ReductionStats stats;rs.statistics=(s.performance||s.research.trace||s.research.topology_fallback)?&stats:nullptr;
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
                    s.performance->nonfinite_costs+=stats.nonfinite_costs;s.performance->appearance_peak_bytes=std::max(s.performance->appearance_peak_bytes,stats.appearance_bytes);
                }
                ++proposals;++result.candidate_evaluations;
                trace.achieved=uint32_t(candidate.data.indices.size()/3);
                trace.attempts=stats.attempts;trace.collapsed=stats.collapsed;trace.geometry_rejections=stats.geometry_rejections;
                trace.uv_rejections=stats.uv_rejections;trace.link_rejections=stats.link_rejections;
                auto prepare_borrowed=[&](Lod& lod) {
                    if(!lod.shared_vertices||slot.direct||parent->lod.shared_vertices)return true;
                    // A reducer borrows from its actual input. An owned previous
                    // LOD has different vertex IDs/storage from the original source.
                    auto local=lod.view(input);
                    if(same_mesh_data(local,input)) {++result.duplicate_proposals;return false;}
                    lod.data=copy_mesh(local);compact(lod.data);lod.shared_vertices=false;return true;
                };
                if(prepare_borrowed(candidate)) {
                    const auto emitted_bytes=vertex_bytes(candidate.view(source));
                    auto gate=offer(std::move(candidate),parent);
                    if(s.research.density_targets&&slot.output==OutputMode::Rebuild)slot.density.observe(target,trace.achieved,emitted_bytes,gate);
                    trials.push_back({target,gate==0});
                    if(automatic)slot.trials.push_back({target,gate==0});
                    if(trace.gate!=8)trace.gate=gate;
                }else trace.gate=7;
                trace.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
                if(s.research.trace)result.proposals.push_back(trace);
                // One extra proposal per level, only after the topology-preserving
                // reducer misses its requested target by more than 4x with
                // link-condition rejections. The usual four visual gates decide
                // whether the topology-changing result is usable.
                if(s.research.topology_fallback&&!proposer&&s.objective==Objective::Quadric&&
                   !topology_fallback_attempted&&stats.link_rejections&&uint64_t(trace.achieved)>uint64_t(target)*4) {
                    if(cancelled()){result.status=Status::Cancelled;return {std::move(result),{}};}
                    topology_fallback_attempted=true;
                    auto relaxed=rs;relaxed.objective=Objective::TopologyRelaxed;
                    ReductionStats relaxed_stats;relaxed.statistics=&relaxed_stats;
                    ProposalTrace fallback_trace;fallback_trace.level=uint8_t(level);fallback_trace.origin=trace.origin;
                    fallback_trace.strategy=3;fallback_trace.input_triangles=trace.input_triangles;
                    fallback_trace.parent_triangles=trace.parent_triangles;fallback_trace.requested=trace.requested;
                    auto fallback_begin=std::chrono::steady_clock::now();
                    Lod alternative;
                    {detail::ScopedTime timer(s.performance?&s.performance->reduction_ns:nullptr);
                     alternative=reduce(input,relaxed);}
                    if(s.performance) {
                        s.performance->solve_attempts+=relaxed_stats.solve_attempts;s.performance->singular_solves+=relaxed_stats.singular_solves;
                        s.performance->nonfinite_solves+=relaxed_stats.nonfinite_solves;s.performance->position_fallbacks+=relaxed_stats.position_fallbacks;
                        s.performance->nonfinite_costs+=relaxed_stats.nonfinite_costs;s.performance->appearance_peak_bytes=std::max(s.performance->appearance_peak_bytes,relaxed_stats.appearance_bytes);
                    }
                    ++result.candidate_evaluations;++result.topology_fallback_proposals;
                    fallback_trace.achieved=uint32_t(alternative.data.indices.size()/3);
                    fallback_trace.attempts=relaxed_stats.attempts;fallback_trace.collapsed=relaxed_stats.collapsed;
                    fallback_trace.geometry_rejections=relaxed_stats.geometry_rejections;
                    fallback_trace.uv_rejections=relaxed_stats.uv_rejections;
                    fallback_trace.link_rejections=relaxed_stats.link_rejections;
                    fallback_trace.gate=prepare_borrowed(alternative)?offer(std::move(alternative),parent):7;
                    fallback_trace.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-fallback_begin).count();
                    if(s.research.trace)result.proposals.push_back(fallback_trace);
                }
            }
        }
        if(cancelled()){result.status=Status::Cancelled;return {std::move(result),{}};}
        if(level+1==s.levels&&tail_seed)for(auto& parent:beam)offer(*tail_seed,parent);
        if(automatic&&!next.empty()) {
            // Reconnect the current triangle leader to cheaper histories. Sharing a
            // prefix does not require its descendants to keep the same reduction path.
            auto leader=*std::min_element(next.begin(),next.end(),[](auto& a,auto& b){return a->triangles<b->triangles;});
            for(auto& parent:beam)if(parent!=leader->parent&&parent!=source_path) {
                if(cancelled()){result.status=Status::Cancelled;return {std::move(result),{}};}
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
            std::stable_sort(all.begin(),all.end(),[&](auto& a,auto& b){
                auto ac=result.added_vertex_budget_bytes?a->storage.added_vertex_bytes:a->storage.total();
                auto bc=result.added_vertex_budget_bytes?b->storage.added_vertex_bytes:b->storage.total();
                return ac!=bc?ac<bc:a->triangles!=b->triangles?a->triangles<b->triangles:a->storage.total()<b->storage.total();
            });
            for(auto& node:all) {
                if(next.size()>=s.beam_width)break;
                bool near=true;
                auto counts=cost(node).triangles;
                // Fixed search envelope, independent of the user's selection allowance.
                // Histories far outside it cannot buy useful memory savings at the tested 0..10% settings.
                if(!result.added_vertex_budget_bytes)
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
    return {std::move(result),std::move(finalists)};
}
}
static Result generate_impl(MeshView source,const Settings& s,const Proposer& proposer) {
    if(auto error=validate(s);!error.empty())throw std::invalid_argument(error);
    if(s.research.graph_passes) {
        const auto begin=std::chrono::steady_clock::now();
        auto seed=s;seed.research.graph_passes=0;seed.research.topology_fallback=false;
        auto incumbent=generate_impl(source,seed,proposer);
        return detail::improve_chain(std::move(incumbent),s,proposer,
            std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count());
    }
    if(s.performance)*s.performance={};
    auto first=search_pass(source,s,proposer);
    auto& result=first.result;
    if(first.finalists.empty())return std::move(result);
    const bool automatic=!s.research.output;
    auto baseline=automatic?select_chain(result.candidates,s.triangle_overhead_bps,result.added_vertex_budget_bytes,ChainObjective::TailFirst):ChainSelection{};
    result.selection=baseline;
    auto selected=first.finalists[baseline.selected];
    const auto budget=result.added_vertex_budget_bytes.value_or(0);
    const bool retry=automatic&&s.research.chain==ChainMode::Hybrid&&!s.research.adaptive_targets&&!proposer&&
        result.status==Status::Complete&&budget&&selected->parent&&
        same_mesh_data(selected->lod.view(source),selected->parent->lod.view(source))&&
        selected->storage.added_vertex_bytes<=budget/2;
    if(retry) {
        auto adaptive=s;adaptive.research.adaptive_targets=true;
        auto second=search_pass(source,adaptive,proposer);
        result.adaptive_retry_attempted=true;
        result.adaptive_retry_evaluations=second.result.candidate_evaluations;
        result.candidate_evaluations+=second.result.candidate_evaluations;
        result.duplicate_proposals+=second.result.duplicate_proposals;
        result.component_builds+=second.result.component_builds;
        result.component_unavailable+=second.result.component_unavailable;
        result.topology_fallback_proposals+=second.result.topology_fallback_proposals;
        result.transition_reconnections+=second.result.transition_reconnections;
        result.vertex_budget_rejections+=second.result.vertex_budget_rejections;
        result.tail_probe_evaluations+=second.result.tail_probe_evaluations;
        for(size_t i=0;i<4;++i) {
            result.audit_evaluations[i]+=second.result.audit_evaluations[i];
            result.rejected_gates[i]+=second.result.rejected_gates[i];
            result.area_rejected_gates[i]+=second.result.area_rejected_gates[i];
            if(second.result.worst_rejected[i].error>=result.worst_rejected[i].error)
                result.worst_rejected[i]=second.result.worst_rejected[i];
        }
        for(auto& trace:second.result.proposals)trace.pass=1;
        result.proposals.insert(result.proposals.end(),
            std::make_move_iterator(second.result.proposals.begin()),
            std::make_move_iterator(second.result.proposals.end()));
        if(second.result.status==Status::Complete&&!second.finalists.empty()) {
            const auto first_count=first.finalists.size();
            result.candidates.insert(result.candidates.end(),
                std::make_move_iterator(second.result.candidates.begin()),
                std::make_move_iterator(second.result.candidates.end()));
            first.finalists.insert(first.finalists.end(),
                std::make_move_iterator(second.finalists.begin()),
                std::make_move_iterator(second.finalists.end()));
            result.selection=select_chain(result.candidates,s.triangle_overhead_bps,result.added_vertex_budget_bytes,ChainObjective::TailFirst);
            result.adaptive_retry_selected=result.selection.selected>=first_count;
            if(result.adaptive_retry_selected)result.tail_reserved_vertex_bytes=second.result.tail_reserved_vertex_bytes;
        }else result.status=second.result.status;
    }
    auto best=first.finalists[result.selection.selected];
    for(size_t i=result.lods.size();i-->0;) {result.lods[i]=std::move(best->lod);best=best->parent;}
    return result;
}
Result generate(MeshView source,const Settings& s,const Proposer& proposer) {
    auto result=generate_impl(source,s,proposer);
    if(s.research.shared_rebuild)try {detail::share_result_vertices(result);}catch(const std::bad_alloc&){result.status=Status::BudgetLimited;}
    return result;
}
}
