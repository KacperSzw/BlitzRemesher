#include "blitz/remesher.hpp"
#include "coverage.hpp"
#include "timing.hpp"
#include <chrono>
#include <numeric>
#include <stdexcept>

namespace blitz::detail {
namespace {
constexpr uint32_t no_parent=UINT32_MAX;
constexpr uint16_t new_mesh=UINT16_MAX;
// A pass retains at most 32 source/incumbent meshes plus 31 * (2*32+2)
// layer meshes. Payload IDs fit u16; path IDs can exceed u16 at large beams.
struct Path {
    uint32_t parent{no_parent};
    uint16_t mesh{};
    uint8_t level{};
    uint64_t triangles{};
    StorageStats storage;
    Measurement source,adjacent;
};
struct Candidate {
    Lod owned;
    Measurement source;
    uint16_t mesh{new_mesh},origin{};
    bool pinned{};
};
struct Trial { size_t target; bool accepted; };
struct Slot {
    uint16_t input{},parent{};
    OutputMode output{};
    bool direct{};
    std::vector<Trial> trials;
};
uint64_t triangle_total(const Result& r) {
    uint64_t total=0;for(size_t i=1;i<r.lods.size();++i)total+=r.lods[i].data.indices.size()/3;return total;
}
EvalSettings evaluation(const Settings& s,ScheduleEntry step,unsigned level,bool audit,bool source) {
    EvalSettings e;e.profile=s.profile;e.weights=s.weights;e.screen_size=step.pixels;
    e.limit=source?step.source:step.transition;e.conservative_screen=!audit&&s.research.conservative_screen;
    e.max_changed_area=audit?s.max_changed_area:1;
    const double t=s.levels==2?0:double(level-1)/(s.levels-2);
    e.weights.normal*=s.normal_importance.at(t);e.weights.color*=s.attribute_importance.at(t);e.weights.material*=s.attribute_importance.at(t);
    e.views=audit?s.audit_views:s.search_views;e.supersample=audit?s.audit_supersample:s.search_supersample;
    e.max_supersample=s.max_supersample;e.force_scalar=s.force_scalar;e.cancelled=s.cancelled;e.performance=s.performance;return e;
}
struct Interrupted {};
class Graph {
    Result& result;
    const Settings& settings;
    const Proposer& proposer;
    uint8_t pass;
    std::vector<Lod> meshes;
    std::vector<Path> paths;
    std::vector<uint32_t> incumbent;
    std::vector<uint32_t> parents;
    uint64_t incumbent_total{};

    void poll() {
        if(settings.cancelled&&settings.cancelled()){result.status=Status::Cancelled;throw Interrupted{};}
    }
    MeshView view(uint16_t id) const { return meshes[id].view(result.source); }
    MeshView view(const Candidate& c) const { return c.mesh==new_mesh?c.owned.view(result.source):view(c.mesh); }
    uint16_t intern(Lod lod) {
        const auto v=lod.view(result.source);
        for(size_t i=0;i<meshes.size();++i)if(same_mesh_data(v,view(uint16_t(i))))return uint16_t(i);
        if(meshes.size()>=new_mesh)throw std::length_error("graph payload ID overflow");
        meshes.push_back(std::move(lod));return uint16_t(meshes.size()-1);
    }
    ChainCost cost(uint32_t id) const {
        ChainCost c;c.storage=paths[id].storage;
        for(auto p=id;p!=no_parent;p=paths[p].parent)c.triangles.push_back(uint32_t(view(paths[p].mesh).triangles()));
        std::reverse(c.triangles.begin(),c.triangles.end());return c;
    }
    uint32_t append(Path path) {
        if(paths.size()>=no_parent)throw std::length_error("graph path ID overflow");
        paths.push_back(std::move(path));return uint32_t(paths.size()-1);
    }
    bool less(uint32_t a,uint32_t b) const {
        const auto& x=paths[a];const auto& y=paths[b];
        if(x.triangles!=y.triangles)return x.triangles<y.triangles;
        if(x.storage.total()!=y.storage.total())return x.storage.total()<y.storage.total();
        return a<b;
    }
    bool measure(CoverageCache& cache,MeshView reference,MeshView candidate,const EvalSettings& e,
                 uint8_t id,unsigned stage,Measurement& m) {
        poll();++result.audit_evaluations[stage];m=cache.evaluate(reference,candidate,e,id,stage>=2);
        if(m.resource_limited){result.status=Status::BudgetLimited;throw Interrupted{};}
        if(m.cancelled){result.status=Status::Cancelled;throw Interrupted{};}
        poll();
        if(m.passed&&m.complete)return true;
        ++result.rejected_gates[stage];
        if(stage>=2&&m.error<=e.limit&&m.changed_area>e.max_changed_area)++result.area_rejected_gates[stage];
        if(m.error>=result.worst_rejected[stage].error)result.worst_rejected[stage]=m;
        return false;
    }
    void trim_candidates(std::vector<Candidate>& candidates) {
        const size_t limit=2u*settings.beam_width;
        std::vector<size_t> order;for(size_t i=0;i<candidates.size();++i)if(!candidates[i].pinned)order.push_back(i);
        if(order.size()<=limit)return;
        std::stable_sort(order.begin(),order.end(),[&](size_t a,size_t b){
            auto va=view(candidates[a]),vb=view(candidates[b]);
            return va.triangles()!=vb.triangles()?va.triangles()<vb.triangles():vertex_bytes(va)<vertex_bytes(vb);
        });
        std::vector<uint8_t> keep(candidates.size());
        for(size_t i=0;i<settings.beam_width;++i)keep[order[i]]=1;
        auto bytes=[&](size_t i){const auto& c=candidates[i];return (c.mesh==new_mesh?c.owned.shared_vertices:meshes[c.mesh].shared_vertices)?0:vertex_bytes(view(c));};
        std::stable_sort(order.begin(),order.end(),[&](size_t a,size_t b){return bytes(a)!=bytes(b)?bytes(a)<bytes(b):view(candidates[a]).triangles()<view(candidates[b]).triangles();});
        size_t retained=settings.beam_width;
        for(auto i:order)if(!keep[i]&&retained<limit){keep[i]=1;++retained;}
        size_t dst=0;for(size_t i=0;i<candidates.size();++i)if(keep[i]||candidates[i].pinned){if(dst!=i)candidates[dst]=std::move(candidates[i]);++dst;}
        result.graph_pruned_candidates+=candidates.size()-dst;candidates.resize(dst);
    }
    // Dominance is valid only at identical terminal geometry. With an allowance,
    // also preserve every earlier per-slot count used by final eligibility.
    bool dominates(uint32_t a,uint32_t b) const {
        const auto& x=paths[a];const auto& y=paths[b];
        if(x.mesh!=y.mesh||x.triangles>y.triangles||x.storage.added_vertex_bytes>y.storage.added_vertex_bytes||x.storage.total()>y.storage.total())return false;
        if(settings.triangle_overhead_bps) {
            auto ca=cost(a),cb=cost(b);for(size_t i=1;i<ca.triangles.size();++i)if(ca.triangles[i]>cb.triangles[i])return false;
        }
        return true;
    }
    void trim_paths(std::vector<uint32_t>& next,uint32_t pinned,uint32_t source_path) {
        std::vector<uint32_t> distinct;
        for(auto id:next) {
            if(id!=pinned&&id!=source_path&&std::any_of(distinct.begin(),distinct.end(),[&](auto other){return dominates(other,id);}))continue;
            std::erase_if(distinct,[&](auto other){return other!=pinned&&other!=source_path&&dominates(id,other);});
            distinct.push_back(id);
        }
        std::stable_sort(distinct.begin(),distinct.end(),[&](auto a,auto b){return less(a,b);});
        const size_t limit=4u*settings.beam_width;
        std::vector<uint32_t> retained;
        for(auto id:distinct)if(retained.size()<limit/2&&id!=pinned&&id!=source_path)retained.push_back(id);
        auto cheap=distinct;std::stable_sort(cheap.begin(),cheap.end(),[&](auto a,auto b){
            auto x=paths[a].storage.added_vertex_bytes,y=paths[b].storage.added_vertex_bytes;return x!=y?x<y:less(a,b);
        });
        for(auto id:cheap)if(retained.size()<limit&&id!=pinned&&id!=source_path&&std::find(retained.begin(),retained.end(),id)==retained.end())retained.push_back(id);
        retained.push_back(pinned);if(source_path!=pinned)retained.push_back(source_path);
        result.graph_pruned_paths+=next.size()-retained.size();next=std::move(retained);
        std::stable_sort(next.begin(),next.end(),[&](auto a,auto b){return less(a,b);});
    }
public:
    Graph(Result& r,const Settings& s,const Proposer& p,uint8_t iteration):result(r),settings(s),proposer(p),pass(iteration) {
        incumbent_total=triangle_total(r);
        meshes.reserve(32+31*(2u*s.beam_width+2));
        StorageStats storage{vertex_bytes(r.source),0,0};uint32_t parent=no_parent;uint64_t total=0;
        for(size_t i=0;i<r.lods.size();++i) {
            auto id=intern(r.lods[i]);const auto v=view(id);
            if(!i||id!=paths[parent].mesh) {
                storage.index_bytes+=v.indices.size()*4;
                if(!meshes[id].shared_vertices)storage.added_vertex_bytes+=vertex_bytes(v);
            }
            if(i)total+=v.triangles();
            parent=append({parent,id,uint8_t(i),total,storage,r.lods[i].source_error,r.lods[i].adjacent});incumbent.push_back(parent);
        }
        parents={incumbent[0]};
    }
    void run() {
        uint32_t source_path=incumbent[0];
        for(uint8_t level=1;level<settings.levels;++level) {
            poll();const auto step=result.lods[level].schedule;
            const auto ss=evaluation(settings,step,level,false,true),sa=evaluation(settings,step,level,true,true);
            const auto ts=evaluation(settings,step,level,false,false),ta=evaluation(settings,step,level,true,false);
            CoverageCache cache(settings.profile==Profile::Coverage?uint32_t(settings.research.coverage_cache_mib)*1024*1024:0,result.reference_bounds);
            std::vector<Candidate> candidates;
            candidates.push_back({{},paths[incumbent[level]].source,paths[incumbent[level]].mesh,paths[incumbent[level-1]].mesh,true});
            if(candidates[0].mesh!=0)candidates.push_back({{},Measurement{},0,0,true});
            auto offer=[&](Candidate candidate)->uint8_t {
                const auto v=view(candidate);
                if(!validate(v).empty())return 5;
                for(auto& previous:candidates)if(same_mesh_data(v,view(previous))){++result.duplicate_proposals;return 7;}
                const bool shared=candidate.mesh==new_mesh?candidate.owned.shared_vertices:meshes[candidate.mesh].shared_vertices;
                if(result.added_vertex_budget_bytes&&!shared&&vertex_bytes(v)>*result.added_vertex_budget_bytes){++result.vertex_budget_rejections;return 9;}
                if(!settings.triangle_overhead_bps&&std::none_of(parents.begin(),parents.end(),[&](auto p){return paths[p].triangles+v.triangles()<=incumbent_total;})) {
                    ++result.graph_pruned_candidates;return 10;
                }
                cache.begin_candidate();Measurement screen;
                if(!measure(cache,result.source,v,ss,0,0,screen))return 1;
                if(!measure(cache,result.source,v,sa,0,2,candidate.source))return 3;
                ++result.graph_candidates;candidates.push_back(std::move(candidate));trim_candidates(candidates);return 0;
            };
            std::vector<uint16_t> input_meshes;
            for(auto parent:parents)if(std::find(input_meshes.begin(),input_meshes.end(),paths[parent].mesh)==input_meshes.end()) {
                auto id=paths[parent].mesh;input_meshes.push_back(id);offer({{},Measurement{},id,id,false});
            }
            // Each target history belongs to one input mesh and one placement.
            // Direct and progressive histories never share pass/fail brackets.
            std::vector<Slot> slots;
            for(auto output:{OutputMode::Rebuild,OutputMode::Reuse}) {
                slots.push_back({0,paths[parents.front()].mesh,output,true,{}});
                size_t count=0;for(auto id:input_meshes)if(id&&count++<settings.beam_width)slots.push_back({id,id,output,false,{}});
            }
            struct Relax { uint16_t input,parent;ReduceSettings settings;bool direct; };
            std::optional<Relax> relaxed;
            for(unsigned attempt=0;attempt<settings.candidate_budget;++attempt) {
                poll();auto& slot=slots[attempt%slots.size()];
                const size_t parent_count=view(slot.parent).triangles();
                constexpr double fractions[]={.5,.25,.75,.125,.9,.02};
                size_t target=std::max<size_t>(1,size_t(parent_count*fractions[(attempt/slots.size()+pass-1)%6]));
                if(!slot.trials.empty()&&attempt/slots.size()%4!=3) {
                    size_t high=parent_count,low=0;
                    for(auto trial:slot.trials)if(trial.accepted)high=std::min(high,trial.target);
                    for(auto trial:slot.trials)if(!trial.accepted&&trial.target<high)low=std::max(low,trial.target);
                    auto midpoint=std::max<size_t>(1,(low+high)/2);
                    if(std::none_of(slot.trials.begin(),slot.trials.end(),[&](auto t){return t.target==midpoint;}))target=midpoint;
                }
                ReduceSettings rs;rs.output=slot.output;rs.objective=settings.objective;rs.target_triangles=target;
                rs.appearance_stage=settings.research.appearance_stage;rs.appearance_weights=ss.weights;if(settings.profile!=Profile::Attributes)rs.appearance_weights.color=0;rs.screen_size=ss.screen_size;
                rs.normal_weight=ss.weights.normal;rs.cancelled=settings.cancelled;rs.prune=settings.prune&&((attempt/slots.size()+pass)%2==0);
                rs.coupled_wedges=settings.coupled_wedges;rs.boundary_weight=settings.research.boundary_weight;
                rs.boundary_placement=settings.research.boundary_placement;rs.independent_seams=settings.research.independent_seams;
                uint16_t input_id=slot.input,parent_id=slot.parent;bool direct=slot.direct,topology=false;
                if(relaxed) {input_id=relaxed->input;parent_id=relaxed->parent;direct=relaxed->direct;rs=relaxed->settings;relaxed.reset();topology=true;}
                if(result.added_vertex_budget_bytes&&rs.output==OutputMode::Rebuild) {
                    const auto stride=vertex_bytes(result.source)/result.source.positions.count;
                    if(stride&&*result.added_vertex_budget_bytes/stride>=3)rs.target_triangles=std::min(rs.target_triangles,size_t(*result.added_vertex_budget_bytes/stride/3));
                }
                rs.screen_size=ss.screen_size*bounds(view(input_id)).diameter()/result.reference_bounds.diameter();
                ReductionStats stats;rs.statistics=&stats;Lod lod;const auto start=std::chrono::steady_clock::now();
                {ScopedTime timer(settings.performance?&settings.performance->reduction_ns:nullptr);lod=proposer?proposer(view(input_id),rs):reduce(view(input_id),rs);}
                ++result.candidate_evaluations;if(topology)++result.topology_fallback_proposals;
                if(auto p=settings.performance) {
                    p->solve_attempts+=stats.solve_attempts;p->singular_solves+=stats.singular_solves;p->nonfinite_solves+=stats.nonfinite_solves;
                    p->position_fallbacks+=stats.position_fallbacks;p->nonfinite_costs+=stats.nonfinite_costs;p->appearance_peak_bytes=std::max(p->appearance_peak_bytes,stats.appearance_bytes);
                }
                poll();const auto achieved=lod.data.indices.size()/3;
                // Borrowed indices from a rebuilt predecessor address that input,
                // never LOD0. Intern unchanged output without copying its streams.
                Candidate c;c.origin=parent_id;
                if(lod.shared_vertices&&!meshes[input_id].shared_vertices) {
                    const auto local=lod.view(view(input_id));
                    if(same_mesh_data(local,view(input_id)))c.mesh=input_id;
                    else {lod.data=copy_mesh(local);compact(lod.data);lod.shared_vertices=false;c.owned=std::move(lod);}
                }else c.owned=std::move(lod);
                auto gate=offer(std::move(c));
                if(!topology)slot.trials.push_back({rs.target_triangles,gate==0||gate==7});
                if(settings.research.trace)result.proposals.push_back({uint32_t(view(input_id).triangles()),uint32_t(view(parent_id).triangles()),uint32_t(rs.target_triangles),uint32_t(achieved),
                    stats.attempts,stats.collapsed,stats.geometry_rejections,stats.uv_rejections,stats.link_rejections,
                    std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count(),level,uint8_t(!direct),uint8_t(topology?3:rs.output==OutputMode::Reuse?1:0),gate,uint8_t(pass+1)});
                if(!topology&&!proposer&&settings.research.topology_fallback&&stats.link_rejections&&uint64_t(achieved)>uint64_t(rs.target_triangles)*4) {
                    rs.objective=Objective::TopologyRelaxed;rs.statistics=nullptr;relaxed=Relax{input_id,parent_id,rs,direct};
                }
            }
            poll();std::vector<uint32_t> next{incumbent[level]};
            auto source_node=paths[source_path];source_node.parent=source_path;source_node.level=level;
            source_node.triangles+=result.source.triangles();source_node.source={};source_node.adjacent={};
            source_path=append(source_node);next.push_back(source_path);
            // Stable reference IDs are local to this layer and fit the cache's u8
            // key: at most 4*beam_width+2 parents (<=130), plus source ID zero.
            std::vector<uint16_t> references{0};
            for(auto parent:parents)if(std::find(references.begin(),references.end(),paths[parent].mesh)==references.end())references.push_back(paths[parent].mesh);
            for(auto& candidate:candidates) {
                auto id=candidate.mesh==new_mesh?intern(std::move(candidate.owned)):candidate.mesh;
                const auto v=view(id);std::vector<uint16_t> incoming;
                auto affordable=[&](uint32_t parent) {
                    if(!result.added_vertex_budget_bytes||meshes[id].shared_vertices||paths[parent].mesh==id)return true;
                    auto used=paths[parent].storage.added_vertex_bytes,added=vertex_bytes(v),cap=*result.added_vertex_budget_bytes;
                    return used<=cap&&added<=cap-used;
                };
                for(auto parent:parents)if(!affordable(parent))++result.vertex_budget_rejections;
                auto add=[&](uint16_t mesh){
                    if(incoming.size()<4&&std::find(incoming.begin(),incoming.end(),mesh)==incoming.end()&&
                       std::any_of(parents.begin(),parents.end(),[&](auto p){return paths[p].mesh==mesh&&view(mesh).triangles()>=v.triangles()&&
                           affordable(p)&&(settings.triangle_overhead_bps||paths[p].triangles+v.triangles()<=incumbent_total);}))incoming.push_back(mesh);
                };
                add(candidate.origin);add(paths[incumbent[level-1]].mesh);
                for(auto p:parents)add(paths[p].mesh);
                cache.begin_candidate();
                for(auto from:incoming) {
                    poll();Measurement adjacent,screen;
                    auto reference_id=uint8_t(std::find(references.begin(),references.end(),from)-references.begin());
                    ++result.graph_edges;++result.transition_reconnections;
                    if(from==id)adjacent={};
                    else if(!from&&step.source==step.transition)adjacent=candidate.source;
                    else if(id==paths[incumbent[level]].mesh&&from==paths[incumbent[level-1]].mesh)adjacent=paths[incumbent[level]].adjacent;
                    else if(!measure(cache,view(from),v,ts,reference_id,1,screen)||!measure(cache,view(from),v,ta,reference_id,3,adjacent))continue;
                    for(auto parent:parents)if(paths[parent].mesh==from) {
                        Path p{parent,id,level,paths[parent].triangles+v.triangles(),paths[parent].storage,candidate.source,adjacent};
                        if(!settings.triangle_overhead_bps&&p.triangles>incumbent_total)continue;
                        if(from!=id) {p.storage.index_bytes+=v.indices.size()*4;if(!meshes[id].shared_vertices)p.storage.added_vertex_bytes+=vertex_bytes(v);}
                        if(result.added_vertex_budget_bytes&&p.storage.added_vertex_bytes>*result.added_vertex_budget_bytes){++result.vertex_budget_rejections;continue;}
                        next.push_back(append(p));
                    }
                }
            }
            trim_paths(next,incumbent[level],source_path);parents=std::move(next);
        }
        poll();std::vector<ChainCost> costs;for(auto id:parents)costs.push_back(cost(id));
        auto selection=select_chain(costs,settings.triangle_overhead_bps,result.added_vertex_budget_bytes,ChainObjective::WholeChain);
        auto id=parents[selection.selected];std::vector<Lod> chain(settings.levels);
        for(size_t i=chain.size();i-->0;) {
            const auto& node=paths[id];chain[i]=meshes[node.mesh];chain[i].schedule=result.lods[i].schedule;
            chain[i].source_error=node.source;chain[i].adjacent=node.adjacent;id=node.parent;
        }
        // Publish only a complete chain. All earlier owned payloads remain live
        // until construction succeeds, including during allocation failure.
        result.lods=std::move(chain);result.candidates=std::move(costs);result.selection=selection;result.chain_objective=ChainObjective::WholeChain;
    }
};
}
Result improve_chain(Result result,const Settings& settings,const Proposer& proposer,double seed_seconds) {
    const auto begin=std::chrono::steady_clock::now();
    auto progress=[&](uint8_t pass) {
        result.search_progress.push_back({result.candidate_evaluations,std::accumulate(result.audit_evaluations.begin(),result.audit_evaluations.end(),uint64_t{}),
            triangle_total(result),storage_stats(result).added_vertex_bytes,seed_seconds+std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count(),pass});
    };
    try {
        progress(0);
        if(result.status!=Status::Complete)return result;
        for(uint8_t pass=1;pass<=settings.research.graph_passes;++pass) {
            Graph graph(result,settings,proposer,pass);graph.run();++result.graph_passes_completed;progress(pass);
        }
    }catch(const Interrupted&) {
        // The incumbent was already fully audited before this pass started.
    }catch(const std::bad_alloc&) {result.status=Status::BudgetLimited;}
    return result;
}
}
