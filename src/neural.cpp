#include "neural_internal.hpp"
#include "neural_action.hpp"
#include "neural_action_cache.hpp"
#include "chain_hooks.hpp"
#include <chrono>
#include <random>
namespace blitz {
struct NeuralModel::Impl {neural::WeightsData weights;NeuralOptions options;std::string hash;};
NeuralModel::NeuralModel(const char* file,const NeuralOptions& options) {
    if(!file||!file[0]||options.device<0||options.memory_mib<128||options.memory_mib>65536||!options.action_trials||options.action_trials>65536||options.ranking>NeuralRanking::CurrentPlane)throw std::invalid_argument("invalid neural model options");
    if(!options.action_batch||options.action_batch>64)throw std::invalid_argument("action batch outside 1..64");
    if(!neural_available(options.device))throw NeuralUnavailable("neural mode requires an available CUDA device and a BLITZ_CUDA build");
#ifdef BLITZ_CUDA
    auto value=std::make_unique<Impl>();value->options=options;value->weights=neural::load_weights(file,&value->hash);impl_=std::move(value);
    if(impl_->weights.architecture!=neural::action_schema&&options.ranking!=NeuralRanking::Learned)throw std::invalid_argument("action ranking controls require architecture 2");
#endif
}
NeuralModel::~NeuralModel()=default;
NeuralModel::NeuralModel(NeuralModel&&) noexcept=default;
NeuralModel& NeuralModel::operator=(NeuralModel&&) noexcept=default;
const std::string& NeuralModel::sha256() const {if(!impl_)throw std::invalid_argument("moved-from neural model");return impl_->hash;}
#ifndef BLITZ_CUDA
bool neural_available(int32_t) noexcept {return false;}
Measurement evaluate_cuda(MeshView,MeshView,const Bounds&,const EvalSettings&,const NeuralOptions&,NeuralStats*) {throw NeuralUnavailable("CUDA evaluator was not built");}
double overlap_cuda(MeshView,const Bounds&,double,ViewSet,const NeuralOptions&) {throw NeuralUnavailable("CUDA evaluator was not built");}
#endif
#ifdef BLITZ_CUDA
std::vector<float> neural::encode_mesh_cuda(const Graph& g,const WeightsData& weights,const NeuralOptions& options,const std::function<bool()>& cancelled) {
    std::vector<float> embedding(g.size()*neural::hidden);std::vector<uint32_t> core;
    // Halos retain the full three-layer receptive field. Split until each patch fits.
    std::function<void(std::span<const uint32_t>)> encode=[&](std::span<const uint32_t> ids){
        if(cancelled&&cancelled())return;
        try{auto p=neural::patch(g,ids);auto values=neural::encode_cuda(p.graph,weights,options);
            for(uint32_t i=0;i<p.core;++i)std::copy_n(values.data()+size_t(i)*neural::hidden,neural::hidden,embedding.data()+size_t(p.ids[i])*neural::hidden);}
        catch(const std::length_error&){if(ids.size()<2)throw;encode(ids.first(ids.size()/2));encode(ids.subspan(ids.size()/2));}
    };
    for(uint32_t i=0;i<g.size();++i){core.push_back(i);if(core.size()==16384){encode(core);core.clear();}}if(!core.empty())encode(core);
    return embedding;
}
#endif
Result generate_neural(MeshView source,const Settings& settings,const NeuralModel& model,NeuralStats* stats) {
    if(stats)*stats={};if(!model.impl_)throw std::invalid_argument("invalid neural model");
#ifndef BLITZ_CUDA
    (void)source;(void)settings;throw NeuralUnavailable("neural mode was not built");
#else
    if(auto e=validate(source);!e.empty())throw std::invalid_argument(e);
    if(auto e=validate(settings);!e.empty())throw std::invalid_argument(e);
    if(settings.research.component_candidates||settings.research.independent_seams||settings.research.topology_fallback)
        throw std::invalid_argument("CPU research proposal options are unsupported in neural mode");
    NeuralStats local;auto& counters=stats?*stats:local;auto& options=model.impl_->options;auto& weights=model.impl_->weights;
    using Clock=std::chrono::steady_clock;auto nanos=[](auto start){return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count());};
    auto start=Clock::now();neural::Graph g;std::vector<float> embedding;
    std::unique_ptr<neural::ActionCuda> action_network;
    if(weights.architecture==neural::schema){g=neural::graph(source);embedding=neural::encode_mesh_cuda(g,weights,options,settings.cancelled);}
    else action_network=std::make_unique<neural::ActionCuda>(weights,options);
    counters.encode_ns=nanos(start);
    // All proposals use the shared source encoding. The conditioned head predicts a
    // complete retention/representative field for each requested size and target.
    Settings s=settings;s.research.chain=ChainMode::Direct;
    detail::GenerationHooks hooks;
    hooks.propose=[&](MeshView,const ReduceSettings& rs,const EvalSettings& config,double adjacent){
        auto e=config;e.max_changed_area=s.max_changed_area;auto begin=Clock::now();
        auto prediction=neural::predict_cuda(embedding,neural::condition(e,adjacent,double(rs.target_triangles)/source.triangles()),weights,options);
        counters.inference_ns+=nanos(begin);begin=Clock::now();neural::DecodeStats decoded;
        auto candidate=neural::decode(source,g,prediction,rs.target_triangles,rs.output,&decoded,s.cancelled);
        counters.decode_ns+=nanos(begin);++counters.decoded;counters.legal_collapses+=decoded.accepted;counters.rejected_collapses+=decoded.rejected;return candidate;
    };
    hooks.evaluate=[&](MeshView a,MeshView b,const Bounds& bounds,const EvalSettings& e){auto begin=Clock::now();auto bounded=e;
        bounded.max_supersample=neural::bounded_refinement(e);if(bounded.max_supersample<e.max_supersample)++counters.bounded_audits;
        auto m=evaluate_cuda(a,b,bounds,bounded,options,&counters);counters.gpu_audit_ns+=nanos(begin);return m;};
    std::mt19937 ranking_random(options.ranking_seed);
    neural::ActionRejections source_rejections;
    if(action_network)hooks.propose_guarded=[&](MeshView input,MeshView fixed_source,MeshView previous,const Bounds& bounds,const ReduceSettings& rs,const EvalSettings& source_eval,const EvalSettings& adjacent_eval){
        auto begin=Clock::now();auto nested_before=counters.inference_ns+counters.gpu_audit_ns;neural::ActionStats stats;
        source_rejections.configure(source_eval);
        auto rank=[&](const neural::ActionState& state,std::span<const neural::ActionRecord> actions){auto t=Clock::now();std::vector<float> scores(actions.size());
            if(options.ranking==NeuralRanking::Learned||options.ranking==NeuralRanking::Shuffled){std::vector<float> x;x.reserve(actions.size()*neural::action_features);
                for(auto& a:actions)x.insert(x.end(),a.x.begin(),a.x.end());auto values=action_network->predict(x);
                for(size_t i=0;i<scores.size();++i)scores[i]=values[i*neural::action_outputs];
                if(options.ranking==NeuralRanking::Shuffled)std::shuffle(scores.begin(),scores.end(),ranking_random);
            }else if(options.ranking==NeuralRanking::ShortestEdge){for(size_t i=0;i<scores.size();++i)scores[i]=-actions[i].x[59];}
            else if(options.ranking==NeuralRanking::CurrentPlane){for(size_t i=0;i<scores.size();++i)scores[i]=float(-state.teacher_cost(actions[i].action));}
            counters.inference_ns+=nanos(t);return scores;};
        auto gate=[&](MeshView candidate){if(s.cancelled&&s.cancelled())return false;
            if(source_rejections.contains(candidate)){++counters.action_audit_cache_hits;return false;}
            auto a=hooks.evaluate(fixed_source,candidate,bounds,source_eval);if(!a.complete||!a.passed){if(neural::action_audit_known(a,source_eval)&&!a.passed)source_rejections.insert(candidate);return false;}
            auto b=hooks.evaluate(previous,candidate,bounds,adjacent_eval);return b.complete&&b.passed;};
        auto candidate=neural::execute_actions(input,neural::condition(source_eval,adjacent_eval.limit,double(rs.target_triangles)/input.triangles()),rs.target_triangles,options.action_trials,rank,gate,&stats,s.cancelled,options.action_batch);
        if(rs.output==OutputMode::Rebuild){candidate.data=copy_mesh(candidate.view(input));compact(candidate.data);candidate.shared_vertices=false;}
        auto elapsed=nanos(begin),nested=counters.inference_ns+counters.gpu_audit_ns-nested_before;counters.decode_ns+=elapsed-std::min(elapsed,nested);
        ++counters.decoded;counters.legal_collapses+=stats.accepted;counters.rejected_collapses+=stats.rejected;counters.action_ranked+=stats.ranked;counters.action_trials+=stats.trials;
        return candidate;
    };
    hooks.confirm=[&](Result& r){auto begin=Clock::now();bool good=true;
        for(size_t i=1;i<r.lods.size()&&good;++i) {
            auto& lod=r.lods[i];EvalSettings e;e.profile=s.profile;e.weights=s.weights;e.views=s.audit_views;e.supersample=s.audit_supersample;e.max_supersample=s.max_supersample;
            e.screen_size=lod.schedule.pixels;e.max_changed_area=s.max_changed_area;e.force_scalar=s.force_scalar;e.cancelled=s.cancelled;
            double t=s.levels==2?0:double(i-1)/(s.levels-2);e.weights.normal*=s.normal_importance.at(t);e.weights.color*=s.attribute_importance.at(t);e.weights.material*=s.attribute_importance.at(t);
            e.limit=lod.schedule.source;auto a=evaluate(source,lod.view(source),r.reference_bounds,e);e.limit=lod.schedule.transition;
            auto b=evaluate(r.lods[i-1].view(source),lod.view(source),r.reference_bounds,e);good=a.passed&&a.complete&&b.passed&&b.complete;
            lod.source_error=a;lod.adjacent=b;
        }
        counters.reference_audit_ns+=nanos(begin);if(!good)++counters.reference_rejections;return good;
    };
    if(options.overdraw_tiebreak)hooks.tiebreak=[&](const Result& r){double total=0;for(size_t i=1;i<r.lods.size();++i)total+=overlap_cuda(r.lods[i].view(source),r.reference_bounds,r.lods[i].schedule.pixels,s.search_views,options);return total;};
    auto result=detail::generate_with_hooks(source,s,{},&hooks);
    for(size_t i=1;i<result.lods.size();++i)if(same_mesh_data(result.lods[i].view(source),result.lods[i-1].view(source)))++counters.fallback_levels;
    return result;
#endif
}
}
