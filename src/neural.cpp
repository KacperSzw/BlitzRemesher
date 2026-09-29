#include "neural_internal.hpp"
#include "neural_action.hpp"
#include "neural_action_cache.hpp"
#include "neural_action_gpu.hpp"
#include "neural_memory.hpp"
#include "chain_hooks.hpp"
#include <chrono>
#include <random>
namespace blitz {
struct NeuralModel::Impl {neural::WeightsData weights;NeuralOptions options;std::string hash;};
NeuralModel::NeuralModel(const char* file,const NeuralOptions& options) {
    if(!file||!file[0]||options.device<0||options.memory_mib<128||options.memory_mib>65536||!options.action_trials||options.action_trials>65536||options.ranking>NeuralRanking::CurrentPlane)throw std::invalid_argument("invalid neural model options");
    if(!options.action_batch||options.action_batch>64)throw std::invalid_argument("action batch outside 1..64");
    if(options.confirmation>NeuralConfirmation::Compare)throw std::invalid_argument("invalid neural confirmation backend");
    if(!neural_available(options.device))throw NeuralUnavailable("neural mode requires an available CUDA device and a BLITZ_CUDA build");
#ifdef BLITZ_CUDA
    auto value=std::make_unique<Impl>();value->options=options;value->weights=neural::load_weights(file,&value->hash);impl_=std::move(value);
    if(impl_->weights.architecture==neural::schema&&options.ranking!=NeuralRanking::Learned)throw std::invalid_argument("action ranking controls require an action policy");
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
    neural::MemoryScope memory(options);
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
    neural::AuditCuda audit(options,source);
    hooks.propose=[&](MeshView,const ReduceSettings& rs,const EvalSettings& config,double adjacent){
        auto e=config;e.max_changed_area=s.max_changed_area;auto begin=Clock::now();
        auto prediction=neural::predict_cuda(embedding,neural::condition(e,adjacent,double(rs.target_triangles)/source.triangles()),weights,options);
        counters.inference_ns+=nanos(begin);begin=Clock::now();neural::DecodeStats decoded;
        auto candidate=neural::decode(source,g,prediction,rs.target_triangles,rs.output,&decoded,s.cancelled);
        counters.decode_ns+=nanos(begin);++counters.decoded;counters.legal_collapses+=decoded.accepted;counters.rejected_collapses+=decoded.rejected;return candidate;
    };
    hooks.evaluate=[&](MeshView a,MeshView b,const Bounds& bounds,const EvalSettings& e){auto begin=Clock::now();auto bounded=e;
        bounded.max_supersample=neural::bounded_refinement(e);if(bounded.max_supersample<e.max_supersample)++counters.bounded_audits;
        auto m=audit.evaluate(a,b,bounds,bounded,&counters);counters.gpu_audit_ns+=nanos(begin);return m;};
    std::unique_ptr<neural::GpuActionState> action_state,placement_state;
    if(action_network)action_state=std::make_unique<neural::GpuActionState>(source,options);
    if(action_network)hooks.propose_guarded=[&](MeshView input,MeshView fixed_source,MeshView previous,const Bounds& bounds,const ReduceSettings& rs,const EvalSettings& source_eval,const EvalSettings& adjacent_eval){
        auto begin=Clock::now();auto nested_before=counters.inference_ns+counters.gpu_audit_ns;neural::ActionStats stats;
        auto* state=action_state.get();if(weights.architecture==neural::placement_schema&&rs.output==OutputMode::Rebuild){
            if(!placement_state)placement_state=std::make_unique<neural::GpuActionState>(source,options,true);state=placement_state.get();}state->reset();
        auto evaluate_device=[&](MeshView reference,neural::DeviceMeshView candidate,const EvalSettings& config){auto t=Clock::now();auto bounded=config;
            bounded.max_supersample=neural::bounded_refinement(config);if(bounded.max_supersample<config.max_supersample)++counters.bounded_audits;
            auto result=audit.evaluate(reference,candidate,bounds,bounded,&counters);counters.gpu_audit_ns+=nanos(t);return result;};
        auto gate=[&](neural::DeviceMeshView candidate){if(s.cancelled&&s.cancelled())return false;
            auto a=evaluate_device(fixed_source,candidate,source_eval);if(!a.complete||!a.passed)return false;
            auto b=evaluate_device(previous,candidate,adjacent_eval);return b.complete&&b.passed;};
        auto candidate=state->execute(neural::condition(source_eval,adjacent_eval.limit,double(rs.target_triangles)/input.triangles()),rs.target_triangles,options.action_trials,action_network.get(),options.ranking,options.ranking_seed,options.action_batch,gate,&stats,s.cancelled);
        counters.inference_ns+=stats.inference_ns;
        if(rs.output==OutputMode::Rebuild&&candidate.shared_vertices){candidate.data=copy_mesh(candidate.view(input));compact(candidate.data);candidate.shared_vertices=false;}
        auto elapsed=nanos(begin),nested=counters.inference_ns+counters.gpu_audit_ns-nested_before;counters.decode_ns+=elapsed-std::min(elapsed,nested);
        ++counters.decoded;counters.legal_collapses+=stats.accepted;counters.rejected_collapses+=stats.rejected;counters.action_ranked+=stats.ranked;counters.action_trials+=stats.trials;
        return candidate;
    };
    hooks.confirm=[&](Result& r){bool good=true;
        auto confirm=[&](MeshView reference,MeshView candidate,const EvalSettings& e,uint32_t level,bool adjacent,Measurement& output){
            auto begin=Clock::now();Measurement cpu,gpu;
            if(options.confirmation!=NeuralConfirmation::Cpu){auto t=Clock::now();gpu=audit.evaluate(reference,candidate,r.reference_bounds,e,&counters);counters.gpu_confirmation_ns+=nanos(t);}
            if(options.confirmation!=NeuralConfirmation::Gpu){auto t=Clock::now();cpu=evaluate(reference,candidate,r.reference_bounds,e);counters.reference_audit_ns+=nanos(t);}
            output=options.confirmation==NeuralConfirmation::Cpu?cpu:gpu;
            auto close=[](double a,double b){return a==b||(std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<=1e-5);};
            bool agrees=options.confirmation!=NeuralConfirmation::Compare||(cpu.passed==gpu.passed&&cpu.complete==gpu.complete&&cpu.resource_limited==gpu.resource_limited
                &&cpu.views_evaluated==gpu.views_evaluated&&close(cpu.error,gpu.error)&&close(cpu.coverage,gpu.coverage)&&close(cpu.coverage_upper,gpu.coverage_upper)
                &&cpu.changed_area==gpu.changed_area&&close(cpu.normal_degrees,gpu.normal_degrees));
            bool stopped=s.cancelled&&s.cancelled();
            auto invalid=[](const Measurement& m){return std::isnan(m.error)||std::isnan(m.coverage)||std::isnan(m.coverage_upper)
                ||!std::isfinite(m.changed_area)||!std::isfinite(m.normal_degrees)||(m.passed&&(!std::isfinite(m.error)||!std::isfinite(m.coverage)||!std::isfinite(m.coverage_upper)));};
            bool nonfinite=invalid(output)||(options.confirmation==NeuralConfirmation::Compare&&invalid(cpu));
            bool resource=output.resource_limited||(options.confirmation==NeuralConfirmation::Compare&&cpu.resource_limited);
            if(!stopped&&!nonfinite&&agrees&&output.complete&&output.passed)return true;
            NeuralConfirmationReason reason;
            if(stopped){reason=NeuralConfirmationReason::Cancelled;++counters.confirmation_cancelled;}
            else if(resource){reason=NeuralConfirmationReason::Resource;++counters.confirmation_resources;}
            else if(nonfinite){reason=NeuralConfirmationReason::Nonfinite;++counters.confirmation_nonfinite;}
            else if(!agrees){reason=NeuralConfirmationReason::Disagreement;++counters.confirmation_disagreements;}
            else {reason=NeuralConfirmationReason::Visual;++counters.reference_rejections;}
            if(!counters.confirmation_failure){
                auto& failure=counters.confirmation_failure.emplace();failure.bounds=r.reference_bounds;failure.settings=e;failure.settings.cancelled={};failure.settings.performance=nullptr;
                failure.cpu=cpu;failure.gpu=gpu;failure.level=level;failure.backend=options.confirmation;failure.reason=reason;failure.adjacent=adjacent;failure.nanoseconds=nanos(begin);
                if(options.capture_confirmation_failure){failure.reference=copy_mesh(reference);failure.candidate=copy_mesh(candidate);}
            }
            output.complete=false;output.passed=false;return false;
        };
        for(size_t i=1;i<r.lods.size()&&good;++i) {
            auto& lod=r.lods[i];EvalSettings e;e.profile=s.profile;e.weights=s.weights;e.views=s.audit_views;e.supersample=s.audit_supersample;e.max_supersample=s.max_supersample;
            e.screen_size=lod.schedule.pixels;e.max_changed_area=s.max_changed_area;e.force_scalar=s.force_scalar;e.cancelled=s.cancelled;
            double t=s.levels==2?0:double(i-1)/(s.levels-2);e.weights.normal*=s.normal_importance.at(t);e.weights.color*=s.attribute_importance.at(t);e.weights.material*=s.attribute_importance.at(t);
            e.limit=lod.schedule.source;good=confirm(source,lod.view(source),e,uint32_t(i),false,lod.source_error);
            if(good){e.limit=lod.schedule.transition;good=confirm(r.lods[i-1].view(source),lod.view(source),e,uint32_t(i),true,lod.adjacent);}
        }
        return good;
    };
    if(options.overdraw_tiebreak)hooks.tiebreak=[&](const Result& r){double total=0;for(size_t i=1;i<r.lods.size();++i)total+=overlap_cuda(r.lods[i].view(source),r.reference_bounds,r.lods[i].schedule.pixels,s.search_views,options);return total;};
    auto result=detail::generate_with_hooks(source,s,{},&hooks);
    for(size_t i=1;i<result.lods.size();++i)if(same_mesh_data(result.lods[i].view(source),result.lods[i-1].view(source)))++counters.fallback_levels;
    return result;
#endif
}
}
