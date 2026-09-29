#include "neural_internal.hpp"
#include "chain_hooks.hpp"
#include <chrono>
namespace blitz {
struct NeuralModel::Impl {neural::WeightsData weights;NeuralOptions options;std::string hash;};
NeuralModel::NeuralModel(const char* file,const NeuralOptions& options) {
    if(!file||!file[0]||options.device<0||options.memory_mib<128||options.memory_mib>65536)throw std::invalid_argument("invalid neural model options");
    if(!neural_available(options.device))throw NeuralUnavailable("neural mode requires an available CUDA device and a BLITZ_CUDA build");
#ifdef BLITZ_CUDA
    auto value=std::make_unique<Impl>();value->options=options;value->weights=neural::load_weights(file,&value->hash);impl_=std::move(value);
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
    auto start=Clock::now();auto g=neural::graph(source);
    auto embedding=neural::encode_mesh_cuda(g,weights,options,settings.cancelled);
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
