#include "chain_hooks.hpp"
#include "neural/action.hpp"
#include "neural/action_cache.hpp"
#include "neural/action_gpu.hpp"
#include "neural/internal.hpp"
#include "neural/memory.hpp"
#include "neural/quantization.hpp"
#include <chrono>
#include <random>
namespace blitz {
struct NeuralModel::Impl {
    neural::WeightsData weights;
    NeuralOptions options;
    std::string hash;
};
NeuralModel::NeuralModel(const char* file, const NeuralOptions& options) {
    if (!file || !file[0] || options.device < 0 || options.memory_mib < 128 ||
        options.memory_mib > 65536 || !options.action_trials || options.action_trials > 65536 ||
        options.ranking > NeuralRanking::CurrentPlane)
        throw std::invalid_argument("invalid neural model options");
    if (!options.action_batch || options.action_batch > 64)
        throw std::invalid_argument("action batch outside 1..64");
    if (options.exact_position_bps > 10000 || options.view_batch < 1 || options.view_batch > 4 ||
        (options.candidate_batch != 1 && options.candidate_batch != 2 &&
         options.candidate_batch != 4 && options.candidate_batch != 8))
        throw std::invalid_argument("invalid neural precision or audit batch");
    if (options.confirmation > NeuralConfirmation::Compare)
        throw std::invalid_argument("invalid neural confirmation backend");
    if (options.origin > NeuralOrigin::Both)
        throw std::invalid_argument("invalid neural proposal origin");
    if (options.raster_backend > NeuralRasterBackend::Vulkan ||
        options.vertex_storage > NeuralVertexStorage::Automatic)
        throw std::invalid_argument("invalid neural raster or storage selection");
    if (options.raster_backend == NeuralRasterBackend::Cuda &&
        options.draw_storage() != NeuralVertexStorage::Float32)
        throw std::invalid_argument("packed draw storage requires Vulkan");
    if (options.raster_backend == NeuralRasterBackend::Vulkan &&
        options.confirmation != NeuralConfirmation::Gpu)
        throw std::invalid_argument("Vulkan raster semantics require GPU confirmation; use the "
                                    "CUDA backend for CPU comparison");
    if (!neural_available(options.device))
        throw NeuralUnavailable(
            "neural mode requires an available CUDA device and a BLITZ_CUDA build");
#ifdef BLITZ_CUDA
    auto value = std::make_unique<Impl>();
    value->options = options;
    value->weights = neural::load_weights(file, &value->hash);
    impl_ = std::move(value);
    if (impl_->weights.architecture == neural::schema && options.ranking != NeuralRanking::Learned)
        throw std::invalid_argument("action ranking controls require an action policy");
    if (impl_->weights.architecture == neural::schema && options.origin != NeuralOrigin::Source)
        throw std::invalid_argument("predecessor origins require an action policy");
    if (!options.preserve_uv && impl_->weights.architecture != neural::conditioned_placement_schema)
        throw std::invalid_argument("UV relaxation requires a UV-conditioned v4 policy");
#endif
}
NeuralModel::~NeuralModel() = default;
NeuralModel::NeuralModel(NeuralModel&&) noexcept = default;
NeuralModel& NeuralModel::operator=(NeuralModel&&) noexcept = default;
const std::string& NeuralModel::sha256() const {
    if (!impl_)
        throw std::invalid_argument("moved-from neural model");
    return impl_->hash;
}
#ifndef BLITZ_CUDA
bool neural_available(int32_t) noexcept {
    return false;
}
Measurement evaluate_cuda(MeshView, MeshView, const Bounds&, const EvalSettings&,
                          const NeuralOptions&, NeuralStats*) {
    throw NeuralUnavailable("CUDA evaluator was not built");
}
Measurement evaluate_gpu(MeshView, MeshView, const Bounds&, const EvalSettings&,
                         const NeuralOptions&, NeuralStats*, MeshView) {
    throw NeuralUnavailable("GPU evaluator was not built");
}
double overlap_cuda(MeshView, const Bounds&, double, ViewSet, const NeuralOptions&) {
    throw NeuralUnavailable("CUDA evaluator was not built");
}
#endif
#ifdef BLITZ_CUDA
std::vector<float> neural::encode_mesh_cuda(const Graph& g, const WeightsData& weights,
                                            const NeuralOptions& options,
                                            const std::function<bool()>& cancelled) {
    std::vector<float> embedding(g.size() * neural::hidden);
    std::vector<uint32_t> core;
    // Halos retain the full three-layer receptive field. Split until each patch fits.
    std::function<void(std::span<const uint32_t>)> encode = [&](std::span<const uint32_t> ids) {
        if (cancelled && cancelled())
            return;
        try {
            auto p = neural::patch(g, ids);
            auto values = neural::encode_cuda(p.graph, weights, options);
            for (uint32_t i = 0; i < p.core; ++i)
                std::copy_n(values.data() + size_t(i) * neural::hidden, neural::hidden,
                            embedding.data() + size_t(p.ids[i]) * neural::hidden);
        } catch (const std::length_error&) {
            if (ids.size() < 2)
                throw;
            encode(ids.first(ids.size() / 2));
            encode(ids.subspan(ids.size() / 2));
        }
    };
    for (uint32_t i = 0; i < g.size(); ++i) {
        core.push_back(i);
        if (core.size() == 16384) {
            encode(core);
            core.clear();
        }
    }
    if (!core.empty())
        encode(core);
    return embedding;
}
#endif
Result generate_neural(MeshView source, const Settings& settings, const NeuralModel& model,
                       NeuralStats* stats) {
    if (stats)
        *stats = {};
    if (!model.impl_)
        throw std::invalid_argument("invalid neural model");
#ifndef BLITZ_CUDA
    (void)source;
    (void)settings;
    throw NeuralUnavailable("neural mode was not built");
#else
    if (auto e = validate(source); !e.empty())
        throw std::invalid_argument(e);
    if (auto e = validate(settings); !e.empty())
        throw std::invalid_argument(e);
    if (settings.research.component_candidates || settings.research.independent_seams ||
        settings.research.topology_fallback || settings.research.graph_passes ||
        settings.research.appearance_stage != AppearanceStage::Off ||
        settings.research.density_targets || settings.research.merge_wedges ||
        settings.research.shared_rebuild || settings.research.conservative_screen)
        throw std::invalid_argument("CPU research proposal options are unsupported in neural mode");
    Settings s = settings;
    bool cancellation_seen = false;
    if (settings.cancelled)
        s.cancelled = [&] {
            if (!cancellation_seen)
                cancellation_seen = settings.cancelled();
            return cancellation_seen;
        };
    NeuralStats local;
    auto& counters = stats ? *stats : local;
    auto& options = model.impl_->options;
    auto& weights = model.impl_->weights;
    neural::MemoryScope memory(options);
    neural::AuditSession session(options);
    using Clock = std::chrono::steady_clock;
    auto nanos = [](auto start) {
        return uint64_t(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
    };
    auto start = Clock::now();
    neural::Graph g;
    std::vector<float> embedding;
    std::unique_ptr<neural::ActionCuda> action_network;
    if (weights.architecture == neural::schema) {
        g = neural::graph(source);
        embedding = neural::encode_mesh_cuda(g, weights, options, s.cancelled);
    } else
        action_network = std::make_unique<neural::ActionCuda>(weights, options);
    counters.encode_ns = nanos(start);
    // All proposals use the shared source encoding. The conditioned head predicts a
    // complete retention/representative field for each requested size and target.
    s.research.chain = options.origin == NeuralOrigin::Source     ? ChainMode::Direct
                       : options.origin == NeuralOrigin::Previous ? ChainMode::Progressive
                                                                  : ChainMode::Hybrid;
    counters.action_diagnostics_version = action_network ? 1 : 0;
    detail::GenerationHooks hooks;
    hooks.source_fallback_requires_audit = options.draw_storage() != NeuralVertexStorage::Float32;
    neural::AuditCuda audit(options, source);
    hooks.propose = [&](MeshView, const ReduceSettings& rs, const EvalSettings& config,
                        double adjacent) {
        auto e = config;
        e.max_changed_area = s.max_changed_area;
        auto begin = Clock::now();
        auto prediction = neural::predict_cuda(
            embedding,
            neural::condition(e, adjacent, double(rs.target_triangles) / source.triangles()),
            weights, options);
        counters.inference_ns += nanos(begin);
        begin = Clock::now();
        neural::DecodeStats decoded;
        auto candidate = neural::decode(source, g, prediction, rs.target_triangles, rs.output,
                                        &decoded, s.cancelled);
        counters.decode_ns += nanos(begin);
        ++counters.decoded;
        counters.legal_collapses += decoded.accepted;
        counters.rejected_collapses += decoded.rejected;
        return candidate;
    };
    hooks.evaluate = [&](MeshView a, MeshView b, const Bounds& bounds, const EvalSettings& e) {
        auto begin = Clock::now();
        auto bounded = e;
        bounded.max_supersample = neural::bounded_refinement(e);
        if (bounded.max_supersample < e.max_supersample)
            ++counters.bounded_audits;
        auto m = audit.evaluate(a, b, bounds, bounded, &counters);
        counters.gpu_audit_ns += nanos(begin);
        return m;
    };
    std::unique_ptr<neural::GpuActionState> action_state, placement_state;
    Mesh action_input, placement_input;
    auto quantization = neural::vertex_bounds(source);
    Mesh baseline_mesh, baseline_previous;
    std::array<EvalSettings, 4> baseline_settings;
    uint64_t baseline_revision = 0;
    struct PackingBaselineRejected : std::runtime_error {
        bool resource_limited;
        explicit PackingBaselineRejected(bool resource)
            : std::runtime_error("packed baseline failed reference audit"),
              resource_limited(resource) {}
    };
    auto packed_baseline = [&](MeshView previous, const EvalSettings& source_eval,
                               const EvalSettings& adjacent_eval, const EvalSettings& search_source,
                               const EvalSettings& search_adjacent) -> const Mesh& {
        std::array<EvalSettings, 4> settings{source_eval, adjacent_eval, search_source,
                                             search_adjacent};
        bool cached = baseline_revision && same_mesh_data(previous, baseline_previous.view());
        for (unsigned i = 0; i < 4 && cached; ++i)
            cached = neural::same_packing_settings(settings[i], baseline_settings[i]);
        if (!cached) {
            // Failed repairs must not associate the previous successful mesh
            // with the new reference/settings key.
            baseline_revision = 0;
            placement_state.reset();
            baseline_previous = copy_mesh(previous);
            baseline_settings = settings;
            std::array<neural::PackingAudit, 4> checks{{{source, settings[0]},
                                                        {baseline_previous.view(), settings[1]},
                                                        {source, settings[2]},
                                                        {baseline_previous.view(), settings[3]}}};
            auto fixed = neural::repair_packing_gpu(source, options, checks, 256);
            counters.packing_trials += fixed.trials;
            counters.packing_changed_vertices += fixed.changed_vertices;
            counters.packing_failures += !fixed.final.passed;
            if (!fixed.final.complete || !fixed.final.passed) {
                if (options.capture_confirmation_failure) {
                    auto& f = counters.confirmation_failure.emplace();
                    auto& check = checks[fixed.failed_check];
                    f.source = copy_mesh(source);
                    f.reference = copy_mesh(check.reference);
                    f.candidate = std::move(fixed.mesh);
                    f.bounds = bounds(source);
                    f.settings = check.settings;
                    f.settings.cancelled = {};
                    f.settings.performance = nullptr;
                    f.gpu = fixed.final;
                    f.backend = NeuralConfirmation::Gpu;
                    f.raster = options.raster_backend;
                    f.storage = options.draw_storage();
                    f.exact_position_bps = options.exact_position_bps;
                    f.stage = NeuralAuditStage::PackingBaseline;
                    f.adjacent = fixed.failed_check % 2;
                    f.reason = fixed.final.resource_limited ? NeuralConfirmationReason::Resource
                                                            : NeuralConfirmationReason::Visual;
                }
                throw PackingBaselineRejected(fixed.final.resource_limited);
            }
            baseline_mesh = std::move(fixed.mesh);
            ++baseline_revision;
        }
        return baseline_mesh;
    };
    if (action_network && neural::is_placement_schema(weights.architecture) &&
        options.draw_storage() != NeuralVertexStorage::Float32 &&
        s.research.output != OutputMode::Reuse)
        hooks.fallback = [&](MeshView previous, const EvalSettings& a, const EvalSettings& b,
                             const EvalSettings& sa, const EvalSettings& sb) -> std::optional<Lod> {
            try {
                Lod lod;
                lod.shared_vertices = false;
                lod.data = packed_baseline(previous, a, b, sa, sb);
                return lod;
            } catch (const PackingBaselineRejected& error) {
                if (error.resource_limited && !(s.cancelled && s.cancelled()))
                    throw;
                return std::nullopt;
            }
        };
    if (action_network)
        hooks.propose_guarded = [&](MeshView input, MeshView fixed_source, MeshView previous,
                                    const Bounds& bounds, const ReduceSettings& rs,
                                    const EvalSettings& source_eval,
                                    const EvalSettings& adjacent_eval) {
            audit.bind_reference(previous);
            struct ReferenceScope {
                neural::AuditCuda& audit;
                ~ReferenceScope() {
                    audit.clear_reference();
                }
            } reference_scope{audit};
            auto begin = Clock::now();
            auto nested_before = counters.inference_ns + counters.gpu_audit_ns;
            neural::ActionStats stats;
            const bool direct = same_mesh_data(input, source),
                       free = neural::is_placement_schema(weights.architecture) &&
                              rs.output == OutputMode::Rebuild;
            NeuralActionProposal diagnostic;
            diagnostic.origin = options.origin == NeuralOrigin::Previous || !direct
                                    ? NeuralOrigin::Previous
                                    : NeuralOrigin::Source;
            diagnostic.output = rs.output;
            diagnostic.screen_pixels = source_eval.screen_size;
            diagnostic.start_triangles = uint32_t(input.triangles());
            diagnostic.target_triangles = uint32_t(rs.target_triangles);
            diagnostic.trial_budget = options.action_trials;
            bool resource = false;
            Lod candidate;
            try {
                auto initial = input;
                if (free && direct && options.draw_storage() != NeuralVertexStorage::Float32) {
                    auto search = source_eval, search_adjacent = adjacent_eval;
                    for (auto* e : {&search, &search_adjacent}) {
                        e->views = s.search_views;
                        e->supersample = s.search_supersample;
                        e->max_changed_area = 1;
                    }
                    initial = packed_baseline(previous, source_eval, adjacent_eval, search,
                                              search_adjacent)
                                  .view();
                }
                auto& owner = free ? placement_state : action_state;
                auto& retained = free ? placement_input : action_input;
                if (!owner || !same_mesh_data(initial, retained.view())) {
                    owner.reset();
                    retained = copy_mesh(initial);
                    owner = std::make_unique<neural::GpuActionState>(retained.view(), options, free,
                                                                     &quantization, fixed_source);
                }
                auto* state = owner.get();
                state->reset();
                auto evaluate_device = [&](MeshView reference, neural::DeviceMeshView candidate,
                                           const EvalSettings& config) {
                    auto t = Clock::now();
                    auto bounded = config;
                    bounded.max_supersample = neural::bounded_refinement(config);
                    if (bounded.max_supersample < config.max_supersample)
                        ++counters.bounded_audits;
                    auto result = audit.evaluate(reference, candidate, bounds, bounded, &counters);
                    resource |= result.resource_limited;
                    counters.gpu_audit_ns += nanos(t);
                    return result;
                };
                auto gate = [&](neural::DeviceMeshView candidate) {
                    if (s.cancelled && s.cancelled())
                        return false;
                    auto a = evaluate_device(fixed_source, candidate, source_eval);
                    if (!a.complete || !a.passed)
                        return false;
                    auto b = evaluate_device(previous, candidate, adjacent_eval);
                    return b.complete && b.passed;
                };
                if (!direct && !gate(state->view())) {
                    candidate.shared_vertices = false;
                    candidate.data = copy_mesh(previous);
                    stats.stop_reason = NeuralActionStop::InfeasibleSeed;
                } else
                    candidate = state->execute(
                        neural::condition(source_eval, adjacent_eval.limit,
                                          double(rs.target_triangles) / fixed_source.triangles()),
                        rs.target_triangles, options.action_trials, action_network.get(),
                        options.ranking, options.ranking_seed, options.action_batch, gate, &stats,
                        s.cancelled);
            } catch (const neural::gpu::ResourceError&) {
                diagnostic.stop_reason = NeuralActionStop::Resource;
                diagnostic.final_triangles = diagnostic.start_triangles;
                counters.action_proposals.push_back(diagnostic);
                throw;
            } catch (const PackingBaselineRejected& error) {
                resource |= error.resource_limited;
                candidate.shared_vertices = false;
                candidate.data = copy_mesh(previous);
                stats.stop_reason = s.cancelled && s.cancelled() ? NeuralActionStop::Cancelled
                                                                 : NeuralActionStop::InfeasibleSeed;
            }
            diagnostic.final_triangles = uint32_t(candidate.data.indices.size() / 3);
            diagnostic.trials = uint32_t(stats.trials);
            diagnostic.accepted_batches = stats.accepted_batches;
            diagnostic.stop_reason = resource ? NeuralActionStop::Resource : stats.stop_reason;
            counters.action_proposals.push_back(diagnostic);
            counters.inference_ns += stats.inference_ns;
            if (rs.output == OutputMode::Rebuild && candidate.shared_vertices) {
                candidate.data = copy_mesh(candidate.view(input));
                compact(candidate.data);
                candidate.shared_vertices = false;
            }
            auto elapsed = nanos(begin),
                 nested = counters.inference_ns + counters.gpu_audit_ns - nested_before;
            counters.decode_ns += elapsed - std::min(elapsed, nested);
            ++counters.decoded;
            counters.legal_collapses += stats.accepted;
            counters.rejected_collapses += stats.rejected;
            counters.action_ranked += stats.ranked;
            counters.action_trials += stats.trials;
            return candidate;
        };
    hooks.confirm = [&](Result& r) {
        bool good = true;
        auto confirm = [&](MeshView reference, MeshView candidate, const EvalSettings& e,
                           uint32_t level, bool adjacent, Measurement& output) {
            auto begin = Clock::now();
            Measurement cpu, gpu;
            if (options.confirmation != NeuralConfirmation::Cpu) {
                auto t = Clock::now();
                gpu = audit.evaluate(reference, candidate, r.reference_bounds, e, &counters);
                counters.gpu_confirmation_ns += nanos(t);
            }
            if (options.confirmation != NeuralConfirmation::Gpu) {
                auto t = Clock::now();
                cpu = evaluate(reference, candidate, r.reference_bounds, e);
                counters.reference_audit_ns += nanos(t);
            }
            output = options.confirmation == NeuralConfirmation::Cpu ? cpu : gpu;
            auto close = [](double a, double b) {
                return a == b || (std::isfinite(a) && std::isfinite(b) && std::abs(a - b) <= 1e-5);
            };
            bool agrees = options.confirmation != NeuralConfirmation::Compare ||
                          (cpu.passed == gpu.passed && cpu.complete == gpu.complete &&
                           cpu.resource_limited == gpu.resource_limited &&
                           cpu.views_evaluated == gpu.views_evaluated &&
                           close(cpu.error, gpu.error) && close(cpu.coverage, gpu.coverage) &&
                           close(cpu.coverage_upper, gpu.coverage_upper) &&
                           cpu.changed_area == gpu.changed_area &&
                           close(cpu.normal_degrees, gpu.normal_degrees));
            bool stopped = output.cancelled || cpu.cancelled || gpu.cancelled ||
                           (s.cancelled && s.cancelled());
            auto invalid = [](const Measurement& m) {
                return std::isnan(m.error) || std::isnan(m.coverage) ||
                       std::isnan(m.coverage_upper) || !std::isfinite(m.changed_area) ||
                       !std::isfinite(m.normal_degrees) ||
                       (m.passed && (!std::isfinite(m.error) || !std::isfinite(m.coverage) ||
                                     !std::isfinite(m.coverage_upper)));
            };
            bool nonfinite = invalid(output) ||
                             (options.confirmation == NeuralConfirmation::Compare && invalid(cpu));
            bool resource =
                output.resource_limited ||
                (options.confirmation == NeuralConfirmation::Compare && cpu.resource_limited);
            if (!stopped && !nonfinite && agrees && output.complete && output.passed)
                return true;
            NeuralConfirmationReason reason;
            if (stopped) {
                reason = NeuralConfirmationReason::Cancelled;
                ++counters.confirmation_cancelled;
            } else if (resource) {
                reason = NeuralConfirmationReason::Resource;
                ++counters.confirmation_resources;
            } else if (nonfinite) {
                reason = NeuralConfirmationReason::Nonfinite;
                ++counters.confirmation_nonfinite;
            } else if (!agrees) {
                reason = NeuralConfirmationReason::Disagreement;
                ++counters.confirmation_disagreements;
            } else {
                reason = NeuralConfirmationReason::Visual;
                ++counters.reference_rejections;
            }
            if (!counters.confirmation_failure) {
                auto& failure = counters.confirmation_failure.emplace();
                failure.exact_position_bps = options.exact_position_bps;
                failure.bounds = r.reference_bounds;
                failure.settings = e;
                failure.settings.cancelled = {};
                failure.settings.performance = nullptr;
                failure.cpu = cpu;
                failure.gpu = gpu;
                failure.level = level;
                failure.backend = options.confirmation;
                failure.reason = reason;
                failure.adjacent = adjacent;
                failure.nanoseconds = nanos(begin);
                failure.raster = options.raster_backend;
                failure.storage = options.draw_storage();
                if (options.capture_confirmation_failure) {
                    failure.reference = copy_mesh(reference);
                    failure.candidate = copy_mesh(candidate);
                    failure.source = copy_mesh(source);
                }
            }
            output.complete = false;
            output.passed = false;
            return false;
        };
        for (size_t i = 1; i < r.lods.size() && good; ++i) {
            auto& lod = r.lods[i];
            EvalSettings e;
            e.profile = s.profile;
            e.weights = s.weights;
            e.views = s.audit_views;
            e.supersample = s.audit_supersample;
            e.max_supersample = s.max_supersample;
            e.screen_size = lod.schedule.pixels;
            e.max_changed_area = s.max_changed_area;
            e.force_scalar = s.force_scalar;
            e.cancelled = s.cancelled;
            double t = s.levels == 2 ? 0 : double(i - 1) / (s.levels - 2);
            e.weights.normal *= s.normal_importance.at(t);
            e.weights.color *= s.attribute_importance.at(t);
            e.weights.material *= s.attribute_importance.at(t);
            e.limit = lod.schedule.source;
            good = confirm(source, lod.view(source), e, uint32_t(i), false, lod.source_error);
            if (good) {
                e.limit = lod.schedule.transition;
                good = confirm(r.lods[i - 1].view(source), lod.view(source), e, uint32_t(i), true,
                               lod.adjacent);
            }
        }
        return good;
    };
    if (options.overdraw_tiebreak)
        hooks.tiebreak = [&](const Result& r) {
            double total = 0;
            for (size_t i = 1; i < r.lods.size(); ++i)
                total += overlap_cuda(r.lods[i].view(source), r.reference_bounds,
                                      r.lods[i].schedule.pixels, s.search_views, options);
            return total;
        };
    auto result = detail::generate_with_hooks(source, s, {}, &hooks);
    for (size_t i = 1; i < result.lods.size(); ++i)
        if (same_mesh_data(result.lods[i].view(source), result.lods[i - 1].view(source)))
            ++counters.fallback_levels;
    return result;
#endif
}
} // namespace blitz
