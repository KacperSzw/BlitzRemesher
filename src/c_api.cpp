#include "blitz/blitz.h"
#include "blitz/neural.hpp"
#include "blitz/remesher.hpp"
#include <new>
#include <stdexcept>
struct blitz_result {
    blitz::Result value;
    std::vector<uint8_t> runtime;
    blitz::NeuralStats neural;
    bool is_neural{};
};
struct blitz_neural_model {
    blitz::NeuralModel value;
};
namespace {
template <class T> blitz::Stream<T> stream(blitz_stream s) {
    blitz::Stream<T> v;
    v.data = static_cast<const std::byte*>(s.data);
    v.count = s.count;
    v.stride = s.stride;
    return v;
}
template <class T> blitz_stream stream(blitz::Stream<T> s) {
    return {s.data, s.count, s.stride};
}
void message(char* out, size_t n, const char* text) {
    if (out && n) {
        size_t k = std::min(n - 1, std::strlen(text));
        std::memcpy(out, text, k);
        out[k] = 0;
    }
}
blitz::Curve curve(const blitz_curve_point* p, size_t n, blitz::Curve fallback) {
    if (!n)
        return fallback;
    if (!p || n > 1024)
        throw std::invalid_argument("invalid curve");
    fallback.points.clear();
    for (size_t i = 0; i < n; ++i)
        fallback.points.push_back({p[i].x, p[i].y});
    return fallback;
}
blitz_mesh exported(blitz::MeshView v) {
    return {sizeof(blitz_mesh),   BLITZ_ABI_VERSION,  stream(v.positions), stream(v.normals),
            stream(v.uv),         stream(v.colors),   stream(v.tangents),  v.indices.data(),
            v.indices.size(),     v.materials.data(), v.materials.size(),  v.double_sided.data(),
            v.double_sided.size()};
}
template <class Options>
blitz_status load_neural(const char* path, Options&& options, blitz_neural_model** out, char* error,
                         size_t capacity) {
    if (out)
        *out = nullptr;
    message(error, capacity, "");
    try {
        if (!out)
            throw std::invalid_argument("null model destination");
        *out = new blitz_neural_model{blitz::NeuralModel(path, options())};
        return BLITZ_OK;
    } catch (const blitz::NeuralUnavailable& e) {
        message(error, capacity, e.what());
        return BLITZ_UNAVAILABLE;
    } catch (const std::invalid_argument& e) {
        message(error, capacity, e.what());
        return BLITZ_INVALID_ARGUMENT;
    } catch (const std::bad_alloc&) {
        message(error, capacity, "allocation failed");
        return BLITZ_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        message(error, capacity, e.what());
        return BLITZ_INTERNAL_ERROR;
    } catch (...) {
        message(error, capacity, "unknown exception");
        return BLITZ_INTERNAL_ERROR;
    }
}
} // namespace
extern "C" {
uint32_t blitz_abi_version(void) {
    return BLITZ_ABI_VERSION;
}
blitz_status blitz_settings_init(blitz_settings* out, size_t n) {
    if (!out || n != sizeof(blitz_settings))
        return BLITZ_INVALID_ARGUMENT;
    try {
        blitz::Settings s;
        *out = {};
        out->struct_size = sizeof(*out);
        out->abi_version = BLITZ_ABI_VERSION;
        out->levels = s.levels;
        out->triangle_overhead_bps = s.triangle_overhead_bps;
        out->max_added_vertex_bytes_bps = s.max_added_vertex_bytes_bps.value_or(UINT32_MAX);
        out->profile = uint8_t(s.profile);
        out->objective = uint8_t(s.objective);
        out->beam_width = s.beam_width;
        out->candidate_budget = s.candidate_budget;
        out->search_supersample = s.search_supersample;
        out->audit_supersample = s.audit_supersample;
        out->max_supersample = s.max_supersample;
        out->search_ortho = s.search_views.orthographic;
        out->search_perspective = s.search_views.perspective;
        out->search_seed = s.search_views.rotation_seed;
        out->audit_ortho = s.audit_views.orthographic;
        out->audit_perspective = s.audit_views.perspective;
        out->audit_seed = s.audit_views.rotation_seed;
        out->pixels_per_meter = s.pixels_per_meter;
        out->meters_per_unit = s.meters_per_unit;
        out->last_pixels = s.last_pixels;
        out->normal_weight = s.weights.normal;
        out->color_weight = s.weights.color;
        out->material_weight = s.weights.material;
        out->max_changed_area = s.max_changed_area;
        out->prune = s.prune;
        out->coupled_wedges = s.coupled_wedges;
        return BLITZ_OK;
    } catch (const std::bad_alloc&) {
        *out = {};
        return BLITZ_OUT_OF_MEMORY;
    } catch (...) {
        *out = {};
        return BLITZ_INTERNAL_ERROR;
    }
}
static blitz_status generate_impl(const blitz_mesh* m, const blitz_settings* c,
                                  const blitz_neural_model* model, blitz_result** out, char* error,
                                  size_t capacity) {
    if (out)
        *out = nullptr;
    message(error, capacity, "");
    try {
        if (!m || !c || !out || m->struct_size != sizeof(*m) || c->struct_size != sizeof(*c) ||
            m->abi_version != BLITZ_ABI_VERSION || c->abi_version != BLITZ_ABI_VERSION)
            throw std::invalid_argument("null pointer, descriptor size or ABI version mismatch");
        if ((!m->indices && m->index_count) || (!m->materials && m->material_count) ||
            (!m->double_sided && m->double_sided_count))
            throw std::invalid_argument("null index or material pointer");
        if ((reinterpret_cast<uintptr_t>(m->indices) % alignof(uint32_t)) ||
            (reinterpret_cast<uintptr_t>(m->materials) % alignof(uint16_t)))
            throw std::invalid_argument("unaligned index or material stream");
        blitz::MeshView v{
            stream<blitz::Vec3>(m->positions), stream<blitz::Vec3>(m->normals),
            stream<blitz::Vec2>(m->uv),        stream<blitz::ColorRGBA8>(m->colors),
            stream<blitz::Vec4>(m->tangents),  {m->indices, m->index_count},
            {m->materials, m->material_count}, {m->double_sided, m->double_sided_count}};
        blitz::Settings s;
        s.levels = c->levels;
        s.triangle_overhead_bps = c->triangle_overhead_bps;
        s.max_added_vertex_bytes_bps = c->max_added_vertex_bytes_bps == UINT32_MAX
                                           ? std::nullopt
                                           : std::optional<uint32_t>(c->max_added_vertex_bytes_bps);
        s.profile = blitz::Profile(c->profile);
        s.objective = blitz::Objective(c->objective);
        s.beam_width = c->beam_width;
        s.candidate_budget = c->candidate_budget;
        s.search_supersample = c->search_supersample;
        s.audit_supersample = c->audit_supersample;
        s.max_supersample = c->max_supersample;
        s.search_views = {c->search_ortho, c->search_perspective, c->search_seed};
        s.audit_views = {c->audit_ortho, c->audit_perspective, c->audit_seed};
        s.pixels_per_meter = c->pixels_per_meter;
        s.meters_per_unit = c->meters_per_unit;
        s.last_pixels = c->last_pixels;
        if (c->base_pixels != 0)
            s.base_pixels = c->base_pixels;
        if (c->max_lod0_delta_px != 0)
            s.max_lod0_delta_px = c->max_lod0_delta_px;
        s.weights = {c->normal_weight, c->color_weight, c->material_weight};
        s.max_changed_area = c->max_changed_area;
        s.prune = c->prune;
        s.force_scalar = c->force_scalar;
        s.coupled_wedges = c->coupled_wedges;
        s.transition = curve(c->transition, c->transition_count, s.transition);
        s.normal_importance =
            curve(c->normal_importance, c->normal_importance_count, s.normal_importance);
        s.attribute_importance =
            curve(c->attribute_importance, c->attribute_importance_count, s.attribute_importance);
        if (c->cancelled)
            s.cancelled = [=] { return c->cancelled(c->user_data) != 0; };
        blitz::NeuralStats stats;
        auto r = model ? blitz::generate_neural(v, s, model->value, &stats) : blitz::generate(v, s);
        auto status = r.status;
        auto runtime = blitz::runtime_levels(r);
        *out = new blitz_result{std::move(r), std::move(runtime), stats, model != nullptr};
        return status == blitz::Status::Cancelled       ? BLITZ_CANCELLED
               : status == blitz::Status::BudgetLimited ? BLITZ_BUDGET_LIMITED
                                                        : BLITZ_OK;
    } catch (const blitz::NeuralUnavailable& e) {
        message(error, capacity, e.what());
        return BLITZ_UNAVAILABLE;
    } catch (const std::invalid_argument& e) {
        message(error, capacity, e.what());
        return BLITZ_INVALID_ARGUMENT;
    } catch (const std::bad_alloc&) {
        message(error, capacity, "allocation failed");
        return BLITZ_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        message(error, capacity, e.what());
        return BLITZ_INTERNAL_ERROR;
    } catch (...) {
        message(error, capacity, "unknown exception");
        return BLITZ_INTERNAL_ERROR;
    }
}
blitz_status blitz_generate(const blitz_mesh* m, const blitz_settings* c, blitz_result** out,
                            char* error, size_t capacity) {
    return generate_impl(m, c, nullptr, out, error, capacity);
}
blitz_status blitz_neural_options_init(blitz_neural_options* out, size_t n) {
    if (!out || n != sizeof(*out))
        return BLITZ_INVALID_ARGUMENT;
    *out = {sizeof(*out), 1, 0, 6144, 1, {0, 0, 0}};
    return BLITZ_OK;
}
blitz_status blitz_neural_model_load(const char* path, const blitz_neural_options* options,
                                     blitz_neural_model** out, char* error, size_t capacity) {
    return load_neural(
        path,
        [&] {
            if (!options || options->struct_size != sizeof(*options) || options->version != 1 ||
                options->overdraw_tiebreak > 1 || options->reserved[0] || options->reserved[1] ||
                options->reserved[2])
                throw std::invalid_argument("invalid neural descriptor");
            return blitz::NeuralOptions{options->device, options->memory_mib,
                                        bool(options->overdraw_tiebreak)};
        },
        out, error, capacity);
}
blitz_status blitz_neural_options_v2_init(blitz_neural_options_v2* out, size_t n) {
    if (!out || n != sizeof(*out))
        return BLITZ_INVALID_ARGUMENT;
    blitz::NeuralOptions o;
    *out = {};
    out->struct_size = sizeof(*out);
    out->version = 2;
    out->device = o.device;
    out->memory_mib = o.memory_mib;
    out->action_trials = o.action_trials;
    out->ranking_seed = o.ranking_seed;
    out->exact_position_bps = o.exact_position_bps;
    out->action_batch = o.action_batch;
    out->origin = uint8_t(o.origin);
    out->ranking = uint8_t(o.ranking);
    out->confirmation = uint8_t(o.confirmation);
    out->raster_backend = uint8_t(o.raster_backend);
    out->vertex_storage = uint8_t(o.vertex_storage);
    out->preserve_uv = o.preserve_uv;
    out->overdraw_tiebreak = o.overdraw_tiebreak;
    out->cache_rasters = o.cache_rasters;
    out->mask_only_coverage = o.mask_only_coverage;
    out->direct_targets = o.direct_targets;
    out->view_batch = o.view_batch;
    out->candidate_batch = o.candidate_batch;
    return BLITZ_OK;
}
blitz_status blitz_neural_model_load_v2(const char* path, const blitz_neural_options_v2* c,
                                        blitz_neural_model** out, char* error, size_t capacity) {
    return load_neural(
        path,
        [&] {
            if (!c || c->struct_size != sizeof(*c) || c->version != 2 || c->reserved[0] ||
                c->reserved[1] || c->reserved[2] || c->preserve_uv > 1 ||
                c->overdraw_tiebreak > 1 || c->cache_rasters > 1 || c->mask_only_coverage > 1 ||
                c->direct_targets > 1)
                throw std::invalid_argument("invalid neural v2 descriptor");
            blitz::NeuralOptions o;
            o.device = c->device;
            o.memory_mib = c->memory_mib;
            o.action_trials = c->action_trials;
            o.ranking_seed = c->ranking_seed;
            o.exact_position_bps = c->exact_position_bps;
            o.action_batch = c->action_batch;
            o.origin = blitz::NeuralOrigin(c->origin);
            o.ranking = blitz::NeuralRanking(c->ranking);
            o.confirmation = blitz::NeuralConfirmation(c->confirmation);
            o.raster_backend = blitz::NeuralRasterBackend(c->raster_backend);
            o.vertex_storage = blitz::NeuralVertexStorage(c->vertex_storage);
            o.preserve_uv = c->preserve_uv;
            o.overdraw_tiebreak = c->overdraw_tiebreak;
            o.cache_rasters = c->cache_rasters;
            o.mask_only_coverage = c->mask_only_coverage;
            o.direct_targets = c->direct_targets;
            o.view_batch = c->view_batch;
            o.candidate_batch = c->candidate_batch;
            return o;
        },
        out, error, capacity);
}
void blitz_neural_model_destroy(blitz_neural_model* model) {
    delete model;
}
const char* blitz_neural_model_sha256(const blitz_neural_model* model) {
    return model ? model->value.sha256().c_str() : nullptr;
}
blitz_status blitz_generate_neural(const blitz_mesh* m, const blitz_settings* c,
                                   const blitz_neural_model* model, blitz_result** out, char* error,
                                   size_t capacity) {
    if (!model) {
        if (out)
            *out = nullptr;
        message(error, capacity, "neural model is required");
        return BLITZ_INVALID_ARGUMENT;
    }
    return generate_impl(m, c, model, out, error, capacity);
}
blitz_status blitz_result_neural_info(const blitz_result* result, blitz_neural_info* out) {
    if (!result || !out || out->struct_size != sizeof(*out) || !result->is_neural)
        return BLITZ_INVALID_ARGUMENT;
    auto& s = result->neural;
    *out = {sizeof(*out),
            s.encode_ns,
            s.inference_ns,
            s.decode_ns,
            s.gpu_audit_ns,
            s.reference_audit_ns,
            s.decoded,
            s.legal_collapses,
            s.rejected_collapses,
            s.reference_rejections,
            s.fallback_levels};
    return BLITZ_OK;
}
size_t blitz_result_neural_action_proposal_count(const blitz_result* result) {
    return result && result->is_neural ? result->neural.action_proposals.size() : 0;
}
blitz_status blitz_result_neural_action_proposal(const blitz_result* result, size_t index,
                                                 blitz_neural_action_proposal* out) {
    if (!result || !result->is_neural || !out || out->struct_size != sizeof(*out) ||
        out->version != 1 || index >= result->neural.action_proposals.size())
        return BLITZ_INVALID_ARGUMENT;
    const auto& p = result->neural.action_proposals[index];
    *out = {sizeof(*out),
            1,
            p.start_triangles,
            p.target_triangles,
            p.final_triangles,
            p.trial_budget,
            p.trials,
            p.accepted_batches,
            p.screen_pixels,
            uint8_t(p.origin),
            uint8_t(p.output),
            uint8_t(p.stop_reason),
            {}};
    return BLITZ_OK;
}
size_t blitz_result_lod_count(const blitz_result* r) {
    return r ? r->value.lods.size() : 0;
}
size_t blitz_result_runtime_lod_count(const blitz_result* r) {
    return r ? r->runtime.size() : 0;
}
size_t blitz_result_runtime_lod_index(const blitz_result* r, size_t i) {
    return r && i < r->runtime.size() ? r->runtime[i] : SIZE_MAX;
}
blitz_status blitz_result_lod(const blitz_result* r, size_t i, blitz_lod_info* out) {
    if (!r || !out || out->struct_size != sizeof(*out) || i >= r->value.lods.size())
        return BLITZ_INVALID_ARGUMENT;
    const auto& l = r->value.lods[i];
    *out = {sizeof(*out),
            exported(l.view(r->value.source)),
            l.schedule.pixels,
            l.schedule.transition,
            l.schedule.source,
            l.adjacent.error,
            l.source_error.error,
            l.adjacent.worst_view,
            l.source_error.worst_view,
            uint8_t(l.shared_vertices),
            uint8_t(l.adjacent.passed && l.source_error.passed),
            r->value.candidates[r->value.selection.reference].triangles[i],
            l.adjacent.changed_area,
            l.source_error.changed_area,
            l.adjacent.changed_area_worst_view,
            l.source_error.changed_area_worst_view};
    return BLITZ_OK;
}
blitz_status blitz_result_storage(const blitz_result* r, blitz_storage_info* out) {
    if (!r || !out || out->struct_size != sizeof(*out))
        return BLITZ_INVALID_ARGUMENT;
    const auto& s = r->value.candidates[r->value.selection.selected].storage;
    *out = {sizeof(*out),
            s.source_vertex_bytes,
            s.added_vertex_bytes,
            s.index_bytes,
            s.total(),
            r->value.added_vertex_budget_bytes.value_or(UINT64_MAX),
            r->value.max_added_vertex_bytes_bps.value_or(UINT32_MAX)};
    return BLITZ_OK;
}
blitz_status blitz_result_runtime_lod_storage(const blitz_result* r, size_t i,
                                              blitz_runtime_lod_storage_info* out) {
    if (!r || !out || out->struct_size != sizeof(*out) || i >= r->runtime.size())
        return BLITZ_INVALID_ARGUMENT;
    uint64_t cumulative = 0;
    for (size_t j = 0; j <= i; ++j) {
        auto index = r->runtime[j];
        const auto& lod = r->value.lods[index];
        auto added = blitz::added_vertex_bytes(lod, r->value.source);
        if (lod.source_prefix_vertices)
            for (size_t k = 0; k < j; ++k) {
                const auto& previous = r->value.lods[r->runtime[k]];
                if (previous.source_prefix_vertices && previous.vertex_pool == lod.vertex_pool) {
                    added = 0;
                    break;
                }
            }
        cumulative += added;
        if (j == i)
            *out = {sizeof(*out), index, added, uint64_t(lod.data.indices.size()) * 4, cumulative};
    }
    return BLITZ_OK;
}
void blitz_result_destroy(blitz_result* r) {
    delete r;
}
}
