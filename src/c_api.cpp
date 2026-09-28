#include "blitz/blitz.h"
#include "blitz/remesher.hpp"
#include <new>
#include <stdexcept>
struct blitz_result { blitz::Result value; std::vector<uint8_t> runtime; };
namespace {
template<class T> blitz::Stream<T> stream(blitz_stream s) {blitz::Stream<T> v;v.data=static_cast<const std::byte*>(s.data);v.count=s.count;v.stride=s.stride;return v;}
template<class T> blitz_stream stream(blitz::Stream<T> s) {return {s.data,s.count,s.stride};}
void message(char* out,size_t n,const char* text) {if(out&&n){size_t k=std::min(n-1,std::strlen(text));std::memcpy(out,text,k);out[k]=0;}}
blitz::Curve curve(const blitz_curve_point* p,size_t n,blitz::Curve fallback) {
    if(!n)return fallback;if(!p||n>1024)throw std::invalid_argument("invalid curve");
    fallback.points.clear();for(size_t i=0;i<n;++i)fallback.points.push_back({p[i].x,p[i].y});return fallback;
}
blitz_mesh exported(blitz::MeshView v) {
    return {sizeof(blitz_mesh),BLITZ_ABI_VERSION,stream(v.positions),stream(v.normals),stream(v.uv),stream(v.colors),stream(v.tangents),
      v.indices.data(),v.indices.size(),v.materials.data(),v.materials.size(),v.double_sided.data(),v.double_sided.size()};
}
}
extern "C" {
uint32_t blitz_abi_version(void){return BLITZ_ABI_VERSION;}
blitz_status blitz_settings_init(blitz_settings* out,size_t n) {
    if(!out||n!=sizeof(blitz_settings))return BLITZ_INVALID_ARGUMENT;
    try {
    blitz::Settings s;*out={};out->struct_size=sizeof(*out);out->abi_version=BLITZ_ABI_VERSION;
    out->levels=s.levels;out->triangle_overhead_bps=s.triangle_overhead_bps;out->profile=uint8_t(s.profile);
    out->objective=uint8_t(s.objective);out->beam_width=s.beam_width;out->candidate_budget=s.candidate_budget;
    out->search_supersample=s.search_supersample;out->audit_supersample=s.audit_supersample;out->max_supersample=s.max_supersample;
    out->search_ortho=s.search_views.orthographic;out->search_perspective=s.search_views.perspective;out->search_seed=s.search_views.rotation_seed;
    out->audit_ortho=s.audit_views.orthographic;out->audit_perspective=s.audit_views.perspective;out->audit_seed=s.audit_views.rotation_seed;
    out->pixels_per_meter=s.pixels_per_meter;out->meters_per_unit=s.meters_per_unit;out->last_pixels=s.last_pixels;
    out->normal_weight=s.weights.normal;out->color_weight=s.weights.color;out->material_weight=s.weights.material;out->prune=s.prune;out->coupled_wedges=s.coupled_wedges;return BLITZ_OK;
    }catch(const std::bad_alloc&){*out={};return BLITZ_OUT_OF_MEMORY;}
     catch(...){*out={};return BLITZ_INTERNAL_ERROR;}
}
blitz_status blitz_generate(const blitz_mesh* m,const blitz_settings* c,blitz_result** out,char* error,size_t capacity) {
    if(out)*out=nullptr;message(error,capacity,"");
    try {
        if(!m||!c||!out||m->struct_size!=sizeof(*m)||c->struct_size!=sizeof(*c)||m->abi_version!=BLITZ_ABI_VERSION||c->abi_version!=BLITZ_ABI_VERSION)
            throw std::invalid_argument("null pointer, descriptor size or ABI version mismatch");
        if((!m->indices&&m->index_count)||(!m->materials&&m->material_count)||(!m->double_sided&&m->double_sided_count))throw std::invalid_argument("null index or material pointer");
        if((reinterpret_cast<uintptr_t>(m->indices)%alignof(uint32_t))||(reinterpret_cast<uintptr_t>(m->materials)%alignof(uint16_t)))throw std::invalid_argument("unaligned index or material stream");
        blitz::MeshView v{stream<blitz::Vec3>(m->positions),stream<blitz::Vec3>(m->normals),stream<blitz::Vec2>(m->uv),
            stream<blitz::ColorRGBA8>(m->colors),stream<blitz::Vec4>(m->tangents),{m->indices,m->index_count},{m->materials,m->material_count},{m->double_sided,m->double_sided_count}};
        blitz::Settings s;s.levels=c->levels;s.triangle_overhead_bps=c->triangle_overhead_bps;s.profile=blitz::Profile(c->profile);s.objective=blitz::Objective(c->objective);
        s.beam_width=c->beam_width;s.candidate_budget=c->candidate_budget;s.search_supersample=c->search_supersample;s.audit_supersample=c->audit_supersample;s.max_supersample=c->max_supersample;
        s.search_views={c->search_ortho,c->search_perspective,c->search_seed};s.audit_views={c->audit_ortho,c->audit_perspective,c->audit_seed};
        s.pixels_per_meter=c->pixels_per_meter;s.meters_per_unit=c->meters_per_unit;s.last_pixels=c->last_pixels;
        if(c->base_pixels!=0)s.base_pixels=c->base_pixels;if(c->max_lod0_delta_px!=0)s.max_lod0_delta_px=c->max_lod0_delta_px;
        s.weights={c->normal_weight,c->color_weight,c->material_weight};s.prune=c->prune;s.force_scalar=c->force_scalar;s.coupled_wedges=c->coupled_wedges;
        s.transition=curve(c->transition,c->transition_count,s.transition);s.normal_importance=curve(c->normal_importance,c->normal_importance_count,s.normal_importance);
        s.attribute_importance=curve(c->attribute_importance,c->attribute_importance_count,s.attribute_importance);
        if(c->cancelled)s.cancelled=[=]{return c->cancelled(c->user_data)!=0;};
        auto r=blitz::generate(v,s);auto status=r.status;auto runtime=blitz::runtime_levels(r);
        *out=new blitz_result{std::move(r),std::move(runtime)};
        return status==blitz::Status::Cancelled?BLITZ_CANCELLED:status==blitz::Status::BudgetLimited?BLITZ_BUDGET_LIMITED:BLITZ_OK;
    } catch(const std::invalid_argument& e){message(error,capacity,e.what());return BLITZ_INVALID_ARGUMENT;}
      catch(const std::bad_alloc&){message(error,capacity,"allocation failed");return BLITZ_OUT_OF_MEMORY;}
      catch(const std::exception& e){message(error,capacity,e.what());return BLITZ_INTERNAL_ERROR;}
      catch(...){message(error,capacity,"unknown exception");return BLITZ_INTERNAL_ERROR;}
}
size_t blitz_result_lod_count(const blitz_result* r){return r?r->value.lods.size():0;}
size_t blitz_result_runtime_lod_count(const blitz_result* r){return r?r->runtime.size():0;}
size_t blitz_result_runtime_lod_index(const blitz_result* r,size_t i){return r&&i<r->runtime.size()?r->runtime[i]:SIZE_MAX;}
blitz_status blitz_result_lod(const blitz_result* r,size_t i,blitz_lod_info* out) {
    if(!r||!out||out->struct_size!=sizeof(*out)||i>=r->value.lods.size())return BLITZ_INVALID_ARGUMENT;
    const auto& l=r->value.lods[i];*out={sizeof(*out),exported(l.view(r->value.source)),l.schedule.pixels,l.schedule.transition,l.schedule.source,
        l.adjacent.error,l.source_error.error,l.adjacent.worst_view,l.source_error.worst_view,uint8_t(l.shared_vertices),uint8_t(l.adjacent.passed&&l.source_error.passed),r->value.candidates[r->value.selection.reference].triangles[i]};return BLITZ_OK;
}
blitz_status blitz_result_storage(const blitz_result* r,blitz_storage_info* out) {
    if(!r||!out||out->struct_size!=sizeof(*out))return BLITZ_INVALID_ARGUMENT;
    const auto& s=r->value.candidates[r->value.selection.selected].storage;
    *out={sizeof(*out),s.source_vertex_bytes,s.added_vertex_bytes,s.index_bytes,s.total()};return BLITZ_OK;
}
void blitz_result_destroy(blitz_result* r){delete r;}
}
