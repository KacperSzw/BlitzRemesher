#ifndef BLITZ_REMESHER_H
#define BLITZ_REMESHER_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define BLITZ_ABI_VERSION 4u
typedef enum blitz_status { BLITZ_OK=0, BLITZ_INVALID_ARGUMENT=1, BLITZ_OUT_OF_MEMORY=2, BLITZ_INTERNAL_ERROR=3, BLITZ_CANCELLED=4, BLITZ_BUDGET_LIMITED=5, BLITZ_UNAVAILABLE=6 } blitz_status;
typedef struct blitz_stream { const void* data; size_t count, stride; } blitz_stream;
typedef struct blitz_color_rgba8 { uint8_t r,g,b,a; } blitz_color_rgba8;
/* Float xyz positions/normals, float uv, linear RGBA8 colors, float xyzw tangents.
   Colors have four bytes per element (0..255); absent alpha on import is 255.
   Indices are aligned uint32_t. Version 1 float color streams are unsupported. */
typedef struct blitz_mesh {
    uint32_t struct_size, abi_version;
    blitz_stream positions, normals, uv, colors, tangents;
    const uint32_t* indices; size_t index_count;
    const uint16_t* materials; size_t material_count;
    const uint8_t* double_sided; size_t double_sided_count;
} blitz_mesh;
typedef struct blitz_curve_point { float x,y; } blitz_curve_point;
typedef struct blitz_settings {
    uint32_t struct_size, abi_version;
    uint8_t levels, profile; /* coverage/normals/attributes */
    uint16_t triangle_overhead_bps; /* 100 = 1%; range 0..10000 */
    uint8_t objective, beam_width, search_supersample, audit_supersample; /* objective: quadric/regularized/visual/topology_relaxed */
    uint8_t max_supersample, prune, force_scalar, coupled_wedges;
    uint16_t candidate_budget, search_ortho, search_perspective, audit_ortho, audit_perspective;
    uint32_t search_seed, audit_seed;
    double pixels_per_meter, meters_per_unit, base_pixels, last_pixels, max_lod0_delta_px; /* optional values: zero disables */
    double normal_weight, color_weight, material_weight;
    double max_changed_area; /* 0..1; 1 disables the coverage-area gate */
    const blitz_curve_point* transition; size_t transition_count;
    const blitz_curve_point* normal_importance; size_t normal_importance_count;
    const blitz_curve_point* attribute_importance; size_t attribute_importance_count;
    int (*cancelled)(void*); void* user_data;
} blitz_settings;
typedef struct blitz_lod_info {
    uint32_t struct_size;
    blitz_mesh mesh;
    double screen_pixels, transition_limit, source_limit, transition_error, source_error;
    uint32_t transition_worst_view, source_worst_view;
    uint8_t shared_vertices, passed;
    uint32_t reference_triangles;
    double transition_changed_area, source_changed_area; /* 1 - conservative mask IoU */
    uint32_t transition_changed_area_worst_view, source_changed_area_worst_view;
} blitz_lod_info;
typedef struct blitz_result blitz_result;
typedef struct blitz_storage_info {
    uint32_t struct_size;
    uint64_t source_vertex_bytes, added_vertex_bytes, index_bytes, total_bytes;
} blitz_storage_info;
blitz_status blitz_result_storage(const blitz_result*,blitz_storage_info*);
uint32_t blitz_abi_version(void);
blitz_status blitz_settings_init(blitz_settings*,size_t);
/* Source streams must remain alive and unchanged until result destruction. Never frees source. */
blitz_status blitz_generate(const blitz_mesh*,const blitz_settings*,blitz_result**,char* error,size_t error_capacity);
size_t blitz_result_lod_count(const blitz_result*);
/* Exact consecutive duplicates share one runtime level; scheduled audit slots remain. */
size_t blitz_result_runtime_lod_count(const blitz_result*);
/* Returns the scheduled slot, or SIZE_MAX for a null result or invalid runtime index. */
size_t blitz_result_runtime_lod_index(const blitz_result*,size_t);
blitz_status blitz_result_lod(const blitz_result*,size_t,blitz_lod_info*);
void blitz_result_destroy(blitz_result*);
/* Additive neural API; existing ABI-4 descriptors retain their sizes and layout. */
typedef struct blitz_neural_model blitz_neural_model;
typedef struct blitz_neural_options {
    uint32_t struct_size, version;
    int32_t device;
    uint32_t memory_mib;
    uint8_t overdraw_tiebreak;
    uint8_t reserved[3];
} blitz_neural_options;
typedef struct blitz_neural_info {
    uint32_t struct_size;
    uint64_t encode_ns, inference_ns, decode_ns, gpu_audit_ns, reference_audit_ns;
    uint64_t decoded, legal_collapses, rejected_collapses, reference_rejections;
    uint32_t fallback_levels;
} blitz_neural_info;
blitz_status blitz_neural_options_init(blitz_neural_options*,size_t);
blitz_status blitz_neural_model_load(const char* path,const blitz_neural_options*,blitz_neural_model**,char* error,size_t error_capacity);
void blitz_neural_model_destroy(blitz_neural_model*);
/* Borrowed hash string, valid until model destruction. */
const char* blitz_neural_model_sha256(const blitz_neural_model*);
blitz_status blitz_generate_neural(const blitz_mesh*,const blitz_settings*,const blitz_neural_model*,blitz_result**,char* error,size_t error_capacity);
blitz_status blitz_result_neural_info(const blitz_result*,blitz_neural_info*);
#ifdef __cplusplus
}
#endif
#endif
