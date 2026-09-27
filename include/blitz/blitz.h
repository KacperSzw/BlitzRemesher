#ifndef BLITZ_REMESHER_H
#define BLITZ_REMESHER_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define BLITZ_ABI_VERSION 1u
typedef enum blitz_status { BLITZ_OK=0, BLITZ_INVALID_ARGUMENT=1, BLITZ_OUT_OF_MEMORY=2, BLITZ_INTERNAL_ERROR=3, BLITZ_CANCELLED=4, BLITZ_BUDGET_LIMITED=5 } blitz_status;
typedef struct blitz_stream { const void* data; size_t count, stride; } blitz_stream;
/* Float xyz, xyz, uv, rgba, xyzw streams. Indices are aligned uint32_t. */
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
    uint8_t levels, output_mode, chain_mode, profile; /* rebuild/reuse; direct/progressive/hybrid; coverage/normals/attributes */
    uint8_t objective, beam_width, search_supersample, audit_supersample;
    uint8_t max_supersample, prune, force_scalar, coupled_wedges;
    uint16_t candidate_budget, search_ortho, search_perspective, audit_ortho, audit_perspective;
    uint32_t search_seed, audit_seed;
    double pixels_per_meter, meters_per_unit, base_pixels, last_pixels, max_lod0_delta_px; /* optional values: zero disables */
    double normal_weight, color_weight, material_weight;
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
} blitz_lod_info;
typedef struct blitz_result blitz_result;
uint32_t blitz_abi_version(void);
blitz_status blitz_settings_init(blitz_settings*,size_t);
/* Source streams must remain alive and unchanged until result destruction. Never frees source. */
blitz_status blitz_generate(const blitz_mesh*,const blitz_settings*,blitz_result**,char* error,size_t error_capacity);
size_t blitz_result_lod_count(const blitz_result*);
blitz_status blitz_result_lod(const blitz_result*,size_t,blitz_lod_info*);
void blitz_result_destroy(blitz_result*);
#ifdef __cplusplus
}
#endif
#endif
