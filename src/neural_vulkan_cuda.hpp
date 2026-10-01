#pragma once
#include "neural_device_view.hpp"
#include "neural_vertex_storage.hpp"
#include "blitz/neural.hpp"
#include <cuda_runtime_api.h>
namespace blitz::neural {
struct AuditPixel;
struct RasterDebugPixel;
// std430 metadata; explicit padding, shared with raster.vert.
struct DrawMetadata {float low[4],extent[4];uint32_t invalid,unused[4],bits_offset,rank_offset,exact_offset;};
enum DrawInvalid : uint32_t { DrawDomain=1, DrawPrecisionCap=2, DrawPosition=4, DrawUv=8, DrawNonfinite=16 };
struct DrawLayout {
    size_t position,normal,uv,color,tangent,indices,faces,metadata,indirect,status,bytes,exact_bits,exact_rank,exact_positions,used;
    uint32_t position_stride,normal_stride;
};
DrawLayout draw_layout(DeviceMeshView,NeuralVertexStorage,bool coverage_only=false);
void pack_draw(DeviceMeshView,NeuralVertexStorage,const DrawLayout&,void* device_buffer,bool coverage_only=false);
struct DrawStatusSources {const DrawMetadata* metadata[8]{};const DeviceTrialStatus* trial[8]{};uint32_t faces[8]{},layers[8]{},count{};};
void collect_draw_status(DrawStatusSources,uint32_t*);
void check_draw_clip(DeviceMeshView,NeuralVertexStorage,const DrawLayout&,void*,const Bounds&,const Camera&,uint32_t,uint8_t,uint32_t layer=0);
void pack_coverage_mask(cudaSurfaceObject_t,uint32_t* bits,uint32_t size);
void unpack_hardware(cudaSurfaceObject_t mask,cudaSurfaceObject_t attributes,cudaSurfaceObject_t colors,
    AuditPixel*,Vec3*,uint32_t size,cudaSurfaceObject_t debug=0,RasterDebugPixel* witnesses=nullptr);
}
