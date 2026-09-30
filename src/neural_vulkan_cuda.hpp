#pragma once
#include "neural_device_view.hpp"
#include "neural_vertex_storage.hpp"
#include "blitz/neural.hpp"
#include <cuda_runtime_api.h>
namespace blitz::neural {
struct AuditPixel;
struct RasterDebugPixel;
// std430 metadata; explicit padding, shared with raster.vert.
struct DrawMetadata {float low[4],extent[4];uint32_t invalid,unused[7];};
struct DrawLayout {
    size_t position,normal,uv,color,tangent,indices,faces,metadata,bytes;
    uint32_t position_stride,normal_stride;
};
DrawLayout draw_layout(DeviceMeshView,NeuralVertexStorage);
void pack_draw(DeviceMeshView,NeuralVertexStorage,const DrawLayout&,void* device_buffer);
void check_draw_clip(DeviceMeshView,NeuralVertexStorage,const DrawLayout&,void*,const Bounds&,const Camera&,uint32_t,uint8_t,uint32_t layer=0);
void unpack_hardware(cudaSurfaceObject_t mask,cudaSurfaceObject_t attributes,cudaSurfaceObject_t colors,
    AuditPixel*,Vec3*,uint32_t size,cudaSurfaceObject_t debug=0,RasterDebugPixel* witnesses=nullptr);
}
