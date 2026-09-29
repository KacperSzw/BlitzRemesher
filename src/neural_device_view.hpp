#pragma once
#include "blitz/mesh.hpp"
namespace blitz::neural {
// Borrowed device storage. The owner keeps every stream alive until all queued
// kernels finish. Counts describe initialized, contiguous elements, not capacity.
struct DeviceMeshView {
    const Vec3 *positions{},*normals{};
    const Vec2* uv{};
    const ColorRGBA8* colors{};
    const Vec4* tangents{};
    const uint32_t* indices{};
    const uint16_t* materials{};
    const uint8_t* double_sided{};
    uint32_t vertices{},faces{},sided_count{};
    uint64_t identity{},revision{};
};
}
