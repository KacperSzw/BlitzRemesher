#pragma once
#include "blitz/mesh.hpp"
namespace blitz::neural {
struct DeviceTrialStatus {
    uint32_t faces{}, invalid{};
};
// Borrowed device storage. The owner keeps every stream alive until all queued
// kernels finish. Counts describe initialized, contiguous elements, not capacity.
struct DeviceMeshView {
    const Vec3 *positions{}, *normals{};
    const Vec2* uv{};
    const ColorRGBA8* colors{};
    const Vec4* tangents{};
    const uint32_t* indices{};
    const uint16_t* materials{};
    const uint8_t* double_sided{};
    uint32_t vertices{}, faces{}, sided_count{};
    uint64_t identity{}, revision{};
    // Optional trajectory-wide quantization domain, borrowed by every trial.
    // The master streams remain FP32. Positions outside this box are invalid.
    Vec3 quant_low{}, quant_extent{};
    bool fixed_quantization{};
    // Optional exact raster reuse for a trial. Parent is a borrowed HOST view;
    // the two arrays are device storage mapping its faces into this trial.
    // All three remain valid until the owner builds another trial or commits.
    const DeviceMeshView* raster_parent{};
    const uint32_t *parent_keep{}, *parent_offsets{};
    // Trial batches use faces as capacity; this device record supplies the live
    // draw count and placement validity without a host round trip.
    const DeviceTrialStatus* trial_status{};
    const uint32_t* exact_position_bits{};
    uint16_t exact_position_bps{500};
};
} // namespace blitz::neural
