#pragma once
#include "blitz/evaluate.hpp"
#include <type_traits>
namespace blitz::neural {
// Audits never read depth or alpha. Keep depth only in the raster thread's
// registers, preserving the original float depth comparison and face ordering.
struct AuditPixel {
    Vec3 normal;
    uint16_t material;
    uint8_t covered, visible;
    __host__ __device__ AuditPixel() = default;
    __host__ __device__ explicit AuditPixel(const Pixel& p)
        : normal(p.normal), material(p.material), covered(p.covered), visible(p.visible) {}
    __host__ __device__ operator Pixel() const {
        Pixel p;
        p.normal = normal;
        p.color = {1, 1, 1, 1};
        p.material = material;
        p.covered = covered;
        p.visible = visible;
        return p;
    }
};
// Research targets only. Quantized images are never used for hard acceptance.
template <class Signed, class Unsigned, int NormalMax, int ColorMax> struct PackedPixel {
    Signed normal[3];
    Unsigned color[3];
    uint16_t material;
    uint8_t flags;
    __host__ __device__ PackedPixel() = default;
    __host__ __device__ explicit PackedPixel(const Pixel& p)
        : material(p.material), flags(uint8_t(p.covered | (p.visible << 1))) {
        normal[0] = Signed(roundf(fminf(1, fmaxf(-1, p.normal.x)) * NormalMax));
        normal[1] = Signed(roundf(fminf(1, fmaxf(-1, p.normal.y)) * NormalMax));
        normal[2] = Signed(roundf(fminf(1, fmaxf(-1, p.normal.z)) * NormalMax));
        color[0] = Unsigned(roundf(fminf(1, fmaxf(0, p.color.x)) * ColorMax));
        color[1] = Unsigned(roundf(fminf(1, fmaxf(0, p.color.y)) * ColorMax));
        color[2] = Unsigned(roundf(fminf(1, fmaxf(0, p.color.z)) * ColorMax));
    }
    __host__ __device__ operator Pixel() const {
        Pixel p;
        p.normal = {float(normal[0]) / NormalMax, float(normal[1]) / NormalMax,
                    float(normal[2]) / NormalMax};
        p.color = {float(color[0]) / ColorMax, float(color[1]) / ColorMax,
                   float(color[2]) / ColorMax, 1};
        p.material = material;
        p.covered = flags & 1;
        p.visible = (flags >> 1) & 1;
        return p;
    }
};
using Pixel16 = PackedPixel<int16_t, uint16_t, 32767, 65535>;
using Pixel8 = PackedPixel<int8_t, uint8_t, 127, 255>;
static_assert(sizeof(Pixel) == 36 && sizeof(AuditPixel) == 16 && sizeof(Pixel16) == 16 &&
              sizeof(Pixel8) == 10);
} // namespace blitz::neural
