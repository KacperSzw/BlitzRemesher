#include "neural/vertex_storage.hpp"
#include <iostream>
#include <stdexcept>
using namespace blitz;
using namespace blitz::neural;
static void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}
int main() {
    try {
        for (float extent : {0.f, .01f, 16.f, 1000.f})
            for (float low : {-32.f, 0.f, 91.f})
                for (unsigned i = 0; i <= 4096; ++i) {
                    float p = low + extent * float(i) / 4096;
                    auto code = pack_unorm16(p, low, extent);
                    float decoded = unpack_unorm16(code, low, extent);
                    double bound =
                        double(extent) / 131070 + 4 * std::numeric_limits<float>::epsilon() *
                                                      std::max(std::abs(double(p)), double(extent));
                    require(std::abs(double(decoded) - p) <= bound,
                            "UNORM16 position error exceeds quantization plus arithmetic bound");
                    if (!extent)
                        require(code == 0 && decoded == low, "degenerate extent");
                }
        require(pack_unorm16(-8, -8, 16) == 0 && pack_unorm16(8, -8, 16) == 65535,
                "UV inclusive endpoints");
        require(pack_direction({}) == 0 && length(unpack_direction(0)) == 0,
                "zero normal fallback lost");
        double max_angle = 0;
        for (int i = 0; i < 4096; ++i) {
            Vec3 n = normalized(
                {float(std::sin(i * .13)), float(std::cos(i * .37)), float(std::sin(i * .71))});
            auto q = unpack_direction(pack_direction(n));
            max_angle = std::max(
                max_angle, std::acos(std::clamp(dot(n, q) / (length(n) * length(q)), -1., 1.)));
            require(length(q) > .99999 && length(q) < 1.00001, "decoded direction not normalized");
            Vec3 t = normalized(cross(n, std::abs(n.x) < .8 ? Vec3{1, 0, 0} : Vec3{0, 1, 0}));
            for (int sign : {-1, 1}) {
                auto p = pack_direction(t, sign);
                require(((p >> 31) ? -1 : 1) == sign, "tangent sign bit");
                auto tq = unpack_direction(p);
                require(std::abs(dot(q, tq)) < .004, "quantized TBN orthogonality bound");
                require(dot(cross(q, tq) * float(sign), cross(n, t) * float(sign)) > .999,
                        "quantized TBN orientation");
            }
        }
        require(max_angle < .002, "SNORM10 angular error bound");
        std::cout << "vertex storage contracts passed; maximum sampled normal error "
                  << max_angle * 180 / 3.141592653589793 << " degrees\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
