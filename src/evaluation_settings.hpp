#pragma once

#include "blitz/evaluate.hpp"
#include <stdexcept>

namespace blitz::detail {

// Shared by CPU and GPU evaluators. Validation must not generate cameras or
// invoke a raster path merely to check a settings descriptor.
inline void validate_evaluation_settings(const Bounds& bounds, const EvalSettings& s) {
    if (!(s.screen_size > 0) || !std::isfinite(s.screen_size) || s.screen_size > 16384 ||
        !(bounds.radius > 0) || !std::isfinite(bounds.radius) || !finite(bounds.center) ||
        unsigned(s.profile) > 2 || !(s.limit >= 0) || !std::isfinite(s.limit) || !s.supersample ||
        s.max_supersample < s.supersample || s.max_supersample > 32 ||
        !std::isfinite(s.weights.normal) || !std::isfinite(s.weights.color) ||
        !std::isfinite(s.weights.material) || s.weights.normal < 0 || s.weights.color < 0 ||
        s.weights.material < 0 || s.weights.normal > 1e12 || s.weights.color > 1e12 ||
        s.weights.material > 1e12 || !std::isfinite(s.max_changed_area) || s.max_changed_area < 0 ||
        s.max_changed_area > 1)
        throw std::invalid_argument("invalid evaluation settings");
    if (!s.views.orthographic && !s.views.perspective)
        throw std::invalid_argument("at least one camera required");
}

} // namespace blitz::detail
