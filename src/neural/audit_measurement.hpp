#pragma once
#include <blitz/evaluate.hpp>
#include <cmath>

namespace blitz::neural {
// A witnessed distance failure may be infinite (for example, an empty render
// against a nonempty reference). Invalid diagnostics never establish a verdict.
inline bool audit_measurement_nonfinite(const Measurement& m) {
    return std::isnan(m.error) || std::isnan(m.coverage) || std::isnan(m.coverage_upper) ||
           !std::isfinite(m.changed_area) || !std::isfinite(m.normal_degrees) ||
           (m.passed && (!std::isfinite(m.error) || !std::isfinite(m.coverage) ||
                         !std::isfinite(m.coverage_upper)));
}
inline bool audit_measurement_passed(const Measurement& m) {
    return m.complete && m.passed && !m.cancelled && !m.resource_limited &&
           !audit_measurement_nonfinite(m);
}
} // namespace blitz::neural
