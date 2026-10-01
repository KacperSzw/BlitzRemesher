#pragma once
#include "neural/audit_measurement.hpp"

namespace blitz::neural::training {
// Before a predecessor is emitted, both source and adjacent audits compare
// against the original source with the same cameras and quality settings.
inline EvalSettings teacher_packing_settings(EvalSettings source, double adjacent_limit) {
    source.limit = std::min(source.limit, adjacent_limit);
    return source;
}
struct TeacherSeedAudit {
    Measurement source, adjacent, destination;
    bool passed() const {
        for (const auto* m : {&source, &adjacent, &destination})
            if (!audit_measurement_passed(*m))
                return false;
        return true;
    }
};
// The callback always compares the same exact working representation to the
// original source. A loaded episode has no exemption from the current limits.
template <class Evaluate>
TeacherSeedAudit audit_teacher_seed(const EvalSettings& source, double adjacent_limit,
                                    const EvalSettings* destination, Evaluate&& evaluate) {
    TeacherSeedAudit result;
    result.source = evaluate(source);
    auto adjacent = source;
    adjacent.limit = adjacent_limit;
    result.adjacent = adjacent_limit == source.limit ? result.source : evaluate(adjacent);
    result.destination = destination ? evaluate(*destination) : result.source;
    return result;
}
} // namespace blitz::neural::training
