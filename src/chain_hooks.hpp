#pragma once
#include "blitz/remesher.hpp"
namespace blitz::detail {
struct GenerationHooks {
    std::function<Lod(MeshView, const ReduceSettings&, const EvalSettings&, double)> propose;
    // Fixed source and preceding emitted LOD, never the last accepted local edit.
    std::function<Lod(MeshView, MeshView, MeshView, const Bounds&, const ReduceSettings&,
                      const EvalSettings&, const EvalSettings&)>
        propose_guarded;
    std::function<Measurement(MeshView, MeshView, const Bounds&, const EvalSettings&)> evaluate;
    std::function<bool(Result&)> confirm;
    std::function<double(const Result&)> tiebreak;
    // Packed rendering may make the raw source an invalid fallback. Supply an
    // owned baseline, or no proposal when that representation is infeasible.
    // The chain still applies every source/adjacent gate and can retain another
    // already audited incumbent. Unexpected failures must propagate.
    std::function<std::optional<Lod>(MeshView, const EvalSettings&, const EvalSettings&,
                                     const EvalSettings&, const EvalSettings&)>
        fallback;
    // Quantized reuse can also make raw source copies invalid as scheduled LODs,
    // even when no owned fallback proposal is supplied.
    bool source_fallback_requires_audit{};
};
Result generate_with_hooks(MeshView, const Settings&, const Proposer&, const GenerationHooks*);
} // namespace blitz::detail
