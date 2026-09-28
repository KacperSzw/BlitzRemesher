#pragma once
#include "blitz/neural.hpp"
#include <nlohmann/json.hpp>
inline nlohmann::json neural_json(const blitz::NeuralStats& s) {
    return {{"encode_seconds",s.encode_ns*1e-9},{"inference_seconds",s.inference_ns*1e-9},{"decode_seconds",s.decode_ns*1e-9},
        {"gpu_audit_seconds",s.gpu_audit_ns*1e-9},{"reference_audit_seconds",s.reference_audit_ns*1e-9},
        {"decoded",s.decoded},{"legal_collapses",s.legal_collapses},{"rejected_collapses",s.rejected_collapses},
        {"reference_rejections",s.reference_rejections},{"fallback_levels",s.fallback_levels}};
}
