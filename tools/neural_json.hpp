#pragma once
#include "blitz/neural.hpp"
#include <nlohmann/json.hpp>
inline nlohmann::json neural_json(const blitz::NeuralStats& s) {
    auto& f=s.first_resource_failure;const char* kinds[]={"none","sample_count","workspace_memory","device_memory","tile_entries"};
    return {{"encode_seconds",s.encode_ns*1e-9},{"inference_seconds",s.inference_ns*1e-9},{"decode_seconds",s.decode_ns*1e-9},
        {"gpu_audit_seconds",s.gpu_audit_ns*1e-9},{"reference_audit_seconds",s.reference_audit_ns*1e-9},
        {"decoded",s.decoded},{"legal_collapses",s.legal_collapses},{"rejected_collapses",s.rejected_collapses},
        {"reference_rejections",s.reference_rejections},{"fallback_levels",s.fallback_levels},{"bounded_audits",s.bounded_audits},
        {"gpu_peak_bytes",s.gpu_peak_bytes},{"resource_failures",s.resource_failures},{"first_resource_failure",s.resource_failures?nlohmann::json{
            {"kind",kinds[unsigned(f.kind)]},{"screen_pixels",f.screen_pixels},{"supersample",f.supersample},{"view",f.view},{"requested",f.requested},{"limit",f.limit}}:nlohmann::json(nullptr)}};
}
