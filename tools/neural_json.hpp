#pragma once
#include "blitz/neural.hpp"
#include <nlohmann/json.hpp>
inline const char* ranking_name(blitz::NeuralRanking r){switch(r){case blitz::NeuralRanking::Learned:return "learned";case blitz::NeuralRanking::Constant:return "constant";case blitz::NeuralRanking::Shuffled:return "shuffled";case blitz::NeuralRanking::ShortestEdge:return "shortest";case blitz::NeuralRanking::CurrentPlane:return "current-plane";}throw std::invalid_argument("invalid neural ranking");}
inline blitz::NeuralRanking ranking_option(std::string_view value){for(auto r:{blitz::NeuralRanking::Learned,blitz::NeuralRanking::Constant,blitz::NeuralRanking::Shuffled,blitz::NeuralRanking::ShortestEdge,blitz::NeuralRanking::CurrentPlane})if(value==ranking_name(r))return r;throw std::invalid_argument("unknown neural ranking control");}
inline uint32_t neural_unsigned(const char* value){std::string text=value;if(text.empty()||text.find_first_not_of("0123456789")!=std::string::npos)throw std::invalid_argument("invalid neural unsigned option");auto n=std::stoull(text);if(n>UINT32_MAX)throw std::invalid_argument("neural option overflow");return uint32_t(n);}
inline uint8_t neural_batch(const char* value){auto n=neural_unsigned(value);if(n<1||n>64)throw std::invalid_argument("action batch outside 1..64");return uint8_t(n);}
inline nlohmann::json neural_json(const blitz::NeuralOptions& o){return {{"device",o.device},{"memory_mib",o.memory_mib},{"overdraw_tiebreak",o.overdraw_tiebreak},{"action_trial_budget",o.action_trials},{"action_batch",o.action_batch},{"ranking",ranking_name(o.ranking)},{"ranking_seed",o.ranking_seed}};}
inline nlohmann::json neural_json(const blitz::NeuralStats& s) {
    auto& f=s.first_resource_failure;const char* kinds[]={"none","sample_count","workspace_memory","device_memory","tile_entries"};
    return {{"encode_seconds",s.encode_ns*1e-9},{"inference_seconds",s.inference_ns*1e-9},{"decode_seconds",s.decode_ns*1e-9},
        {"gpu_audit_seconds",s.gpu_audit_ns*1e-9},{"reference_audit_seconds",s.reference_audit_ns*1e-9},
        {"decoded",s.decoded},{"legal_collapses",s.legal_collapses},{"rejected_collapses",s.rejected_collapses},
        {"action_ranked",s.action_ranked},{"action_trials",s.action_trials},{"action_audit_cache_hits",s.action_audit_cache_hits},
        {"reference_rejections",s.reference_rejections},{"fallback_levels",s.fallback_levels},{"bounded_audits",s.bounded_audits},
        {"gpu_peak_bytes",s.gpu_peak_bytes},{"resource_failures",s.resource_failures},{"first_resource_failure",s.resource_failures?nlohmann::json{
            {"kind",kinds[unsigned(f.kind)]},{"screen_pixels",f.screen_pixels},{"supersample",f.supersample},{"view",f.view},{"requested",f.requested},{"limit",f.limit}}:nlohmann::json(nullptr)}};
}
