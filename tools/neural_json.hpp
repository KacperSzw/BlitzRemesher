#pragma once
#include "blitz/neural.hpp"
#include <nlohmann/json.hpp>
inline const char* raster_name(blitz::NeuralRasterBackend b){return b==blitz::NeuralRasterBackend::Vulkan?"vulkan-v1":"cuda-v1";}
inline blitz::NeuralRasterBackend raster_option(std::string_view s){if(s=="cuda")return blitz::NeuralRasterBackend::Cuda;if(s=="vulkan")return blitz::NeuralRasterBackend::Vulkan;throw std::invalid_argument("raster backend must be cuda or vulkan");}
inline const char* storage_name(blitz::NeuralVertexStorage s){switch(s){case blitz::NeuralVertexStorage::Float32:return "fp32";case blitz::NeuralVertexStorage::Position16:return "position16";case blitz::NeuralVertexStorage::Packed:return "packed";case blitz::NeuralVertexStorage::Automatic:return "auto";}throw std::invalid_argument("invalid vertex storage");}
inline blitz::NeuralVertexStorage storage_option(std::string_view s){for(auto value:{blitz::NeuralVertexStorage::Float32,blitz::NeuralVertexStorage::Position16,blitz::NeuralVertexStorage::Packed,blitz::NeuralVertexStorage::Automatic})if(s==storage_name(value))return value;throw std::invalid_argument("vertex storage must be auto, fp32, position16 or packed");}
inline const char* confirmation_name(blitz::NeuralConfirmation c){switch(c){case blitz::NeuralConfirmation::Cpu:return "cpu";case blitz::NeuralConfirmation::Gpu:return "gpu";case blitz::NeuralConfirmation::Compare:return "compare";}throw std::invalid_argument("invalid confirmation backend");}
inline blitz::NeuralConfirmation confirmation_option(std::string_view s){for(auto c:{blitz::NeuralConfirmation::Cpu,blitz::NeuralConfirmation::Gpu,blitz::NeuralConfirmation::Compare})if(s==confirmation_name(c))return c;throw std::invalid_argument("unknown confirmation backend");}
inline const char* confirmation_reason(blitz::NeuralConfirmationReason r){switch(r){case blitz::NeuralConfirmationReason::Visual:return "visual";case blitz::NeuralConfirmationReason::Cancelled:return "cancelled";case blitz::NeuralConfirmationReason::Resource:return "resource";case blitz::NeuralConfirmationReason::Nonfinite:return "nonfinite";case blitz::NeuralConfirmationReason::Disagreement:return "disagreement";}return "unknown";}
inline const char* ranking_name(blitz::NeuralRanking r){switch(r){case blitz::NeuralRanking::Learned:return "learned";case blitz::NeuralRanking::Constant:return "constant";case blitz::NeuralRanking::Shuffled:return "shuffled";case blitz::NeuralRanking::ShortestEdge:return "shortest";case blitz::NeuralRanking::CurrentPlane:return "current-plane";}throw std::invalid_argument("invalid neural ranking");}
inline blitz::NeuralRanking ranking_option(std::string_view value){for(auto r:{blitz::NeuralRanking::Learned,blitz::NeuralRanking::Constant,blitz::NeuralRanking::Shuffled,blitz::NeuralRanking::ShortestEdge,blitz::NeuralRanking::CurrentPlane})if(value==ranking_name(r))return r;throw std::invalid_argument("unknown neural ranking control");}
inline uint32_t neural_unsigned(const char* value){std::string text=value;if(text.empty()||text.find_first_not_of("0123456789")!=std::string::npos)throw std::invalid_argument("invalid neural unsigned option");auto n=std::stoull(text);if(n>UINT32_MAX)throw std::invalid_argument("neural option overflow");return uint32_t(n);}
inline uint8_t neural_batch(const char* value){auto n=neural_unsigned(value);if(n<1||n>64)throw std::invalid_argument("action batch outside 1..64");return uint8_t(n);}
inline nlohmann::json neural_json(const blitz::NeuralOptions& o){return {{"device",o.device},{"memory_mib",o.memory_mib},{"raster",raster_name(o.raster_backend)},{"vertex_storage",storage_name(o.draw_storage())},{"overdraw_tiebreak",o.overdraw_tiebreak},{"action_trial_budget",o.action_trials},{"action_batch",o.action_batch},{"ranking",ranking_name(o.ranking)},{"ranking_seed",o.ranking_seed},{"confirmation",confirmation_name(o.confirmation)},{"cache_rasters",o.cache_rasters}};}
inline nlohmann::json neural_json(const blitz::NeuralStats& s) {
    auto& f=s.first_resource_failure;const char* kinds[]={"none","sample_count","workspace_memory","device_memory","tile_entries"};
    return {{"encode_seconds",s.encode_ns*1e-9},{"inference_seconds",s.inference_ns*1e-9},{"decode_seconds",s.decode_ns*1e-9},
        {"gpu_audit_seconds",s.gpu_audit_ns*1e-9},{"reference_audit_seconds",s.reference_audit_ns*1e-9},
        {"gpu_confirmation_seconds",s.gpu_confirmation_ns*1e-9},{"gpu_evaluations",s.gpu_evaluations},{"gpu_measurement_cache_hits",s.gpu_measurement_cache_hits},
        {"gpu_allocations",s.gpu_allocations},{"gpu_buffer_reuses",s.gpu_buffer_reuses},{"gpu_upload_bytes",s.gpu_upload_bytes},{"gpu_download_bytes",s.gpu_download_bytes},
        {"gpu_rasters",s.gpu_rasters},{"gpu_reference_render_hits",s.gpu_reference_render_hits},{"gpu_candidate_render_hits",s.gpu_candidate_render_hits},
        {"gpu_sparse_passes",s.gpu_sparse_passes},{"gpu_sparse_failures",s.gpu_sparse_failures},{"gpu_sparse_fallbacks",s.gpu_sparse_fallbacks},{"gpu_sparse_queries",s.gpu_sparse_queries},
        {"confirmation_cancelled",s.confirmation_cancelled},{"confirmation_resources",s.confirmation_resources},{"confirmation_nonfinite",s.confirmation_nonfinite},{"confirmation_disagreements",s.confirmation_disagreements},
        {"confirmation_failure",s.confirmation_failure?nlohmann::json{{"reason",confirmation_reason(s.confirmation_failure->reason)},{"backend",confirmation_name(s.confirmation_failure->backend)},
            {"level",s.confirmation_failure->level},{"adjacent",s.confirmation_failure->adjacent},{"seconds",s.confirmation_failure->nanoseconds*1e-9}}:nlohmann::json(nullptr)},
        {"decoded",s.decoded},{"legal_collapses",s.legal_collapses},{"rejected_collapses",s.rejected_collapses},
        {"action_ranked",s.action_ranked},{"action_trials",s.action_trials},{"action_audit_cache_hits",s.action_audit_cache_hits},
        {"reference_rejections",s.reference_rejections},{"fallback_levels",s.fallback_levels},{"bounded_audits",s.bounded_audits},
        {"gpu_peak_bytes",s.gpu_peak_bytes},{"resource_failures",s.resource_failures},{"first_resource_failure",s.resource_failures?nlohmann::json{
            {"kind",kinds[unsigned(f.kind)]},{"screen_pixels",f.screen_pixels},{"supersample",f.supersample},{"view",f.view},{"requested",f.requested},{"limit",f.limit}}:nlohmann::json(nullptr)}};
}
