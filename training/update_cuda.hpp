#pragma once
#include "training/compact.hpp"
#include <array>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace blitz::neural::training {
// Ranges address category -> asset -> nonempty progress bin -> state arrays.
// Each range is nonempty; the host validates the hierarchy before uploading it.
struct SampleRange {
    uint32_t first, count;
};
struct SamplingView {
    const SampleRange *categories, *assets, *bins;
    const uint32_t* states;
    uint32_t category_count, seed;
};
struct ResidentPage {
    const float *values, *conditions, *placements;
    const uint32_t* flags;
    const uint8_t* labels;
    CompactView compact{};
    bool encoded{};
    uint8_t condition_width{8};
};
struct ResidentState {
    uint32_t page, first, condition, rows;
};
struct ResidentRoot {
    SamplingView sampling;
    const ResidentState* states;
    const ResidentPage* pages;
    const uint32_t* const* bin_states{};
};
struct UpdateState {
    uint32_t step{}, failure{}, pairs{}, valid{}, segment_start{};
    uint32_t known[5]{};
    float loss{}, first_loss{}, gradient{}, clip{}, correction1{}, correction2{};
};
struct AdamParameter {
    float *value, *gradient, *mean, *variance;
    uint32_t count, offset;
};
struct UpdateSettings {
    float margin{1}, auxiliary{.25f}, penalty{1e-4f}, lr{.001f}, decay{.0001f}, max_norm{1};
};
enum class UpdateBackend : uint8_t { Reference, Fused };
// V3 features, versioned hidden width. Every pointer borrows stable tensor storage through graph
// replay.
struct FusedMlpBuffers {
    float *parameter[6]{}, *gradient[6]{}, *input{}, *hidden[2]{}, *delta[2]{}, *output{},
        *derivative{};
    void* workspace{};
    size_t workspace_bytes{};
    uint32_t rows{}, width{64};
};
struct FusedMlp;
FusedMlp* create_fused_mlp(const FusedMlpBuffers&);
void destroy_fused_mlp(FusedMlp*) noexcept;
void fused_mlp_forward(FusedMlp*, cudaStream_t);
void fused_mlp_backward(FusedMlp*, cudaStream_t);
uint32_t fused_mlp_epilogues(const FusedMlp*);
std::array<int32_t, 3> fused_mlp_algorithms(const FusedMlp*);
// Counter-based sampling has no hidden RNG state. Version 1 is keyed by
// (seed, completed update, batch lane, hierarchy draw), including rejection draws.
inline constexpr uint32_t sampler_version = 1, update_checkpoint_version = 2;
void sample_update(SamplingView, const float*, const uint8_t*, float*, uint8_t*, uint32_t*,
                   UpdateState*, uint32_t, uint32_t, uint32_t, cudaStream_t);
void sample_packed_update(SamplingView, const float*, const uint32_t*, const float*, const uint8_t*,
                          float*, uint8_t*, uint32_t*, UpdateState*, uint32_t, uint32_t, uint32_t,
                          cudaStream_t, uint8_t condition_width = 8);
void sample_resident_update(const ResidentRoot*, float*, uint8_t*, float*, uint32_t*, UpdateState*,
                            uint32_t, uint32_t, uint32_t, cudaStream_t);
void expand_actions(const float*, const uint32_t*, const float*, const uint8_t*, float*,
                    const int64_t*, uint32_t, uint32_t, uint32_t, uint32_t, cudaStream_t,
                    uint8_t condition_width = 8);
void loss_update(const float*, const uint8_t*, float*, float*, UpdateState*, uint32_t, uint32_t,
                 UpdateSettings, cudaStream_t);
void placement_loss_update(const float*, const uint8_t*, const float*, const uint32_t*, float*,
                           float*, UpdateState*, uint32_t, uint32_t, UpdateSettings, cudaStream_t);
void adam_update(const AdamParameter*, uint32_t, uint32_t, float*, UpdateState*, UpdateSettings,
                 cudaStream_t);
} // namespace blitz::neural::training
