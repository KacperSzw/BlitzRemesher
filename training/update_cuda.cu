#include "training/packed.hpp"
#include "training/update_cuda.hpp"
#include <cmath>
#include <cub/block/block_reduce.cuh>
#include <math_constants.h>

namespace blitz::neural::training {
namespace {
__device__ uint32_t random_word(uint64_t& counter) {
    uint64_t x = (counter += 0x9e3779b97f4a7c15ull);
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return uint32_t((x ^ (x >> 31)) >> 32);
}
__device__ uint32_t choose(uint64_t& counter, uint32_t count) {
    // Rejection, rather than modulo reduction, preserves uniform probabilities.
    uint32_t threshold = uint32_t(-count) % count;
    for (;;) {
        uint64_t product = uint64_t(random_word(counter)) * count;
        if (uint32_t(product) >= threshold)
            return uint32_t(product >> 32);
    }
}
__global__ void select_states(SamplingView s, uint32_t* ids, const UpdateState* state,
                              uint32_t batch) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch || state->failure)
        return;
    uint64_t counter = (uint64_t(s.seed) << 32) ^ uint64_t(state->step) * 0xd1342543de82ef95ull ^
                       uint64_t(i) * 0xa24baed4963ee407ull;
    auto c = s.categories[choose(counter, s.category_count)];
    auto a = s.assets[c.first + choose(counter, c.count)];
    auto b = s.bins[a.first + choose(counter, a.count)];
    ids[i] = s.states[b.first + choose(counter, b.count)];
}
__global__ void select_resident_states(const ResidentRoot* root, uint32_t* ids,
                                       const UpdateState* state, uint32_t batch) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch || state->failure)
        return;
    auto s = root->sampling;
    uint64_t counter = (uint64_t(s.seed) << 32) ^ uint64_t(state->step) * 0xd1342543de82ef95ull ^
                       uint64_t(i) * 0xa24baed4963ee407ull;
    auto c = s.categories[choose(counter, s.category_count)];
    auto a = s.assets[c.first + choose(counter, c.count)];
    auto bin = a.first + choose(counter, a.count);
    auto b = s.bins[bin];
    auto selected = choose(counter, b.count);
    ids[i] = root->bin_states ? root->bin_states[bin][selected] : s.states[b.first + selected];
}
__global__ void gather(const float* x, const uint8_t* labels, float* input, uint8_t* target,
                       const uint32_t* ids, const UpdateState* state, uint32_t batch, uint32_t pool,
                       uint32_t width) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x, n = size_t(batch) * pool;
    if (state->failure)
        return;
    if (i < n)
        target[i] = labels[size_t(ids[i / pool]) * pool + i % pool];
    if (i < n * width) {
        size_t row = i / (pool * width);
        input[i] = x[size_t(ids[row]) * pool * width + i % (pool * width)];
    }
}
__device__ float unpack(const float* values, const uint32_t* flags, const float* conditions,
                        size_t row, uint32_t state, uint32_t channel, uint32_t width,
                        uint8_t condition_width = 8) {
    if (channel == 79 && condition_width == 9)
        return conditions[size_t(state) * condition_width + 8];
    auto slot = feature_slot(channel);
    return slot >= 0     ? values[row * packed_width(width) + unsigned(slot)]
           : slot >= -32 ? float((flags[row] >> unsigned(-slot - 1)) & 1)
                         : conditions[size_t(state) * condition_width + unsigned(-slot - 33)];
}
__global__ void gather_resident(const ResidentRoot* root, float* input, uint8_t* labels,
                                float* placements, const uint32_t* ids, const UpdateState* state,
                                uint32_t batch, uint32_t pool, uint32_t width) {
    if (state->failure)
        return;
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(batch) * pool * width)
        return;
    auto selected = root->states[ids[i / (pool * width)]];
    uint32_t row = uint32_t(i / width) % pool, channel = uint32_t(i % width);
    bool live = row < selected.rows;
    auto page = root->pages[selected.page];
    auto view = page.compact;
    view.first = selected.first;
    view.conditions = page.conditions + size_t(selected.condition) * page.condition_width;
    auto label = live ? page.labels[selected.first + row] : 0;
    input[i] = live ? (page.encoded
                           ? compact_feature(view, row, channel, width)
                           : unpack(page.values, page.flags, page.conditions, selected.first + row,
                                    selected.condition, channel, width, page.condition_width))
                    : 0;
    if (channel == 0)
        labels[i / width] = label;
    if (placements && channel < 9)
        placements[(i / width) * 9 + channel] =
            live ? (page.encoded ? compact_target(view, row, label, channel)
                                 : page.placements[size_t(selected.first + row) * 9 + channel])
                 : 0;
}
__global__ void gather_packed(const float* values, const uint32_t* flags, const float* conditions,
                              const uint8_t* labels, float* input, uint8_t* target,
                              const uint32_t* ids, const UpdateState* state, uint32_t batch,
                              uint32_t pool, uint32_t width, uint8_t condition_width) {
    if (state->failure)
        return;
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(batch) * pool * width)
        return;
    auto selected = ids[i / (pool * width)];
    size_t row = size_t(selected) * pool + (i / width) % pool;
    auto label = labels[row];
    input[i] = label ? unpack(values, flags, conditions, row, selected, uint32_t(i % width), width,
                              condition_width)
                     : 0;
    if (i % width == 0)
        target[i / width] = label;
}
__global__ void expand_packed(const float* values, const uint32_t* flags, const float* conditions,
                              const uint8_t* labels, float* input, const int64_t* ids,
                              uint32_t first, uint32_t count, uint32_t pool, uint32_t width,
                              uint8_t condition_width) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(count) * pool * width)
        return;
    uint32_t selected =
        ids ? uint32_t(ids[i / (pool * width)]) : first + uint32_t(i / (pool * width));
    size_t row = size_t(selected) * pool + (i / width) % pool;
    input[i] = labels[row] ? unpack(values, flags, conditions, row, selected, uint32_t(i % width),
                                    width, condition_width)
                           : 0;
}
__global__ void denominators(const uint8_t* labels, UpdateState* state, uint32_t batch,
                             uint32_t pool) {
    if (state->failure)
        return;
    uint32_t pairs = 0, valid = 0;
    for (uint32_t row = threadIdx.x; row < batch; row += blockDim.x) {
        uint32_t count = 0, preferred = 0;
        for (uint32_t j = 0; j < pool; ++j) {
            auto label = labels[row * pool + j];
            count += bool(label & 8);
            preferred += bool((label & 12) == 12);
        }
        pairs += preferred * (count - preferred);
        valid += count;
    }
    __shared__ cub::BlockReduce<uint32_t, 256>::TempStorage temp;
    auto p = cub::BlockReduce<uint32_t, 256>(temp).Sum(pairs);
    __syncthreads();
    auto v = cub::BlockReduce<uint32_t, 256>(temp).Sum(valid);
    if (threadIdx.x == 0) {
        state->pairs = p;
        state->valid = v;
    }
}
__global__ void loss_gradient(const float* y, const uint8_t* labels, float* dy, float* losses,
                              const UpdateState* state, uint32_t pool, UpdateSettings settings) {
    if (state->failure)
        return;
    uint32_t row = blockIdx.x, j = threadIdx.x;
    float loss = 0;
    const float* prediction = y + size_t(row) * pool * 3;
    const uint8_t* target = labels + row * pool;
    if (j < pool) {
        auto label = target[j];
        float gradients[3]{};
        if (label & 8) {
            float inv_pairs = 1.f / max(1u, state->pairs), inv_valid = 1.f / max(1u, state->valid);
            bool preferred = label & 4;
            float score = prediction[j * 3];
            for (uint32_t k = 0; k < pool; ++k)
                if ((target[k] & 8) && bool(target[k] & 4) != preferred) {
                    float value = settings.margin + (preferred ? prediction[k * 3] - score
                                                               : score - prediction[k * 3]);
                    if (value > 0) {
                        gradients[0] += (preferred ? -1.f : 1.f) * inv_pairs;
                        if (preferred)
                            loss += value * inv_pairs;
                    }
                }
            for (unsigned h = 1; h < 3; ++h) {
                float z = prediction[j * 3 + h], t = bool(label & (h == 1 ? 1 : 2));
                loss += settings.auxiliary * (fmaxf(z, 0) - z * t + log1pf(expf(-fabsf(z)))) *
                        inv_valid;
                gradients[h] = settings.auxiliary *
                               ((z >= 0 ? 1.f / (1.f + expf(-z)) : expf(z) / (1.f + expf(z))) - t) *
                               inv_valid;
            }
            for (unsigned h = 0; h < 3; ++h) {
                float z = prediction[j * 3 + h], weight = settings.penalty * inv_valid / 3;
                loss += weight * z * z;
                gradients[h] += 2 * weight * z;
            }
        }
        for (unsigned h = 0; h < 3; ++h)
            dy[(size_t(row) * pool + j) * 3 + h] = gradients[h];
    }
    __shared__ cub::BlockReduce<float, 32>::TempStorage temp;
    float total = cub::BlockReduce<float, 32>(temp).Sum(loss);
    if (threadIdx.x == 0)
        losses[row] = total;
}
__global__ void sum_loss(const float* losses, UpdateState* state, uint32_t batch) {
    if (state->failure)
        return;
    float sum = 0;
    for (uint32_t i = threadIdx.x; i < batch; i += blockDim.x)
        sum += losses[i];
    __shared__ cub::BlockReduce<float, 256>::TempStorage temp;
    sum = cub::BlockReduce<float, 256>(temp).Sum(sum);
    if (threadIdx.x == 0) {
        state->loss = sum;
        if (state->step == state->segment_start)
            state->first_loss = sum;
        if (!isfinite(sum))
            state->failure = 1;
    }
}
__global__ void placement_denominators(const uint8_t* labels, UpdateState* state, uint32_t batch,
                                       uint32_t pool) {
    if (state->failure)
        return;
    uint32_t pairs = 0, joint = 0, known[5]{};
    for (uint32_t row = threadIdx.x; row < batch; row += blockDim.x) {
        uint32_t count = 0, preferred = 0;
        for (uint32_t j = 0; j < pool; ++j) {
            auto label = labels[row * pool + j];
            bool valid = (label & 24) == 24;
            count += valid;
            preferred += valid && bool(label & 4);
            for (unsigned g = 0; g < 5; ++g)
                known[g] += bool(label & (8u << g));
        }
        pairs += preferred * (count - preferred);
        joint += count;
    }
    __shared__ cub::BlockReduce<uint32_t, 256>::TempStorage temp;
    auto p = cub::BlockReduce<uint32_t, 256>(temp).Sum(pairs);
    if (threadIdx.x == 0)
        state->pairs = p;
    __syncthreads();
    auto n = cub::BlockReduce<uint32_t, 256>(temp).Sum(joint);
    if (threadIdx.x == 0)
        state->valid = n;
    __syncthreads();
    for (unsigned g = 0; g < 5; ++g) {
        auto value = cub::BlockReduce<uint32_t, 256>(temp).Sum(known[g]);
        if (threadIdx.x == 0)
            state->known[g] = value;
        __syncthreads();
    }
}
__global__ void placement_gradient(const float* y, const uint8_t* labels, const float* targets,
                                   const uint32_t* ids, float* dy, float* losses,
                                   const UpdateState* state, uint32_t pool,
                                   UpdateSettings settings) {
    if (state->failure)
        return;
    uint32_t row = blockIdx.x, j = threadIdx.x;
    float loss = 0;
    const float* prediction = y + size_t(row) * pool * 12;
    auto* mask = labels + row * pool;
    if (j < pool) {
        auto label = mask[j];
        float gradients[12]{};
        bool joint = (label & 24) == 24;
        if (joint) {
            bool preferred = label & 4;
            float score = prediction[j * 12], inverse = 1.f / max(1u, state->pairs);
            for (uint32_t k = 0; k < pool; ++k)
                if ((mask[k] & 24) == 24 && bool(mask[k] & 4) != preferred) {
                    float v = settings.margin +
                              (preferred ? prediction[k * 12] - score : score - prediction[k * 12]);
                    if (v > 0) {
                        gradients[0] += (preferred ? -1.f : 1.f) * inverse;
                        if (preferred)
                            loss += v * inverse;
                    }
                }
        }
        for (unsigned h = 0; h < 3; ++h)
            if (h ? bool(label & (h == 1 ? 8 : 16)) : joint) {
                float z = prediction[j * 12 + h],
                      inverse = 1.f / max(1u, h ? state->known[h - 1] : state->valid),
                      weight = settings.penalty * inverse / 3;
                loss += weight * z * z;
                gradients[h] += 2 * weight * z;
                if (h) {
                    float t = bool(label & (h == 1 ? 1 : 2));
                    loss += settings.auxiliary * (fmaxf(z, 0) - z * t + log1pf(expf(-fabsf(z)))) *
                            inverse;
                    gradients[h] +=
                        settings.auxiliary *
                        ((z >= 0 ? 1.f / (1.f + expf(-z)) : expf(z) / (1.f + expf(z))) - t) *
                        inverse;
                }
            }
        for (unsigned group = 0; group < 3; ++group)
            if (label & (32u << group)) {
                float inverse = 1.f / (max(1u, state->known[group + 2]) * 3.f);
                for (unsigned k = 0; k < 3; ++k) {
                    unsigned channel = group * 3 + k;
                    float delta = prediction[j * 12 + 3 + channel] -
                                  targets[(size_t(ids ? ids[row] : row) * pool + j) * 9 + channel];
                    float absolute = fabsf(delta);
                    loss += (absolute < 1 ? .5f * delta * delta : absolute - .5f) * inverse;
                    gradients[3 + channel] = fminf(1, fmaxf(-1, delta)) * inverse;
                }
            }
        for (unsigned h = 0; h < 12; ++h)
            dy[(size_t(row) * pool + j) * 12 + h] = gradients[h];
    }
    __shared__ cub::BlockReduce<float, 32>::TempStorage temp;
    auto total = cub::BlockReduce<float, 32>(temp).Sum(loss);
    if (threadIdx.x == 0)
        losses[row] = total;
}
__device__ AdamParameter parameter(const AdamParameter* parameters, uint32_t count, uint32_t i) {
    for (uint32_t p = 0; p < count; ++p)
        if (i < parameters[p].offset + parameters[p].count)
            return parameters[p];
    return {};
}
__global__ void norm_parts(const AdamParameter* parameters, uint32_t count, uint32_t total,
                           float* partial, const UpdateState* state) {
    if (state->failure)
        return;
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if (i < total) {
        auto p = parameter(parameters, count, i);
        float g = p.gradient[i - p.offset];
        sum = isfinite(g) ? g * g : CUDART_INF_F;
    }
    __shared__ cub::BlockReduce<float, 256>::TempStorage temp;
    sum = cub::BlockReduce<float, 256>(temp).Sum(sum);
    if (threadIdx.x == 0)
        partial[blockIdx.x] = sum;
}
__global__ void norm_finish(const float* partial, uint32_t parts, UpdateState* state,
                            UpdateSettings settings) {
    if (state->failure)
        return;
    float sum = 0;
    for (uint32_t i = threadIdx.x; i < parts; i += blockDim.x)
        sum += partial[i];
    __shared__ cub::BlockReduce<float, 256>::TempStorage temp;
    sum = cub::BlockReduce<float, 256>(temp).Sum(sum);
    if (threadIdx.x == 0) {
        if (state->step == UINT32_MAX) {
            state->failure = 3;
            return;
        }
        state->gradient = sqrtf(sum);
        if (!isfinite(state->gradient)) {
            state->failure = 2;
            return;
        }
        state->clip = fminf(1.f, settings.max_norm / (state->gradient + 1e-6f));
        state->correction1 = float(double(settings.lr) / (1 - pow(.9, double(state->step) + 1)));
        state->correction2 = float(sqrt(1 - pow(.999, double(state->step) + 1)));
    }
}
__global__ void adam(const AdamParameter* parameters, uint32_t count, uint32_t total,
                     const UpdateState* state, UpdateSettings settings) {
    if (state->failure)
        return;
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total)
        return;
    auto p = parameter(parameters, count, i);
    i -= p.offset;
    float g = p.gradient[i] * state->clip;
    float m = p.mean[i] * .9f + g * .1f, v = p.variance[i] * .999f + (g * g) * .001f;
    p.mean[i] = m;
    p.variance[i] = v;
    p.value[i] = p.value[i] * (1 - settings.lr * settings.decay) -
                 state->correction1 * m / (sqrtf(v) / state->correction2 + 1e-8f);
}
__global__ void finish(UpdateState* state) {
    if (!state->failure)
        ++state->step;
}
} // namespace
void sample_update(SamplingView s, const float* x, const uint8_t* labels, float* input,
                   uint8_t* target, uint32_t* ids, UpdateState* state, uint32_t batch,
                   uint32_t pool, uint32_t width, cudaStream_t stream) {
    select_states<<<(batch + 255) / 256, 256, 0, stream>>>(s, ids, state, batch);
    gather<<<(size_t(batch) * pool * width + 255) / 256, 256, 0, stream>>>(
        x, labels, input, target, ids, state, batch, pool, width);
}
void sample_packed_update(SamplingView s, const float* x, const uint32_t* flags,
                          const float* conditions, const uint8_t* labels, float* input,
                          uint8_t* target, uint32_t* ids, UpdateState* state, uint32_t batch,
                          uint32_t pool, uint32_t width, cudaStream_t stream,
                          uint8_t condition_width) {
    select_states<<<(batch + 255) / 256, 256, 0, stream>>>(s, ids, state, batch);
    gather_packed<<<(size_t(batch) * pool * width + 255) / 256, 256, 0, stream>>>(
        x, flags, conditions, labels, input, target, ids, state, batch, pool, width,
        condition_width);
}
void sample_resident_update(const ResidentRoot* root, float* input, uint8_t* labels,
                            float* placements, uint32_t* ids, UpdateState* state, uint32_t batch,
                            uint32_t pool, uint32_t width, cudaStream_t stream) {
    select_resident_states<<<(batch + 255) / 256, 256, 0, stream>>>(root, ids, state, batch);
    gather_resident<<<(size_t(batch) * pool * width + 255) / 256, 256, 0, stream>>>(
        root, input, labels, placements, ids, state, batch, pool, width);
}
void expand_actions(const float* x, const uint32_t* flags, const float* conditions,
                    const uint8_t* labels, float* input, const int64_t* ids, uint32_t first,
                    uint32_t count, uint32_t pool, uint32_t width, cudaStream_t stream,
                    uint8_t condition_width) {
    expand_packed<<<(size_t(count) * pool * width + 255) / 256, 256, 0, stream>>>(
        x, flags, conditions, labels, input, ids, first, count, pool, width, condition_width);
}
void loss_update(const float* y, const uint8_t* labels, float* gradient, float* losses,
                 UpdateState* state, uint32_t batch, uint32_t pool, UpdateSettings settings,
                 cudaStream_t stream) {
    validate_update_settings(settings);
    denominators<<<1, 256, 0, stream>>>(labels, state, batch, pool);
    loss_gradient<<<batch, 32, 0, stream>>>(y, labels, gradient, losses, state, pool, settings);
    sum_loss<<<1, 256, 0, stream>>>(losses, state, batch);
}
void placement_loss_update(const float* y, const uint8_t* labels, const float* targets,
                           const uint32_t* ids, float* gradient, float* losses, UpdateState* state,
                           uint32_t batch, uint32_t pool, UpdateSettings settings,
                           cudaStream_t stream) {
    validate_update_settings(settings);
    placement_denominators<<<1, 256, 0, stream>>>(labels, state, batch, pool);
    placement_gradient<<<batch, 32, 0, stream>>>(y, labels, targets, ids, gradient, losses, state,
                                                 pool, settings);
    sum_loss<<<1, 256, 0, stream>>>(losses, state, batch);
}
void adam_update(const AdamParameter* parameters, uint32_t count, uint32_t total, float* partial,
                 UpdateState* state, UpdateSettings settings, cudaStream_t stream) {
    validate_update_settings(settings);
    uint32_t parts = (total + 255) / 256;
    norm_parts<<<parts, 256, 0, stream>>>(parameters, count, total, partial, state);
    norm_finish<<<1, 256, 0, stream>>>(partial, parts, state, settings);
    adam<<<parts, 256, 0, stream>>>(parameters, count, total, state, settings);
    finish<<<1, 1, 0, stream>>>(state);
}
} // namespace blitz::neural::training
