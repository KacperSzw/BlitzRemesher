#pragma once
#include <cuda_runtime_api.h>
#include <stdexcept>
namespace blitz::neural::gpu {
// A worker owns one nonblocking stream. All borrowed CUDA/Vulkan resources must
// be destroyed before its scope. The learner may borrow its LibTorch stream.
inline thread_local cudaStream_t active_stream = nullptr;
inline cudaStream_t stream() {
    return active_stream;
}
struct Event {
    cudaEvent_t value{};
    Event() {
        if (cudaEventCreateWithFlags(&value, cudaEventDisableTiming) != cudaSuccess)
            throw std::runtime_error("cannot create CUDA event");
    }
    ~Event() {
        cudaEventDestroy(value);
    }
    Event(const Event&) = delete;
};
struct StreamScope {
    cudaStream_t previous = active_stream, value{};
    bool owned{};
    StreamScope() {
        if (cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking) != cudaSuccess)
            throw std::runtime_error("cannot create CUDA worker stream");
        owned = true;
        active_stream = value;
    }
    explicit StreamScope(cudaStream_t borrowed) : value(borrowed) {
        active_stream = value;
    }
    ~StreamScope() {
        if (owned) {
            cudaStreamSynchronize(value);
            cudaStreamDestroy(value);
        }
        active_stream = previous;
    }
    StreamScope(const StreamScope&) = delete;
};
inline cudaError_t copy(void* to, const void* from, size_t bytes, cudaMemcpyKind kind) {
    auto result = cudaMemcpyAsync(to, from, bytes, kind, stream());
    // Host spans may be pageable and expire on return. D2D storage is retired
    // in stream order; host readback only waits for this worker's dependency.
    return result == cudaSuccess && kind != cudaMemcpyDeviceToDevice
               ? cudaStreamSynchronize(stream())
               : result;
}
inline cudaError_t memset(void* to, int value, size_t bytes) {
    return cudaMemsetAsync(to, value, bytes, stream());
}
} // namespace blitz::neural::gpu
