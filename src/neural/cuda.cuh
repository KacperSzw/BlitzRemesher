#pragma once
#include "neural/internal.hpp"
#include "neural/memory.hpp"
#include "neural/stream.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <utility>
namespace blitz::neural::gpu {
inline void check(cudaError_t e) {
    if (e == cudaErrorMemoryAllocation)
        throw ResourceError(NeuralResourceLimit::DeviceMemory, 0, 0,
                            "CUDA allocation exceeds available device memory");
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e));
}
inline void check(cublasStatus_t e) {
    if (e != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS failure " + std::to_string(int(e)));
}
struct Device {
    struct Block {
        void* pointer;
        size_t bytes;
        bool used;
    };
    int previous{};
    size_t limit{}, live{}, peak{};
    bool recycle{};
    uint64_t allocations{}, reuses{}, upload_bytes{}, download_bytes{};
    std::vector<Block> blocks;
    MemoryBudget* shared{};
    explicit Device(const NeuralOptions& o, bool pooling = false)
        : limit(size_t(o.memory_mib) * 1024 * 1024), recycle(pooling) {
        if (o.memory_mib < 128 || o.memory_mib > 65536 || o.device < 0)
            throw std::invalid_argument("invalid CUDA options");
        check(cudaGetDevice(&previous));
        check(cudaSetDevice(o.device));
        if (current_memory_budget && current_memory_budget->device == o.device)
            shared = current_memory_budget;
    }
    ~Device() {
        for (auto b : blocks)
            cudaFreeAsync(b.pointer, stream());
        if (shared)
            shared->release(live);
        cudaSetDevice(previous);
    }
    Device(const Device&) = delete;
    bool fits(size_t bytes) const {
        return bytes <= limit - live &&
               (!shared || bytes <= shared->limit - std::min(shared->limit, shared->live.load()));
    }
    void freed(size_t bytes) {
        live -= bytes;
        if (shared)
            shared->release(bytes);
    }
    void* acquire(size_t bytes) {
        if (!bytes)
            return nullptr;
        if (recycle) {
            Block* best = nullptr;
            for (auto& b : blocks)
                if (!b.used && b.bytes >= bytes && b.bytes - bytes <= bytes &&
                    (!best || b.bytes < best->bytes))
                    best = &b;
            if (best) {
                best->used = true;
                ++reuses;
                return best->pointer;
            }
            // Retained capacity counts against the cap. Discard idle blocks before
            // denying a request that fits alongside the actual live buffers.
            for (size_t i = blocks.size(); !fits(bytes) && i-- > 0;)
                if (!blocks[i].used) {
                    cudaFreeAsync(blocks[i].pointer, stream());
                    freed(blocks[i].bytes);
                    blocks.erase(blocks.begin() + i);
                }
        }
        if (bytes > limit - live || (shared && !shared->reserve(bytes)))
            throw ResourceError(
                NeuralResourceLimit::WorkspaceMemory, (shared ? shared->live.load() : live) + bytes,
                shared ? shared->limit : limit, "CUDA workspace exceeds configured memory cap");
        void* p = nullptr;
        auto status = cudaMallocAsync(&p, bytes, stream());
        if (status == cudaErrorMemoryAllocation && recycle) {
            for (size_t i = blocks.size(); i-- > 0;)
                if (!blocks[i].used) {
                    cudaFreeAsync(blocks[i].pointer, stream());
                    freed(blocks[i].bytes);
                    blocks.erase(blocks.begin() + i);
                }
            status = cudaMallocAsync(&p, bytes, stream());
        }
        if (status != cudaSuccess && shared)
            shared->release(bytes);
        check(status);
        if (recycle) {
            try {
                blocks.push_back({p, bytes, true});
            } catch (...) {
                cudaFreeAsync(p, stream());
                if (shared)
                    shared->release(bytes);
                throw;
            }
        }
        live += bytes;
        peak = std::max(peak, live);
        ++allocations;
        return p;
    }
    void release(void* p, size_t bytes) noexcept {
        if (recycle) {
            for (auto& b : blocks)
                if (b.pointer == p) {
                    b.used = false;
                    return;
                }
        } else {
            cudaFreeAsync(p, stream());
            freed(bytes);
        }
    }
};
template <class T> struct Buffer {
    Device* device{};
    T* p{};
    size_t n{};
    Buffer() = default;
    Buffer(Device& d, size_t count) : device(&d), n(count) {
        if (n > SIZE_MAX / sizeof(T))
            throw ResourceError(NeuralResourceLimit::WorkspaceMemory, UINT64_MAX, d.limit,
                                "CUDA allocation size overflow");
        p = static_cast<T*>(d.acquire(n * sizeof(T)));
    }
    ~Buffer() {
        if (p)
            device->release(p, n * sizeof(T));
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& b) noexcept : device(b.device), p(std::exchange(b.p, nullptr)), n(b.n) {}
    Buffer& operator=(Buffer&& b) noexcept {
        if (this != &b) {
            if (p)
                device->release(p, n * sizeof(T));
            device = b.device;
            p = std::exchange(b.p, nullptr);
            n = b.n;
        }
        return *this;
    }
    void zero() {
        if (n)
            check(gpu::memset(p, 0, n * sizeof(T)));
    }
    void upload(std::span<const T> a) {
        if (a.size() != n)
            throw std::invalid_argument("CUDA upload size");
        if (n) {
            check(gpu::copy(p, a.data(), n * sizeof(T), cudaMemcpyHostToDevice));
            device->upload_bytes += n * sizeof(T);
        }
    }
    std::vector<T> download() const {
        std::vector<T> a(n);
        if (n) {
            check(gpu::copy(a.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
            device->download_bytes += n * sizeof(T);
        }
        return a;
    }
};
template <class T> Buffer<T> upload_stream(Device& d, Stream<T> stream) {
    Buffer<T> out(d, stream.count);
    if (stream.count) {
        if (stream.stride == sizeof(T))
            check(gpu::copy(out.p, stream.data, stream.count * sizeof(T), cudaMemcpyHostToDevice));
        else {
            check(cudaMemcpy2DAsync(out.p, sizeof(T), stream.data, stream.stride, sizeof(T),
                                    stream.count, cudaMemcpyHostToDevice, gpu::stream()));
            check(cudaStreamSynchronize(gpu::stream()));
        }
        d.upload_bytes += stream.count * sizeof(T);
    }
    return out;
}
inline unsigned blocks(size_t n) {
    return unsigned((n + 255) / 256);
}
struct Blas {
    cublasHandle_t value{};
    Blas() {
        check(cublasCreate(&value));
        check(cublasSetStream(value, stream()));
    }
    ~Blas() {
        cublasDestroy(value);
    }
    operator cublasHandle_t() const {
        return value;
    }
};
} // namespace blitz::neural::gpu
