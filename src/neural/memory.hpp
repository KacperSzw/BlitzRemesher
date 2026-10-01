#pragma once
#include "blitz/neural.hpp"
#include <atomic>
namespace blitz::neural::gpu {
struct ResourceError : std::length_error {
    NeuralResourceLimit kind;
    uint64_t requested, limit;
    ResourceError(NeuralResourceLimit k, uint64_t r, uint64_t l, const char* message)
        : std::length_error(message), kind(k), requested(r), limit(l) {}
};
} // namespace blitz::neural::gpu
namespace blitz::neural {
// A generation owns all its CUDA workspaces on one thread. Devices borrow this
// budget and must be destroyed before the enclosing scope. Independent calls
// on other threads retain independent budgets; no global allocator is imposed.
struct MemoryBudget {
    size_t limit{};
    std::atomic<size_t> live{}, peak{};
    int32_t device{};
    bool reserve(size_t bytes) {
        auto value = live.load(std::memory_order_relaxed);
        for (;;) {
            if (value > limit || bytes > limit - value)
                return false;
            if (live.compare_exchange_weak(value, value + bytes, std::memory_order_relaxed))
                break;
        }
        auto high = peak.load(std::memory_order_relaxed);
        while (high < value + bytes &&
               !peak.compare_exchange_weak(high, value + bytes, std::memory_order_relaxed)) {
        }
        return true;
    }
    void release(size_t bytes) {
        live.fetch_sub(bytes, std::memory_order_relaxed);
    }
};
inline thread_local MemoryBudget* current_memory_budget = nullptr;
struct MemoryScope {
    MemoryBudget budget;
    MemoryBudget* previous;
    explicit MemoryScope(const NeuralOptions& o)
        : budget{size_t(o.memory_mib) * 1024 * 1024, 0, 0, o.device},
          previous(current_memory_budget) {
        if (!previous || previous->device != o.device)
            current_memory_budget = &budget;
    }
    explicit MemoryScope(MemoryBudget& shared) : previous(current_memory_budget) {
        current_memory_budget = &shared;
    }
    ~MemoryScope() {
        current_memory_budget = previous;
    }
    MemoryScope(const MemoryScope&) = delete;
};
} // namespace blitz::neural
