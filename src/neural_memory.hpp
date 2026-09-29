#pragma once
#include "blitz/neural.hpp"
namespace blitz::neural {
// A generation owns all its CUDA workspaces on one thread. Devices borrow this
// budget and must be destroyed before the enclosing scope. Independent calls
// on other threads retain independent budgets; no global allocator is imposed.
struct MemoryBudget {size_t limit{},live{},peak{};int32_t device{};};
inline thread_local MemoryBudget* current_memory_budget=nullptr;
struct MemoryScope {
    MemoryBudget budget;MemoryBudget* previous;
    explicit MemoryScope(const NeuralOptions& o):budget{size_t(o.memory_mib)*1024*1024,0,0,o.device},previous(current_memory_budget){
        if(!previous||previous->device!=o.device)current_memory_budget=&budget;
    }
    ~MemoryScope(){current_memory_budget=previous;}
    MemoryScope(const MemoryScope&)=delete;
};
}
