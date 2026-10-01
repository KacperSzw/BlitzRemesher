#pragma once
#include "neural/teardown_trace.hpp"
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

namespace blitz::neural::training {
// One owner controls up to three resident workers. All workers park while their
// native resources are still alive. Joining the permitted OS thread also waits
// for driver thread-local cleanup, which runs after C++ resource destructors.
// The owner must not hold the work-queue mutex while joining.
class WorkerRetirement {
    std::mutex mutex_;
    std::condition_variable changed_;
    uint8_t parked_{};
    uint8_t permitted_{3};

  public:
    void park(uint8_t worker) {
        assert(worker < 3);
        teardown_trace("worker.retirement", &worker, "begin");
        std::unique_lock lock(mutex_);
        parked_ |= uint8_t(1u << worker);
        changed_.notify_all();
        changed_.wait(lock, [&] { return permitted_ == worker; });
        teardown_trace("worker.retirement", &worker, "end");
    }
    // IDs are the span indices. Non-joinable entries are ignored, including on
    // repeated shutdown. This object serves one set of threads, not a new pool.
    void join(std::span<std::thread> threads) {
        assert(threads.size() <= 3);
        uint8_t pending = 0;
        for (size_t i = 0; i < threads.size(); ++i)
            if (threads[i].joinable())
                pending |= uint8_t(1u << i);
        if (!pending)
            return;
        {
            std::unique_lock lock(mutex_);
            changed_.wait(lock, [&] { return (parked_ & pending) == pending; });
        }
        for (uint8_t i = 0; i < threads.size(); ++i)
            if (pending & uint8_t(1u << i)) {
                {
                    std::lock_guard lock(mutex_);
                    permitted_ = i;
                }
                changed_.notify_all();
                teardown_trace("worker.join", &threads[i], "begin");
                threads[i].join();
                teardown_trace("worker.join", &threads[i], "end");
            }
    }
};

// Sequential readiness matters on failure: earlier initialized workers remain
// idle while the newest thread's partial construction unwinds. Join that failed
// thread through TLS cleanup before the caller retires the preceding workers.
template <class Worker, class Ready>
void start_worker_threads(std::vector<std::thread>& threads, uint8_t count, Worker worker,
                          Ready wait_ready) {
    if (!threads.empty() || count < 1 || count > 3)
        throw std::invalid_argument("teacher worker count outside 1..3 or pool already started");
    threads.reserve(count);
    for (uint8_t i = 0; i < count; ++i) {
        threads.emplace_back([worker, i] { worker(i); });
        try {
            wait_ready();
        } catch (...) {
            threads.back().join();
            threads.pop_back();
            throw;
        }
    }
}
} // namespace blitz::neural::training
