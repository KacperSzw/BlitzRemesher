#include "training/worker_retirement.hpp"
#include <array>
#include <atomic>
#include <exception>
#include <iostream>
#include <string>
using namespace blitz::neural::training;

static void require(bool value, const char* why) {
    if (!value)
        throw std::runtime_error(why);
}

struct Probe {
    std::mutex mutex;
    std::condition_variable changed;
    WorkerRetirement retirement;
    std::vector<std::thread> threads;
    uint8_t count{}, ready{}, started{}, live{}, working{}, quiesced{}, tls_started{}, tls_done{};
    uint8_t release_tls{}, observed_cancel{};
    int fail_start{-1}, fail_work{-1};
    bool begin_work{}, stop{}, cancelled{}, startup_caught{};
    std::exception_ptr error;
    std::string violation;
    explicit Probe(uint8_t workers) : count(workers) {}
    static uint8_t bit(uint8_t id) {
        return uint8_t(1u << id);
    }
    uint8_t all() const {
        return uint8_t((1u << count) - 1);
    }
    void check(bool condition, const char* why) {
        if (!condition && violation.empty())
            violation = why;
    }
    template <class Predicate> void wait(Predicate condition) {
        std::unique_lock lock(mutex);
        changed.wait(lock, condition);
    }
    void publish_error() {
        std::lock_guard lock(mutex);
        error = std::current_exception();
        changed.notify_all();
    }
    void exit_thread(uint8_t id) {
        std::unique_lock lock(mutex);
        check(!(live & bit(id)), "thread-local cleanup preceded local resource destruction");
        tls_started |= bit(id);
        changed.notify_all();
        changed.wait(lock, [&] { return release_tls & bit(id); });
        tls_done |= bit(id);
        changed.notify_all();
    }
    struct ThreadExit {
        Probe* probe{};
        uint8_t id{};
        ~ThreadExit() {
            if (probe)
                probe->exit_thread(id);
        }
    };
    struct Resource {
        Probe& probe;
        uint8_t id;
        ~Resource() {
            std::lock_guard lock(probe.mutex);
            if (id != probe.fail_start) {
                const auto expected =
                    probe.fail_start < 0 ? probe.all() : uint8_t((1u << probe.fail_start) - 1);
                probe.check((probe.quiesced & expected) == expected,
                            "resource retired before every initialized worker quiesced");
                probe.check((probe.tls_done & uint8_t((1u << id) - 1)) == uint8_t((1u << id) - 1),
                            "next resource retired before preceding thread-local cleanup");
                if (probe.fail_start >= 0)
                    probe.check(probe.tls_done & bit(uint8_t(probe.fail_start)),
                                "initialized worker retired before failed startup thread exited");
            }
            probe.live &= uint8_t(~bit(id));
            probe.changed.notify_all();
        }
    };
    void worker(uint8_t id) {
        static thread_local ThreadExit exit;
        exit.probe = this;
        exit.id = id;
        try {
            {
                std::lock_guard lock(mutex);
                started |= bit(id);
                live |= bit(id);
            }
            Resource resource{*this, id};
            if (id == fail_start)
                throw std::runtime_error("test initialization failure after resource construction");
            {
                std::lock_guard lock(mutex);
                ++ready;
                changed.notify_all();
            }
            try {
                std::unique_lock lock(mutex);
                changed.wait(lock, [&] { return begin_work || stop; });
                if (!stop) {
                    working |= bit(id);
                    changed.notify_all();
                    // Every worker must run concurrently; sequential startup is
                    // allowed, but serializing the warm work would strand this.
                    changed.wait(lock, [&] { return working == all(); });
                    if (id == fail_work)
                        throw std::runtime_error("test fatal work-loop failure");
                    changed.wait(lock, [&] { return stop; });
                    if (cancelled)
                        observed_cancel |= bit(id);
                }
            } catch (...) {
                publish_error();
            }
            {
                std::lock_guard lock(mutex);
                quiesced |= bit(id);
            }
            retirement.park(id);
        } catch (...) {
            publish_error();
        }
    }
    void start() {
        start_worker_threads(
            threads, count, [this](uint8_t id) { worker(id); },
            [this] {
                std::unique_lock lock(mutex);
                changed.wait(lock, [&] { return error || ready == threads.size(); });
                if (error)
                    std::rethrow_exception(error);
            });
    }
    void stop_work(bool cancel = false) {
        std::lock_guard lock(mutex);
        cancelled = cancel;
        stop = true;
        changed.notify_all();
    }
};

static void normal_and_fatal_shutdown(uint8_t count, bool fatal) {
    Probe probe(count);
    probe.release_tls = uint8_t(probe.all() & ~Probe::bit(0));
    probe.fail_work = fatal ? count - 1 : -1;
    probe.start();
    {
        std::lock_guard lock(probe.mutex);
        probe.begin_work = true;
        probe.changed.notify_all();
    }
    probe.wait([&] { return probe.working == probe.all(); });
    if (fatal)
        // The controller cannot start shutdown until the fatal worker reports
        // its failure. Reporting only after park() would deadlock this test.
        probe.wait([&] { return bool(probe.error); });
    probe.stop_work(true);
    std::thread controller([&] { probe.retirement.join(probe.threads); });
    probe.wait([&] { return probe.tls_started & Probe::bit(0); });
    {
        std::lock_guard lock(probe.mutex);
        probe.check(probe.live == uint8_t(probe.all() & ~Probe::bit(0)),
                    "later worker resources disappeared while first TLS destructor was blocked");
        probe.release_tls |= Probe::bit(0);
        probe.changed.notify_all();
    }
    controller.join();
    probe.retirement.join(probe.threads);
    require(probe.violation.empty(), probe.violation.c_str());
    require(probe.live == 0 && probe.tls_done == probe.all(), "shutdown left a live worker");
    const auto cancelled = fatal ? uint8_t(probe.all() & ~Probe::bit(count - 1)) : probe.all();
    require(probe.observed_cancel == cancelled, "active workers did not observe cancellation");
    require(bool(probe.error) == fatal, "worker failure reporting changed");
}

static void failed_startup(uint8_t failure) {
    Probe probe(3);
    probe.fail_start = failure;
    probe.release_tls = uint8_t(probe.all() & ~Probe::bit(failure));
    std::thread constructor([&] {
        try {
            probe.start();
        } catch (...) {
            {
                std::lock_guard lock(probe.mutex);
                probe.startup_caught = true;
            }
            probe.stop_work();
            probe.retirement.join(probe.threads);
        }
    });
    probe.wait([&] { return probe.tls_started & Probe::bit(failure); });
    {
        std::lock_guard lock(probe.mutex);
        probe.check(probe.started == uint8_t((1u << (failure + 1)) - 1),
                    "a later worker started before the failed startup thread exited");
        probe.check(probe.live == uint8_t((1u << failure) - 1),
                    "an earlier initialized worker retired during failed startup cleanup");
        probe.check(!probe.startup_caught,
                    "startup error escaped before failed thread-local cleanup completed");
        probe.release_tls |= Probe::bit(failure);
        probe.changed.notify_all();
    }
    constructor.join();
    probe.retirement.join(probe.threads);
    require(probe.violation.empty(), probe.violation.c_str());
    require(probe.startup_caught && probe.threads.size() == failure,
            "failed newest thread was not joined and removed");
    require(probe.live == 0 && probe.tls_done == probe.started,
            "failed startup left resources or thread-local cleanup live");
}

int main() {
    try {
        for (uint8_t count : {1, 2, 3}) {
            normal_and_fatal_shutdown(count, false);
            normal_and_fatal_shutdown(count, true);
        }
        for (uint8_t failure : {0, 1, 2})
            failed_startup(failure);
        WorkerRetirement empty;
        std::array<std::thread, 3> none;
        empty.join(none);
        empty.join(none);
        WorkerRetirement sparse;
        std::array<std::thread, 3> slots;
        std::atomic<uint8_t> finished{};
        for (uint8_t id : {1, 2})
            slots[id] = std::thread([&, id] {
                sparse.park(id);
                finished.fetch_or(Probe::bit(id));
            });
        sparse.join(slots);
        sparse.join(slots);
        require(finished == 6, "non-joinable slot changed the remaining worker IDs");
        std::cout << "worker retirement contracts passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
