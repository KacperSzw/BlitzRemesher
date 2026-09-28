#pragma once
#include <chrono>
#include <cstdint>
namespace blitz::detail {
// Optional caller-owned accumulator; no clock reads when instrumentation is off.
struct ScopedTime {
    using Clock=std::chrono::steady_clock;
    uint64_t* total;
    Clock::time_point start;
    explicit ScopedTime(uint64_t* p):total(p),start(p?Clock::now():Clock::time_point{}){}
    ~ScopedTime(){if(total)*total+=uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count());}
};
}
