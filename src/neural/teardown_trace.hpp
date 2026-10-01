#pragma once
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <thread>
#ifdef __linux__
#include <sys/prctl.h>
#endif

namespace blitz::neural {
// Called only by the explicitly requested diagnostic benchmark/test entrypoints.
// This independent opt-in lets a sibling debugger collect stacks under Yama;
// container seccomp/capability restrictions may still reject the request.
inline bool allow_debugger_attach() noexcept {
    const char* enabled = std::getenv("BLITZ_ALLOW_DEBUGGER_ATTACH");
    if (!enabled || std::strcmp(enabled, "1"))
        return true;
#ifdef __linux__
    const bool permitted = prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0) == 0;
    const int error = permitted ? 0 : errno;
#else
    const bool permitted = false;
    const int error = ENOSYS;
#endif
    std::fprintf(stderr, "debugger_attach requested=true permitted=%s errno=%d\n",
                 permitted ? "true" : "false", error);
    std::fflush(stderr);
    return permitted;
}
// Opt-in diagnostics only. One stderr record brackets each native teardown
// call; no synchronization or resource lifetime changes are introduced.
inline void teardown_trace(const char* stage, const void* owner, const char* boundary) noexcept {
    const char* enabled = std::getenv("BLITZ_TEARDOWN_TRACE");
    if (!enabled || !*enabled || *enabled == '0')
        return;
    const auto now = std::chrono::duration_cast<std::chrono::microseconds>(
                         std::chrono::steady_clock::now().time_since_epoch())
                         .count();
    std::fprintf(stderr, "teardown us=%lld thread=%zu owner=%p stage=%s %s\n",
                 static_cast<long long>(now),
                 std::hash<std::thread::id>{}(std::this_thread::get_id()), owner, stage, boundary);
    std::fflush(stderr);
}
template <class Work>
void teardown_call(const char* stage, const void* owner, Work&& work) noexcept {
    teardown_trace(stage, owner, "begin");
    work();
    teardown_trace(stage, owner, "end");
}
struct TeardownEnd {
    const char* stage;
    const void* owner;
    ~TeardownEnd() {
        teardown_trace(stage, owner, "end");
    }
};
} // namespace blitz::neural
