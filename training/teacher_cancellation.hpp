#pragma once
#include "neural/internal.hpp"

namespace blitz::neural::training {
// One preparation owns this state. A callback may report a stop only once, and
// a cancelled batch may return before it knows which trial lanes are valid.
class TeacherCancellation {
  public:
    template <class Poll> bool poll(Poll&& requested) {
        if (!stopped_)
            stopped_ = requested();
        return stopped_;
    }
    void observe(std::span<const CandidateAudit> batch) {
        for (const auto& candidate : batch)
            stopped_ |= candidate.value.cancelled;
    }
    bool stopped() const {
        return stopped_;
    }

  private:
    bool stopped_{};
};
} // namespace blitz::neural::training
