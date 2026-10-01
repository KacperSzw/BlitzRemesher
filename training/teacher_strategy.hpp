#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>

namespace blitz::neural::training {
enum class TeacherStrategy : uint8_t { Exhaustive, CoverageCoreFirst };
inline std::string_view teacher_strategy_name(TeacherStrategy strategy) {
    switch (strategy) {
    case TeacherStrategy::Exhaustive:
        return "exhaustive";
    case TeacherStrategy::CoverageCoreFirst:
        return "coverage-core-first";
    }
    throw std::invalid_argument("invalid teacher strategy");
}
inline TeacherStrategy teacher_strategy_option(std::string_view value) {
    if (value == "exhaustive")
        return TeacherStrategy::Exhaustive;
    if (value == "coverage-core-first")
        return TeacherStrategy::CoverageCoreFirst;
    throw std::invalid_argument("teacher strategy must be exhaustive or coverage-core-first");
}

// Candidate IDs are the coverage proposal contract: endpoints, midpoint and
// plane solution (0..3), six offsets (4..9), optional learned placement (10).
// This state belongs to one edge in one immutable mesh revision. In particular,
// a rejected exact finalist invalidates every bound pruned by its incumbent.
class CoverageTeacherSearch {
  public:
    static constexpr std::array<uint8_t, 11> order{0, 1, 2, 3, 10, 4, 5, 6, 7, 8, 9};
    struct Observation {
        double margin{std::numeric_limits<double>::infinity()};
        bool valid{}, known{}, safe{}, pruned{};
    };
    explicit CoverageTeacherSearch(size_t count) {
        if (count != 10 && count != 11)
            throw std::invalid_argument("coverage teacher requires 10 or 11 placements");
        available_ = uint16_t((1u << count) - 1);
        active_ = available_ & uint16_t(0xf | (1u << 10));
    }
    uint16_t pending() const {
        return active_ & ~(observed_ | rejected_);
    }
    uint16_t queried() const {
        return queried_;
    }
    uint16_t rejected() const {
        return rejected_;
    }
    bool incomplete() const {
        return incomplete_;
    }
    bool expanded() const {
        return active_ == available_;
    }
    std::optional<uint8_t> best() const {
        std::optional<uint8_t> result;
        for (auto id : order) {
            if (!(observed_ & (1u << id)) || (rejected_ & (1u << id)))
                continue;
            const auto& candidate = observations_[id];
            if (!candidate.valid || !candidate.known || candidate.pruned)
                continue;
            if (!result || (candidate.safe && !observations_[*result].safe) ||
                (candidate.safe == observations_[*result].safe &&
                 candidate.margin < observations_[*result].margin))
                result = id;
        }
        return result;
    }
    double cutoff() const {
        const auto id = best();
        return id && observations_[*id].safe ? observations_[*id].margin
                                             : std::numeric_limits<double>::infinity();
    }
    std::optional<uint8_t> next() const {
        if (incomplete_ || cutoff() == 0)
            return {};
        for (auto id : order)
            if (pending() & (1u << id))
                return id;
        return {};
    }
    bool safe(uint8_t id) const {
        return observations_.at(id).safe;
    }
    void observe(uint8_t id, Observation observation) {
        if (id >= observations_.size() || !(pending() & (1u << id)))
            throw std::logic_error("teacher observation is not pending");
        if (!std::isfinite(observation.margin))
            observation.margin = std::numeric_limits<double>::infinity();
        observations_[id] = observation;
        observed_ |= uint16_t(1u << id);
        queried_ |= uint16_t(1u << id);
        incomplete_ |= observation.valid && !observation.pruned && !observation.known;
    }
    bool expand() {
        const bool changed = !expanded();
        active_ = available_;
        return changed;
    }
    uint16_t reject(uint8_t id) {
        if (id >= observations_.size() || !(observed_ & (1u << id)) || (rejected_ & (1u << id)) ||
            !observations_[id].safe)
            throw std::logic_error("only an observed safe finalist can be rejected");
        rejected_ |= uint16_t(1u << id);
        expand();
        uint16_t invalidated = 0;
        for (auto i : order)
            if ((observed_ & (1u << i)) && observations_[i].pruned)
                invalidated |= uint16_t(1u << i);
        observed_ &= uint16_t(~invalidated);
        return invalidated;
    }

  private:
    std::array<Observation, 11> observations_{};
    uint16_t available_{}, active_{}, observed_{}, queried_{}, rejected_{};
    bool incomplete_{};
};
} // namespace blitz::neural::training
