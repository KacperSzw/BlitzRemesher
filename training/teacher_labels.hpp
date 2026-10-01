#pragma once
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace blitz::neural::training {
struct TeacherChoice {
    uint32_t faces{};
    double margin{};
    bool confirmed{};
};
// Only completed independent audits establish equivalent preferred choices.
// Unknown/pruned observations cannot become positive or negative rank labels.
inline uint16_t preferred_actions(std::span<const TeacherChoice> choices) {
    if (choices.size() > 16)
        throw std::invalid_argument("teacher choice mask capacity");
    uint32_t faces = UINT32_MAX;
    double margin = std::numeric_limits<double>::infinity();
    uint16_t result = 0;
    for (size_t i = 0; i < choices.size(); ++i) {
        const auto& choice = choices[i];
        if (!choice.confirmed || !std::isfinite(choice.margin))
            continue;
        if (choice.faces < faces || (choice.faces == faces && choice.margin < margin)) {
            faces = choice.faces;
            margin = choice.margin;
            result = 0;
        }
        if (choice.faces == faces && choice.margin == margin)
            result |= uint16_t(1u << i);
    }
    return result;
}
} // namespace blitz::neural::training
