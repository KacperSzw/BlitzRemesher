#pragma once
#include "neural/placement.hpp"
#include "training/data.hpp"

namespace blitz::neural::training {
struct TrainingCondition {
    std::string asset;
    uint32_t pixels{}, previous{};
    double previous_pixels{}, source_limit{3}, adjacent_limit{3};
    bool preserve_uv{true};
};
inline std::vector<TrainingCondition> training_conditions(const json& schedule,
                                                          uint32_t architecture) {
    const auto version = schedule.at("version").get<uint32_t>();
    if ((version != 1 && version != 2) || !schedule.at("conditions").is_array() ||
        schedule.at("conditions").empty() || schedule.at("conditions").size() > 4096)
        throw std::invalid_argument("invalid curriculum");
    std::vector<TrainingCondition> result;
    for (const auto& c : schedule.at("conditions")) {
        TrainingCondition item{c.at("asset").get<std::string>(),
                               c.at("pixels").get<uint32_t>(),
                               c.at("previous_steps").get<uint32_t>(),
                               c.value("previous_pixels", 0.),
                               c.value("source_limit", 3.),
                               c.value("adjacent_limit", 3.),
                               c.value("preserve_uv", true)};
        if (item.asset.empty() || item.pixels < 16 || item.pixels > 512 || item.previous > 64 ||
            !std::isfinite(item.source_limit) || item.source_limit <= 0 || item.source_limit > 16 ||
            !std::isfinite(item.adjacent_limit) || item.adjacent_limit <= 0 ||
            item.adjacent_limit > 16 || !std::isfinite(item.previous_pixels) ||
            item.previous_pixels < 0 || item.previous_pixels > 512 ||
            (item.previous_pixels && (!item.previous || item.previous_pixels < item.pixels)) ||
            (!item.preserve_uv && architecture != conditioned_placement_schema))
            throw std::invalid_argument("invalid curriculum condition");
        result.push_back(std::move(item));
    }
    return result;
}
} // namespace blitz::neural::training
