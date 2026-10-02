#pragma once
#include <cstdint>
namespace blitz::neural::training {
// Objective and parameter ownership shared by data admission and optimizers.
enum class TrainableScope : uint8_t { Joint, RankingRow, EndpointScorer };
} // namespace blitz::neural::training
