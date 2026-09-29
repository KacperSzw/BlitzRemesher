#pragma once
#include "neural_internal.hpp"

namespace blitz::neural {
// Diagnostic state is owned; traces never borrow tensors that a later layer overwrites.
struct NumericLayer {
    std::string name;
    uint32_t width{};
    std::vector<double> values;
};
struct NumericDifference {
    bool finite{true},same_shape{true};
    double maximum{},rms{},relative{};
    size_t worst{};
};
void validate_graph(const Graph&);
NumericTrace reference_fp64(const Graph&,const std::array<float,conditions>&,const WeightsData&);
NumericDifference numeric_difference(std::span<const double>,std::span<const double>);
// Kept at the original threshold until a recorded discrepancy is diagnosed.
bool legacy_numeric_pass(const NumericDifference&,double absolute=2e-4);
}
