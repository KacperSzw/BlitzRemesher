#include "neural/numeric.hpp"

namespace blitz::neural {
void validate_graph(const Graph& g) {
    const size_t n = g.size();
    if (n > UINT32_MAX || g.x.size() != n * features || g.offsets.size() != n + 1 ||
        g.offsets[0] != 0 || g.offsets.back() != g.neighbors.size() ||
        !std::is_sorted(g.offsets.begin(), g.offsets.end()))
        throw std::invalid_argument("invalid neural graph dimensions");
    for (auto id : g.neighbors)
        if (id >= n)
            throw std::invalid_argument("neural neighbor outside graph");
    for (float x : g.x)
        if (!std::isfinite(x))
            throw std::invalid_argument("nonfinite neural feature");
}
NumericDifference numeric_difference(std::span<const double> a, std::span<const double> b) {
    NumericDifference d;
    d.same_shape = a.size() == b.size();
    if (!d.same_shape)
        return d;
    long double squares = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        // std::max(0, NaN) is zero: finiteness must be tested before reducing.
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            d.finite = false;
            d.worst = i;
            continue;
        }
        const double error = std::abs(a[i] - b[i]);
        if (!std::isfinite(error)) {
            d.finite = false;
            d.worst = i;
            continue;
        }
        if (error > d.maximum) {
            d.maximum = error;
            d.worst = i;
        }
        squares += static_cast<long double>(error) * error;
        const double scale = std::max(std::abs(a[i]), std::abs(b[i]));
        if (scale > 1e-12)
            d.relative = std::max(d.relative, error / scale);
    }
    d.rms = a.empty() ? 0 : std::sqrt(double(squares / a.size()));
    return d;
}
bool legacy_numeric_pass(const NumericDifference& d, double absolute) {
    return d.finite && d.same_shape && std::isfinite(absolute) && absolute >= 0 &&
           d.maximum <= absolute;
}
NumericTrace reference_fp64(const Graph& g, const std::array<float, conditions>& c,
                            const WeightsData& w) {
    validate_graph(g);
    if (w.values.size() != weight_count)
        throw std::invalid_argument("reference weight dimensions");
    for (float x : w.values)
        if (!std::isfinite(x))
            throw std::invalid_argument("nonfinite reference weight");
    for (float x : c)
        if (!std::isfinite(x))
            throw std::invalid_argument("nonfinite reference condition");
    const size_t n = g.size();
    std::vector<double> x(g.x.begin(), g.x.end());
    NumericTrace trace;
    size_t at = 0;
    auto record = [&](std::string name, uint32_t width, const std::vector<double>& v) {
        trace.push_back({std::move(name), width, v});
    };
    for (unsigned layer = 0; layer < 5; ++layer) {
        const uint32_t in = layer_in[layer], out = layer_out[layer];
        std::vector<double> input(n * in);
        const auto prefix = "layer" + std::to_string(layer) + "/";
        if (layer < 3) {
            const uint32_t width = in / 2;
            std::vector<double> sums(n * width);
            for (size_t v = 0; v < n; ++v)
                for (uint32_t j = 0; j < width; ++j) {
                    // FP64 with compensated summation; no production aggregation helper is reused.
                    double sum = 0, correction = 0;
                    for (uint32_t k = g.offsets[v]; k < g.offsets[v + 1]; ++k) {
                        const double value = x[size_t(g.neighbors[k]) * width + j] - correction;
                        const double next = sum + value;
                        correction = (next - sum) - value;
                        sum = next;
                    }
                    sums[v * width + j] = sum;
                    input[v * in + j] = x[v * width + j];
                    input[v * in + width + j] = sum / std::max(1u, g.offsets[v + 1] - g.offsets[v]);
                }
            record(prefix + "sum", width, sums);
        } else if (layer == 3) {
            for (size_t v = 0; v < n; ++v) {
                std::copy_n(x.data() + v * hidden, hidden, input.data() + v * in);
                std::copy(c.begin(), c.end(), input.data() + v * in + hidden);
            }
        } else
            input = x;
        record(prefix + "input", in, input);
        std::vector<double> gemm(n * out), pre(n * out);
        x.resize(n * out);
        for (size_t v = 0; v < n; ++v)
            for (uint32_t j = 0; j < out; ++j) {
                double sum = 0, correction = 0;
                for (uint32_t k = 0; k < in; ++k) {
                    const double value =
                        input[v * in + k] * double(w.values[at + size_t(j) * in + k]) - correction;
                    const double next = sum + value;
                    correction = (next - sum) - value;
                    sum = next;
                }
                gemm[v * out + j] = sum;
                pre[v * out + j] = sum + w.values[at + size_t(in) * out + j];
                x[v * out + j] = layer == 4 ? pre[v * out + j] : std::max(0., pre[v * out + j]);
            }
        record(prefix + "gemm", out, gemm);
        record(prefix + "bias", out, pre);
        record(prefix + "output", out, x);
        at += size_t(out) * (in + 1);
    }
    return trace;
}
} // namespace blitz::neural
