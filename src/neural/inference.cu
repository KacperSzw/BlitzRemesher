#include "neural/cuda.cuh"
#include "neural/linear.hpp"
#include "neural/numeric.hpp"
#include "neural/placement.hpp"
namespace blitz {
bool neural_available(int32_t device) noexcept {
    int count = 0;
    auto e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return device >= 0 && device < count;
}
namespace neural {
using namespace gpu;
namespace {
__global__ void aggregate(const float* in, float* out, const uint32_t* offsets,
                          const uint32_t* neighbors, uint32_t n, uint32_t width) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(n) * width)
        return;
    auto vertex = uint32_t(i / width), channel = uint32_t(i % width), first = offsets[vertex],
         end = offsets[vertex + 1];
    float sum = 0;
    for (auto k = first; k < end; ++k)
        sum += in[size_t(neighbors[k]) * width + channel];
    out[size_t(vertex) * width * 2 + channel] = in[i];
    out[size_t(vertex) * width * 2 + width + channel] = sum / float(max(1u, end - first));
}
__global__ void activate(float* x, const float* bias, uint32_t n, uint32_t width, bool relu) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(n) * width)
        return;
    float v = x[i] + (bias ? bias[i % width] : 0);
    x[i] = relu ? fmaxf(0, v) : v;
}
__global__ void concatenate(const float* x, const float* c, float* out, uint32_t n) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(n) * (hidden + conditions))
        return;
    auto channel = i % (hidden + conditions);
    out[i] =
        channel < hidden ? x[i / (hidden + conditions) * hidden + channel] : c[channel - hidden];
}
void capture(NumericTrace* trace, const std::string& name, const float* data, uint32_t n,
             uint32_t width) {
    if (!trace)
        return;
    std::vector<float> host(size_t(n) * width);
    check(gpu::copy(host.data(), data, host.size() * sizeof(float), cudaMemcpyDeviceToHost));
    trace->push_back({name, width, {host.begin(), host.end()}});
}
void linear(cublasHandle_t handle, const float* x, const float* weights, float* y, uint32_t n,
            int in, int out, bool relu, NumericTrace* trace = nullptr, unsigned layer = 0) {
    const auto prefix = "layer" + std::to_string(layer) + "/";
    capture(trace, prefix + "input", x, n, in);
    float alpha = 1, beta = 0;
    check(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, out, int(n), in, &alpha, weights, in, x, in,
                      &beta, y, out));
    capture(trace, prefix + "gemm", y, n, out);
    activate<<<blocks(size_t(n) * out), 256, 0, gpu::stream()>>>(y, weights + size_t(in) * out, n,
                                                                 out, trace ? false : relu);
    check(cudaGetLastError());
    if (trace) {
        capture(trace, prefix + "bias", y, n, out);
        if (relu) {
            activate<<<blocks(size_t(n) * out), 256, 0, gpu::stream()>>>(y, nullptr, n, out, true);
            check(cudaGetLastError());
        }
    }
    capture(trace, prefix + "output", y, n, out);
}
} // namespace
__global__ void compensated_linear_kernel(const float* input, const float* weights,
                                          const float* bias, float* output, uint32_t rows,
                                          uint32_t in, uint32_t out, bool relu) {
    // Eight lanes read adjacent weights. Independent compensated partials are
    // combined in a fixed order, including their residuals and the bias.
    uint32_t i = (blockIdx.x * blockDim.x + threadIdx.x) / 8, lane = threadIdx.x % 8;
    auto mask = __ballot_sync(0xffffffffu, i < rows * out);
    if (i >= rows * out)
        return;
    uint32_t row = i / out, channel = i % out;
    float sum = 0, correction = 0;
    for (uint32_t k = lane; k < in; k += 8) {
        float term =
            __fmul_rn(input[size_t(row) * in + k], weights[size_t(channel) * in + k]) - correction;
        float next = __fadd_rn(sum, term);
        correction = __fsub_rn(__fsub_rn(next, sum), term);
        sum = next;
    }
    float total = bias[channel], error = 0;
    for (unsigned j = 0; j < 8; ++j) {
        float part = __shfl_sync(mask, sum, j, 8), residual = __shfl_sync(mask, correction, j, 8);
        if (lane == 0) {
            float term = __fsub_rn(part, error), next = __fadd_rn(total, term);
            error = __fsub_rn(__fsub_rn(next, total), term);
            total = next;
            term = __fsub_rn(-residual, error);
            next = __fadd_rn(total, term);
            error = __fsub_rn(__fsub_rn(next, total), term);
            total = next;
        }
    }
    if (lane == 0)
        output[i] = relu && total < 0 ? 0 : total;
}
void compensated_linear(const float* input, const float* weights, const float* bias, float* output,
                        uint32_t rows, uint32_t in, uint32_t out, bool relu, cudaStream_t stream) {
    if (!rows)
        return;
    compensated_linear_kernel<<<(size_t(rows) * out * 8 + 127) / 128, 128, 0, stream>>>(
        input, weights, bias, output, rows, in, out, relu);
    check(cudaGetLastError());
}
std::vector<float> encode_cuda(const Graph& g, const WeightsData& w, const NeuralOptions& options,
                               NumericTrace* trace) {
    validate_graph(g);
    if (g.size() > 262144 || w.values.size() != weight_count || g.x.size() != g.size() * features ||
        g.offsets.size() != g.size() + 1)
        throw std::invalid_argument("invalid CUDA graph dimensions");
    if (g.size() == 0)
        return {};
    Device d(options);
    Blas blas;
    uint32_t n = uint32_t(g.size());
    Buffer<uint32_t> offsets(d, g.offsets.size()), neighbors(d, g.neighbors.size());
    offsets.upload(g.offsets);
    neighbors.upload(g.neighbors);
    Buffer<float> weights(d, w.values.size()), input(d, g.x.size()), a(d, size_t(n) * hidden),
        b(d, size_t(n) * hidden), cat(d, size_t(n) * hidden * 2);
    weights.upload(w.values);
    input.upload(g.x);
    const float* x = input.p;
    float* y = a.p;
    size_t at = 0;
    for (unsigned layer = 0; layer < 3; ++layer) {
        uint32_t width = layer == 0 ? features : hidden;
        aggregate<<<blocks(size_t(n) * width), 256, 0, gpu::stream()>>>(x, cat.p, offsets.p,
                                                                        neighbors.p, n, width);
        check(cudaGetLastError());
        linear(blas, cat.p, weights.p + at, y, n, width * 2, hidden, true, trace, layer);
        at += size_t(hidden) * (width * 2 + 1);
        x = y;
        y = y == a.p ? b.p : a.p;
    }
    check(cudaStreamSynchronize(gpu::stream()));
    return a.download();
}
Prediction predict_cuda(std::span<const float> embedding,
                        const std::array<float, conditions>& condition, const WeightsData& w,
                        const NeuralOptions& options, NumericTrace* trace) {
    if (embedding.size() % hidden || w.values.size() != weight_count)
        throw std::invalid_argument("invalid embedding dimensions");
    Prediction result;
    result.values.resize(embedding.size() / hidden * outputs);
    Device d(options);
    Blas blas;
    Buffer<float> weights(d, w.values.size()), c(d, conditions);
    weights.upload(w.values);
    c.upload(condition);
    size_t at = 0;
    for (unsigned i = 0; i < 3; ++i)
        at += size_t(layer_out[i]) * (layer_in[i] + 1);
    for (size_t first = 0; first < embedding.size() / hidden; first += 65536) {
        uint32_t n = uint32_t(std::min<size_t>(65536, embedding.size() / hidden - first));
        Buffer<float> input(d, size_t(n) * hidden), cat(d, size_t(n) * (hidden + conditions)),
            h(d, size_t(n) * hidden), out(d, size_t(n) * outputs);
        input.upload(embedding.subspan(first * hidden, size_t(n) * hidden));
        concatenate<<<blocks(cat.n), 256, 0, gpu::stream()>>>(input.p, c.p, cat.p, n);
        check(cudaGetLastError());
        linear(blas, cat.p, weights.p + at, h.p, n, hidden + conditions, hidden, true, trace, 3);
        linear(blas, h.p, weights.p + at + hidden * (hidden + conditions + 1), out.p, n, hidden,
               outputs, false, trace, 4);
        check(gpu::copy(result.values.data() + first * outputs, out.p, out.n * sizeof(float),
                        cudaMemcpyDeviceToHost));
    }
    return result;
}
struct ActionCuda::Impl {
    Device device;
    Buffer<float> weights, input, a, b, output;
    uint32_t batch, architecture, hidden_width;
    int id;
    Impl(const WeightsData& w, const NeuralOptions& options, uint32_t count)
        : device(options), weights(device, w.values.size()),
          input(device, size_t(count) * policy_inputs(w.architecture)),
          a(device, size_t(count) * w.hidden_width), b(device, size_t(count) * w.hidden_width),
          output(device, size_t(count) * policy_outputs(w.architecture)), batch(count),
          architecture(w.architecture), hidden_width(w.hidden_width), id(options.device) {
        weights.upload(w.values);
        check(cudaSetDevice(device.previous));
    }
    ~Impl() {
        cudaSetDevice(id);
    }
};
ActionCuda::ActionCuda(const WeightsData& w, const NeuralOptions& options, uint32_t count) {
    if (!policy_weights(w.architecture, w.hidden_width) ||
        w.values.size() != policy_weights(w.architecture, w.hidden_width) || count < 1 ||
        count > 65536)
        throw std::invalid_argument("invalid action model or batch dimensions");
    for (float v : w.values)
        if (!std::isfinite(v))
            throw std::invalid_argument("nonfinite action weights");
    impl_ = std::make_unique<Impl>(w, options, count);
}
ActionCuda::~ActionCuda() {
    if (impl_) {
        int previous = 0;
        cudaGetDevice(&previous);
        impl_.reset();
        cudaSetDevice(previous);
    }
}
uint32_t ActionCuda::architecture() const {
    return impl_->architecture;
}
void ActionCuda::refresh_device(const float* weights, size_t count) {
    auto& p = *impl_;
    if (!weights || count != p.weights.n)
        throw std::invalid_argument("resident inference weight layout");
    NeuralOptions options;
    options.device = p.id;
    Device guard(options);
    check(gpu::copy(p.weights.p, weights, count * sizeof(float), cudaMemcpyDeviceToDevice));
}
void ActionCuda::predict_device(const float* input, float* output, uint32_t rows) {
    auto& p = *impl_;
    NeuralOptions options;
    options.device = p.id;
    Device guard(options);
    auto width = policy_inputs(p.architecture), outputs = policy_outputs(p.architecture);
    for (uint32_t row = 0; row < rows; row += p.batch) {
        auto n = std::min(p.batch, rows - row);
        const float* in = input + size_t(row) * width;
        size_t at = 0;
        for (unsigned layer = 0; layer < 3; ++layer) {
            auto columns = layer ? p.hidden_width : width,
                 channels = layer == 2 ? outputs : p.hidden_width;
            float* target = layer == 0   ? p.a.p
                            : layer == 1 ? p.b.p
                                         : output + size_t(row) * outputs;
            compensated_linear(in, p.weights.p + at, p.weights.p + at + size_t(columns) * channels,
                               target, n, columns, channels, layer < 2, gpu::stream());
            in = target;
            at += size_t(channels) * (columns + 1);
        }
    }
}
std::vector<float> ActionCuda::predict(std::span<const float> x) {
    auto& p = *impl_;
    auto width = policy_inputs(p.architecture), outputs = policy_outputs(p.architecture);
    if (x.size() % width)
        throw std::invalid_argument("invalid action input dimensions");
    for (float v : x)
        if (!std::isfinite(v))
            throw std::invalid_argument("nonfinite action input");
    NeuralOptions options;
    options.device = p.id;
    Device guard(options);
    std::vector<float> out(x.size() / width * outputs);
    for (size_t row = 0; row < x.size() / width; row += p.batch) {
        auto n = uint32_t(std::min<size_t>(p.batch, x.size() / width - row));
        check(gpu::copy(p.input.p, x.data() + row * width, size_t(n) * width * sizeof(float),
                        cudaMemcpyHostToDevice));
        const float* in = p.input.p;
        size_t at = 0;
        for (unsigned layer = 0; layer < 3; ++layer) {
            auto columns = layer ? p.hidden_width : width,
                 channels = layer == 2 ? outputs : p.hidden_width;
            float* target = layer == 0 ? p.a.p : layer == 1 ? p.b.p : p.output.p;
            compensated_linear(in, p.weights.p + at, p.weights.p + at + size_t(columns) * channels,
                               target, n, columns, channels, layer < 2, gpu::stream());
            in = target;
            at += size_t(channels) * (columns + 1);
        }
        check(gpu::copy(out.data() + row * outputs, p.output.p, size_t(n) * outputs * sizeof(float),
                        cudaMemcpyDeviceToHost));
    }
    for (float v : out)
        if (!std::isfinite(v))
            throw std::runtime_error("nonfinite action prediction");
    return out;
}
} // namespace neural
} // namespace blitz
