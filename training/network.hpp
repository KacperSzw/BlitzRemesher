#pragma once
#include "neural/numeric.hpp"
#include "training/data.hpp"
#include <ATen/Context.h>
#include <torch/torch.h>
namespace blitz::neural::training {
struct NetworkImpl : torch::nn::Module {
    std::vector<torch::nn::Linear> layer;
    NetworkImpl() {
        for (int i = 0; i < 5; ++i)
            layer.push_back(register_module("layer" + std::to_string(i),
                                            torch::nn::Linear(layer_in[i], layer_out[i])));
    }
    torch::Tensor encode(torch::Tensor x, const torch::Tensor& from, const torch::Tensor& to,
                         const torch::Tensor& degree) {
        for (unsigned i = 0; i < 3; ++i) {
            auto mean = torch::zeros_like(x);
            mean.index_add_(0, from, x.index_select(0, to));
            x = torch::relu(layer[i]->forward(torch::cat({x, mean / degree}, 1)));
        }
        return x;
    }
    torch::Tensor head(torch::Tensor x, const torch::Tensor& c) {
        return layer[4]->forward(torch::relu(layer[3]->forward(torch::cat({x, c}, 1))));
    }
};
TORCH_MODULE(Network);
inline void ieee_fp32() {
    // Use only the legacy family supported by the pinned C++ distribution.
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setAllowTF32CuDNN(false);
}
inline void import_weights(Network& net, const WeightsData& w, torch::Device device) {
    if (w.values.size() != weight_count)
        throw std::invalid_argument("network weight dimensions");
    torch::NoGradGuard guard;
    size_t at = 0;
    for (auto& layer : net->layer)
        for (auto value : {layer->weight, layer->bias}) {
            auto t = torch::from_blob(const_cast<float*>(w.values.data()) + at, value.sizes(),
                                      torch::kFloat32)
                         .to(device);
            value.copy_(t);
            at += value.numel();
        }
}
inline std::vector<double> doubles(const torch::Tensor& t) {
    auto cpu = t.detach().to(torch::kCPU, torch::kFloat64).contiguous();
    return {cpu.data_ptr<double>(), cpu.data_ptr<double>() + cpu.numel()};
}
inline json difference_json(const NumericDifference& d) {
    return {{"finite", d.finite}, {"same_shape", d.same_shape}, {"max_abs", d.maximum},
            {"rms", d.rms},       {"max_relative", d.relative}, {"worst_index", d.worst}};
}
} // namespace blitz::neural::training
