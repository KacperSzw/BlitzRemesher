#pragma once
#include "neural/linear.hpp"
#include "neural/placement.hpp"
#include "training/action_data.hpp"
#include "training/network.hpp"
#include <c10/cuda/CUDAStream.h>
namespace blitz::neural::training {
struct CompensatedLinear : torch::autograd::Function<CompensatedLinear> {
    static torch::Tensor forward(torch::autograd::AutogradContext* ctx, torch::Tensor input,
                                 torch::Tensor weight, torch::Tensor bias) {
        input = input.contiguous();
        weight = weight.contiguous();
        bias = bias.contiguous();
        ctx->save_for_backward({input, weight});
        auto sizes = input.sizes().vec();
        sizes.back() = weight.size(0);
        auto output = torch::empty(sizes, input.options());
        compensated_linear(input.data_ptr<float>(), weight.data_ptr<float>(),
                           bias.data_ptr<float>(), output.data_ptr<float>(),
                           uint32_t(input.numel() / weight.size(1)), uint32_t(weight.size(1)),
                           uint32_t(weight.size(0)), false, c10::cuda::getCurrentCUDAStream());
        return output;
    }
    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                   torch::autograd::variable_list gradients) {
        auto saved = ctx->get_saved_variables();
        auto x = saved[0].reshape({-1, saved[1].size(1)}),
             dy = gradients[0].reshape({-1, saved[1].size(0)});
        return {torch::matmul(dy, saved[1]).reshape(saved[0].sizes()), torch::matmul(dy.t(), x),
                dy.sum(0)};
    }
};
struct ActionNetworkImpl : torch::nn::Module {
    std::vector<torch::nn::Linear> layers;
    uint32_t architecture, hidden_width;
    ModelUse use{ModelUse::Unrestricted};
    explicit ActionNetworkImpl(uint32_t version = action_schema, uint32_t width = 64)
        : architecture(version), hidden_width(width) {
        if (!policy_weights(version, width))
            throw std::invalid_argument("unsupported action policy architecture");
        for (unsigned i = 0; i < 3; ++i)
            layers.push_back(
                register_module("layer" + std::to_string(i),
                                torch::nn::Linear(i ? width : policy_inputs(version),
                                                  i == 2 ? policy_outputs(version) : width)));
    }
    torch::Tensor forward(torch::Tensor x) {
        for (unsigned i = 0; i < 3; ++i) {
            x = x.is_cuda() ? CompensatedLinear::apply(x, layers[i]->weight, layers[i]->bias)
                            : layers[i]->forward(x);
            if (i < 2)
                x = torch::relu(x);
        }
        return x;
    }
};
TORCH_MODULE(ActionNetwork);
inline WeightsData export_actions(ActionNetwork& model, const json& provenance) {
    WeightsData w;
    w.architecture = model->architecture;
    w.hidden_width = model->hidden_width;
    w.use = model->use;
    w.provenance = provenance.dump();
    for (auto& p : model->parameters()) {
        auto cpu = p.detach().to(torch::kCPU).contiguous();
        w.values.insert(w.values.end(), cpu.data_ptr<float>(), cpu.data_ptr<float>() + cpu.numel());
    }
    return w;
}
inline torch::Tensor action_loss(const torch::Tensor& prediction, const torch::Tensor& labels,
                                 const torch::Tensor& valid, double margin = 1,
                                 double auxiliary = .25, double penalty = 1e-4) {
    auto preferred = labels.bitwise_and(Preferred).ne(0).logical_and(valid),
         other = preferred.logical_not().logical_and(valid);
    auto pairs = preferred.unsqueeze(2).logical_and(other.unsqueeze(1));
    auto scores = prediction.select(2, 0);
    auto ranking = (torch::relu(margin + scores.unsqueeze(1) - scores.unsqueeze(2)) * pairs).sum() /
                   pairs.sum().clamp_min(1);
    auto result = ranking;
    for (int h = 1; h < 3; ++h) {
        auto target =
            labels.bitwise_and(h == 1 ? SourcePass : AdjacentPass).ne(0).to(torch::kFloat32);
        result =
            result + auxiliary *
                         (torch::binary_cross_entropy_with_logits(prediction.select(2, h), target,
                                                                  {}, {}, torch::Reduction::None) *
                          valid)
                             .sum() /
                         valid.sum().clamp_min(1);
    }
    return result + penalty * (prediction.square() * valid.unsqueeze(2)).sum() /
                        (valid.sum().clamp_min(1) * action_outputs);
}
inline double action_membership(const torch::Tensor& prediction, const torch::Tensor& labels,
                                const torch::Tensor& valid) {
    auto preferred = labels.bitwise_and(Preferred).ne(0).logical_and(valid), has = preferred.any(1);
    if (!has.any().item<bool>())
        return 0;
    auto index =
        prediction.select(2, 0).masked_fill(valid.logical_not(), -INFINITY).argmax(1, true);
    return (preferred.gather(1, index).squeeze(1).logical_and(has)).sum().item<double>() /
           has.sum().item<double>();
}
inline std::vector<double> action_oracle(std::span<const float> x, const WeightsData& w) {
    auto width = policy_inputs(w.architecture), outputs = policy_outputs(w.architecture);
    if (!width || w.values.size() != policy_weights(w.architecture, w.hidden_width) ||
        x.size() % width)
        throw std::invalid_argument("action oracle dimensions");
    std::vector<double> input(x.begin(), x.end());
    const auto rows = x.size() / width;
    size_t at = 0;
    for (unsigned l = 0; l < 3; ++l) {
        auto in = l ? w.hidden_width : width, out = l == 2 ? outputs : w.hidden_width;
        std::vector<double> next(rows * out);
        for (size_t row = 0; row < rows; ++row)
            for (unsigned j = 0; j < out; ++j) {
                double sum = w.values[at + size_t(in) * out + j], correction = 0;
                for (unsigned k = 0; k < in; ++k) {
                    double term =
                               input[row * in + k] * w.values[at + size_t(j) * in + k] - correction,
                           next_sum = sum + term;
                    correction = (next_sum - sum) - term;
                    sum = next_sum;
                }
                next[row * out + j] = l < 2 ? std::max(0., sum) : sum;
            }
        input = std::move(next);
        at += size_t(out) * (in + 1);
    }
    return input;
}
inline WeightsData condition_policy(const WeightsData& original, std::span<const float> probe,
                                    json& verification) {
    if (original.architecture != placement_schema || probe.empty() ||
        probe.size() % placement_features)
        throw std::invalid_argument(
            "policy migration requires a v3 policy and recorded feature rows");
    if (original.values.size() != policy_weights(original.architecture, original.hidden_width))
        throw std::invalid_argument("migration model dimensions");
    auto migrated = original;
    migrated.architecture = conditioned_placement_schema;
    for (uint32_t row = 0; row < original.hidden_width; ++row) {
        auto* weights = migrated.values.data() + size_t(row) * placement_features;
        weights[47] = float(double(weights[47]) + weights[79]);
        weights[79] = 0;
        if (!std::isfinite(weights[47]))
            throw std::invalid_argument("migration weight overflow");
    }
    std::vector<float> conditioned(probe.begin(), probe.end());
    for (size_t row = 0; row < probe.size() / placement_features; ++row) {
        if (probe[row * placement_features + 79] != probe[row * placement_features + 47])
            throw std::invalid_argument("recorded v3 probe has inconsistent duplicate features");
        conditioned[row * placement_features + 79] = 1;
    }
    auto difference =
        numeric_difference(action_oracle(probe, original), action_oracle(conditioned, migrated));
    verification = {{"source_schema", placement_schema},
                    {"destination_schema", conditioned_placement_schema},
                    {"rows", probe.size() / placement_features},
                    {"difference", difference_json(difference)},
                    {"threshold", 2e-4},
                    {"passed", legacy_numeric_pass(difference)}};
    migrated.provenance = json{
        {"migration", "fold redundant feature 79 into 47; UV input starts at zero weight"},
        {"source_provenance", original.provenance},
        {"verification",
         verification}}.dump();
    if (!legacy_numeric_pass(difference))
        throw std::runtime_error("policy migration failed unchanged numerical gate");
    return migrated;
}
} // namespace blitz::neural::training
