#pragma once
#include "neural/numeric.hpp"
#include "training/action_network.hpp"
namespace blitz::neural::training {
inline void diagnose_checkpoint(const fs::path& directory, const fs::path& output) {
    auto w = load_weights(directory / "model.blzn");
    torch::Tensor input, expected;
    torch::load(input, (directory / "input.pt").string());
    torch::load(expected, (directory / "expected.pt").string());
    input = input.cpu().contiguous();
    expected = expected.cpu().contiguous();
    const size_t rows = input.numel() / policy_inputs(w.architecture);
    std::vector<double> precise(input.data_ptr<float>(), input.data_ptr<float>() + input.numel());
    std::vector<float> ordinary(precise.begin(), precise.end()), compensated = ordinary;
    size_t at = 0;
    json layers = json::array();
    for (unsigned l = 0; l < 3; ++l) {
        uint32_t in = l ? w.hidden_width : policy_inputs(w.architecture),
                 out = l == 2 ? policy_outputs(w.architecture) : w.hidden_width;
        std::vector<double> reference(rows * out);
        std::vector<float> fp32(rows * out), kahan(rows * out);
        double largest_sum = 0;
        for (size_t row = 0; row < rows; ++row)
            for (uint32_t j = 0; j < out; ++j) {
                double sum = w.values[at + size_t(in) * out + j];
                float f = float(sum), k = float(sum), correction = 0;
                double absolute = std::abs(sum);
                for (uint32_t c = 0; c < in; ++c) {
                    auto weight = w.values[at + size_t(j) * in + c];
                    sum += precise[row * in + c] * weight;
                    absolute += std::abs(precise[row * in + c] * weight);
                    f += ordinary[row * in + c] * weight;
                    float term = compensated[row * in + c] * weight - correction, next = k + term;
                    correction = (next - k) - term;
                    k = next;
                }
                largest_sum = std::max(largest_sum, absolute);
                reference[row * out + j] = l < 2 ? std::max(0., sum) : sum;
                fp32[row * out + j] = l < 2 ? std::max(0.f, f) : f;
                kahan[row * out + j] = l < 2 ? std::max(0.f, k) : k;
            }
        layers.push_back(
            {{"layer", l},
             {"largest_absolute_dot_sum", largest_sum},
             {"sequential_fp32", difference_json(numeric_difference(
                                     std::vector<double>(fp32.begin(), fp32.end()), reference))},
             {"compensated_fp32",
              difference_json(numeric_difference(std::vector<double>(kahan.begin(), kahan.end()),
                                                 reference))}});
        precise = std::move(reference);
        ordinary = std::move(fp32);
        compensated = std::move(kahan);
        at += size_t(out) * (in + 1);
    }
    auto diff = numeric_difference(doubles(expected), precise);
    auto worst = diff.worst;
    json report = {{"layers", layers},
                   {"saved_gpu", difference_json(diff)},
                   {"worst_expected", expected.flatten()[int64_t(worst)].item<float>()},
                   {"worst_fp64", precise[worst]},
                   {"worst_compensated", compensated[worst]},
                   {"model_sha256", file_sha256(directory / "model.blzn")}};
    bool native_pass = true;
    report["native_checked"] = torch::cuda::is_available();
    if (report["native_checked"].get<bool>()) {
        NeuralOptions options;
        options.memory_mib = 128;
        ActionCuda gpu(w, options, 32);
        auto native = gpu.predict({input.data_ptr<float>(), size_t(input.numel())});
        auto native_difference =
            numeric_difference(std::vector<double>(native.begin(), native.end()), precise);
        report["compensated_cuda"] = difference_json(native_difference);
        native_pass = legacy_numeric_pass(native_difference);
        report["compensated_cuda_pass"] = native_pass;
    } else
        report["compensated_cuda_status"] = "skipped: CUDA unavailable; CPU diagnosis retained";
    write_json(output, report);
    std::cout << report.dump(2) << '\n';
    if (!native_pass)
        throw std::runtime_error("compensated CUDA failed the unchanged FP64 limit");
}
} // namespace blitz::neural::training
