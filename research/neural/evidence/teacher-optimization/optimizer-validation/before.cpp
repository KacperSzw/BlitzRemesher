#include "training/update.hpp"
#include <iostream>
#include <limits>
#include <sstream>
using namespace blitz::neural::training;
int main() {
    torch::set_num_threads(2);
    auto gpu = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    UpdateSettings bad;
    bad.margin = std::numeric_limits<float>::quiet_NaN();
    bad.auxiliary = bad.penalty = 0;
    DeviceAdam invalid_config({torch::ones({1}, gpu).requires_grad_()}, bad);
    auto prediction = torch::zeros({1, 2, 3}, gpu);
    auto labels = torch::tensor({15, 11}, torch::kUInt8).view({1, 2}).to(torch::kCUDA);
    auto gradient = torch::ones_like(prediction), losses = torch::empty({1}, gpu);
    loss_update(prediction.data_ptr<float>(), labels.data_ptr<uint8_t>(),
                gradient.data_ptr<float>(), losses.data_ptr<float>(),
                records<UpdateState>(invalid_config.control), 1, 2, bad,
                c10::cuda::getCurrentCUDAStream());
    auto result = invalid_config.state();
    std::cout << "{\"case\":\"nan_margin\",\"constructor_accepted\":true,\"cuda_loss\":"
              << result.loss << ",\"gradient_abs_sum\":" << gradient.abs().sum().item<float>()
              << ",\"failure\":" << result.failure << ",\"reference_relu_isnan\":"
              << std::boolalpha << torch::isnan(torch::relu(torch::tensor(bad.margin))).item<bool>() << "}\n";
    DeviceAdam corrupt({torch::ones({1}, gpu).requires_grad_()});
    corrupt.variance[0].fill_(-1);
    torch::serialize::OutputArchive saved;
    corrupt.save(saved);
    std::stringstream bytes;
    saved.save_to(bytes);
    torch::serialize::InputArchive loaded;
    loaded.load_from(bytes, torch::Device(torch::kCUDA));
    DeviceAdam restored({torch::ones({1}, gpu).requires_grad_()});
    restored.load(loaded);
    restored.zero_grad();
    restored.update();
    auto state = restored.state();
    std::cout << "{\"case\":\"negative_saved_variance\",\"load_accepted\":true,\"parameter_finite\":"
              << torch::isfinite(restored.parameters[0]).all().item<bool>()
              << ",\"step\":" << state.step << ",\"failure\":" << state.failure << "}\n";
}
