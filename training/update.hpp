#pragma once
#include "training/action_network.hpp"
#include "training/packed_data.hpp"
#include "training/update_cuda.hpp"
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>

namespace blitz::neural::training {
inline void cuda_check(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}
template <class T> torch::Tensor device_records(std::span<const T> records, torch::Device device) {
    return torch::from_blob(const_cast<T*>(records.data()), {int64_t(records.size_bytes())},
                            torch::kUInt8)
        .to(device)
        .clone();
}
template <class T> T* records(const torch::Tensor& tensor) {
    return reinterpret_cast<T*>(tensor.data_ptr<uint8_t>());
}
struct SamplingTables {
    std::vector<SampleRange> categories, assets, bins;
    std::vector<uint32_t> states;
    SamplingTables(const std::vector<std::vector<uint32_t>>& groups,
                   const std::vector<std::array<std::vector<uint32_t>, 4>>& asset_bins,
                   uint32_t state_count) {
        if (groups.empty())
            throw std::invalid_argument("empty sampler");
        for (auto& category : groups) {
            if (category.empty())
                throw std::invalid_argument("empty sampler category");
            categories.push_back({uint32_t(assets.size()), uint32_t(category.size())});
            for (auto asset : category) {
                if (asset >= asset_bins.size())
                    throw std::invalid_argument("invalid sampler asset");
                SampleRange a{uint32_t(bins.size()), 0};
                for (auto& bin : asset_bins[asset])
                    if (!bin.empty()) {
                        if (states.size() + bin.size() > UINT32_MAX)
                            throw std::length_error("sampler exceeds u32");
                        for (auto s : bin)
                            if (s >= state_count)
                                throw std::invalid_argument("invalid sampler state");
                        bins.push_back({uint32_t(states.size()), uint32_t(bin.size())});
                        states.insert(states.end(), bin.begin(), bin.end());
                        ++a.count;
                    }
                if (!a.count)
                    throw std::invalid_argument("empty sampler asset");
                assets.push_back(a);
            }
        }
    }
};
// All tensors own their storage. Descriptors borrow tensor addresses and must be
// rebuilt if those addresses change. Loading copies into existing allocations.
class DeviceAdam {
  public:
    std::vector<torch::Tensor> parameters, mean, variance;
    torch::Tensor control, descriptors, partial;
    UpdateSettings settings;
    uint32_t total{};
    explicit DeviceAdam(std::vector<torch::Tensor> values, UpdateSettings config = {})
        : parameters(std::move(values)), settings(config) {
        if (parameters.empty())
            throw std::invalid_argument("optimizer has no parameters");
        auto device = parameters.front().device();
        std::vector<AdamParameter> table;
        if (!device.is_cuda() || !std::isfinite(settings.lr) || settings.lr <= 0 ||
            settings.lr > 1 || !std::isfinite(settings.decay) || settings.decay < 0 ||
            settings.decay > 1 || !std::isfinite(settings.max_norm) || settings.max_norm <= 0)
            throw std::invalid_argument("invalid device optimizer settings");
        for (auto& p : parameters) {
            if (p.device() != device || p.scalar_type() != torch::kFloat32 || !p.is_contiguous() ||
                uint64_t(total) + p.numel() > UINT32_MAX)
                throw std::invalid_argument("device optimizer parameter layout");
            p.mutable_grad() = torch::zeros_like(p);
            mean.push_back(torch::zeros_like(p));
            variance.push_back(torch::zeros_like(p));
            table.push_back({p.data_ptr<float>(), p.grad().data_ptr<float>(),
                             mean.back().data_ptr<float>(), variance.back().data_ptr<float>(),
                             uint32_t(p.numel()), total});
            total += uint32_t(p.numel());
        }
        control = torch::zeros({int64_t(sizeof(UpdateState))},
                               torch::TensorOptions().dtype(torch::kUInt8).device(device));
        descriptors = device_records<AdamParameter>(table, device);
        partial = torch::empty({int64_t((total + 255) / 256)},
                               parameters.front().options().requires_grad(false));
    }
    UpdateState state() const {
        auto host = control.cpu();
        UpdateState value;
        std::memcpy(&value, host.data_ptr(), sizeof(value));
        return value;
    }
    void begin_segment() {
        auto current = state();
        current.segment_start = current.step;
        current.first_loss = 0;
        control.copy_(torch::from_blob(&current, {int64_t(sizeof(current))}, torch::kUInt8));
    }
    void zero_grad() {
        for (auto& p : parameters)
            p.grad().zero_();
    }
    void update() {
        adam_update(records<AdamParameter>(descriptors), uint32_t(parameters.size()), total,
                    partial.data_ptr<float>(), records<UpdateState>(control), settings,
                    c10::cuda::getCurrentCUDAStream());
        cuda_check(cudaGetLastError());
    }
    std::vector<torch::Tensor> snapshot() const {
        std::vector<torch::Tensor> out;
        for (auto& p : parameters)
            out.push_back(p.detach().clone());
        for (auto& p : mean)
            out.push_back(p.clone());
        for (auto& p : variance)
            out.push_back(p.clone());
        out.push_back(control.clone());
        return out;
    }
    void restore(const std::vector<torch::Tensor>& saved) {
        torch::NoGradGuard guard;
        if (saved.size() != parameters.size() * 3 + 1)
            throw std::invalid_argument("optimizer snapshot shape");
        size_t i = 0;
        for (auto* group : {&parameters, &mean, &variance})
            for (auto& p : *group) {
                if (p.sizes() != saved[i].sizes() || p.scalar_type() != saved[i].scalar_type())
                    throw std::invalid_argument("optimizer snapshot tensor shape");
                p.copy_(saved[i++]);
            }
        control.copy_(saved[i]);
    }
    void save(torch::serialize::OutputArchive& archive) {
        archive.write("version", torch::tensor(int64_t(update_checkpoint_version)));
        archive.write("sampler", torch::tensor(int64_t(sampler_version)));
        archive.write("control", control);
        for (size_t i = 0; i < mean.size(); ++i) {
            archive.write("mean" + std::to_string(i), mean[i]);
            archive.write("variance" + std::to_string(i), variance[i]);
        }
    }
    void load(torch::serialize::InputArchive& archive) {
        torch::Tensor v, s;
        archive.read("version", v);
        archive.read("sampler", s);
        if (v.item<int64_t>() != update_checkpoint_version || s.item<int64_t>() != sampler_version)
            throw std::invalid_argument("incompatible device optimizer checkpoint");
        auto copy = [&](const std::string& name, torch::Tensor& target) {
            torch::Tensor value;
            archive.read(name, value);
            if (value.sizes() != target.sizes() || value.scalar_type() != target.scalar_type())
                throw std::invalid_argument("optimizer checkpoint tensor layout: " + name +
                                            " got " + c10::str(value.sizes()) + " expected " +
                                            c10::str(target.sizes()));
            target.copy_(value);
        };
        copy("control", control);
        for (size_t i = 0; i < mean.size(); ++i) {
            copy("mean" + std::to_string(i), mean[i]);
            copy("variance" + std::to_string(i), variance[i]);
        }
        auto current = state();
        if (current.failure || current.segment_start > current.step)
            throw std::invalid_argument("failed or invalid optimizer checkpoint");
        for (auto* group : {&parameters, &mean, &variance})
            for (auto& p : *group)
                if (!torch::isfinite(p).all().item<bool>())
                    throw std::invalid_argument("nonfinite optimizer checkpoint");
    }
};
class ActionUpdate {
    ActionNetwork& model_;
    torch::Tensor x_, flags_, conditions_, labels_, placements_, categories_, assets_, bins_,
        states_, losses_, derivative_;
    SamplingView sampler_{};
    uint32_t batch_, width_, outputs_;
    std::unique_ptr<at::cuda::CUDAGraph> graph_;

  public:
    DeviceAdam optimizer;
    torch::Tensor input, target, ids, prediction;
    ~ActionUpdate() {
        cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
        graph_.reset();
    }
    ActionUpdate(ActionNetwork& model, torch::Tensor x, torch::Tensor labels,
                 const SamplingTables& tables, uint32_t batch, uint32_t seed,
                 UpdateSettings settings = {}, torch::Tensor placements = {},
                 torch::Tensor flags = {}, torch::Tensor conditions = {})
        : model_(model), x_(std::move(x)), flags_(std::move(flags)),
          conditions_(std::move(conditions)), labels_(std::move(labels)),
          placements_(std::move(placements)), batch_(batch),
          width_(policy_inputs(model->architecture)), outputs_(policy_outputs(model->architecture)),
          optimizer(model->parameters(), settings) {
        if (!batch || batch > 4096 || x_.dim() != 3 || x_.size(0) < 1 ||
            x_.size(1) != action_pool ||
            x_.size(2) != (flags_.defined() ? packed_width(width_) : width_) ||
            labels_.sizes() != x_.sizes().slice(0, 2))
            throw std::invalid_argument("resident update dimensions");
        if (!x_.is_cuda() || !x_.is_contiguous() || x_.scalar_type() != torch::kFloat32 ||
            labels_.device() != x_.device() || labels_.scalar_type() != torch::kUInt8 ||
            !labels_.is_contiguous() || model_->parameters().front().device() != x_.device() ||
            (placements_.defined() != (is_placement_schema(model->architecture))))
            throw std::invalid_argument("resident update storage");
        if (is_placement_schema(model->architecture) &&
            (!placements_.defined() ||
             placements_.sizes() != torch::IntArrayRef({x_.size(0), action_pool, 9}) ||
             placements_.device() != x_.device() || placements_.scalar_type() != torch::kFloat32 ||
             !placements_.is_contiguous()))
            throw std::invalid_argument("resident placement targets");
        if (flags_.defined() &&
            (!conditions_.defined() || flags_.device() != x_.device() ||
             conditions_.device() != x_.device() || flags_.sizes() != labels_.sizes() ||
             conditions_.sizes() !=
                 torch::IntArrayRef({x_.size(0), action_condition_width(model->architecture)}) ||
             flags_.scalar_type() != torch::kInt32 ||
             conditions_.scalar_type() != torch::kFloat32 || !flags_.is_contiguous() ||
             !conditions_.is_contiguous()))
            throw std::invalid_argument("packed resident storage");
        auto device = x_.device();
        categories_ = device_records<SampleRange>(tables.categories, device);
        assets_ = device_records<SampleRange>(tables.assets, device);
        bins_ = device_records<SampleRange>(tables.bins, device);
        states_ = device_records<uint32_t>(tables.states, device);
        sampler_ = {records<SampleRange>(categories_),  records<SampleRange>(assets_),
                    records<SampleRange>(bins_),        records<uint32_t>(states_),
                    uint32_t(tables.categories.size()), seed};
        input = torch::zeros({batch, action_pool, width_}, x_.options());
        target = torch::zeros({batch, action_pool}, labels_.options());
        ids = torch::zeros({batch}, x_.options().dtype(torch::kInt32));
        losses_ = torch::empty({batch}, x_.options());
        derivative_ = torch::zeros({batch, action_pool, outputs_}, x_.options());
    }
    void eager() {
        auto stream = c10::cuda::getCurrentCUDAStream();
        if (flags_.defined())
            sample_packed_update(sampler_, x_.data_ptr<float>(),
                                 reinterpret_cast<uint32_t*>(flags_.data_ptr<int32_t>()),
                                 conditions_.data_ptr<float>(), labels_.data_ptr<uint8_t>(),
                                 input.data_ptr<float>(), target.data_ptr<uint8_t>(),
                                 reinterpret_cast<uint32_t*>(ids.data_ptr<int32_t>()),
                                 records<UpdateState>(optimizer.control), batch_, action_pool,
                                 width_, stream, action_condition_width(model_->architecture));
        else
            sample_update(sampler_, x_.data_ptr<float>(), labels_.data_ptr<uint8_t>(),
                          input.data_ptr<float>(), target.data_ptr<uint8_t>(),
                          reinterpret_cast<uint32_t*>(ids.data_ptr<int32_t>()),
                          records<UpdateState>(optimizer.control), batch_, action_pool, width_,
                          stream);
        optimizer.zero_grad();
        prediction = model_->forward(input);
        if (placements_.defined())
            placement_loss_update(prediction.data_ptr<float>(), target.data_ptr<uint8_t>(),
                                  placements_.data_ptr<float>(),
                                  reinterpret_cast<uint32_t*>(ids.data_ptr<int32_t>()),
                                  derivative_.data_ptr<float>(), losses_.data_ptr<float>(),
                                  records<UpdateState>(optimizer.control), batch_, action_pool,
                                  optimizer.settings, stream);
        else
            loss_update(prediction.data_ptr<float>(), target.data_ptr<uint8_t>(),
                        derivative_.data_ptr<float>(), losses_.data_ptr<float>(),
                        records<UpdateState>(optimizer.control), batch_, action_pool,
                        optimizer.settings, stream);
        prediction.backward(derivative_);
        optimizer.update();
    }
    void capture() {
        if (graph_)
            throw std::logic_error("update already captured");
        auto saved = optimizer.snapshot();
        cuda_check(cudaDeviceSynchronize());
        {
            c10::cuda::CUDAStreamGuard stream(c10::cuda::getStreamFromPool());
            for (unsigned i = 0; i < 3; ++i)
                eager();
            cuda_check(cudaStreamSynchronize(stream.current_stream()));
            optimizer.restore(saved);
            graph_ = std::make_unique<at::cuda::CUDAGraph>();
            graph_->capture_begin();
            eager();
            graph_->capture_end();
            optimizer.restore(saved);
            cuda_check(cudaStreamSynchronize(stream.current_stream()));
        }
    }
    void run(uint32_t count) {
        for (uint32_t i = 0; i < count; ++i) {
            if (graph_)
                graph_->replay();
            else
                eager();
        }
    }
    bool captured() const {
        return bool(graph_);
    }
    torch::Tensor inputs(uint32_t first, uint32_t count, const torch::Tensor& ids = {}) const {
        if (!count || uint64_t(first) + count > uint64_t(x_.size(0)))
            throw std::invalid_argument("dataset input range");
        if (ids.defined() &&
            (ids.numel() != count || ids.scalar_type() != torch::kInt64 ||
             ids.device() != x_.device() || !ids.is_contiguous() || ids.min().item<int64_t>() < 0 ||
             ids.max().item<int64_t>() >= x_.size(0)))
            throw std::invalid_argument("dataset input indices");
        if (!flags_.defined())
            return ids.defined() ? x_.index_select(0, ids) : x_.slice(0, first, first + count);
        auto result = torch::empty({count, action_pool, width_}, x_.options());
        expand_actions(
            x_.data_ptr<float>(), reinterpret_cast<uint32_t*>(flags_.data_ptr<int32_t>()),
            conditions_.data_ptr<float>(), labels_.data_ptr<uint8_t>(), result.data_ptr<float>(),
            ids.defined() ? ids.data_ptr<int64_t>() : nullptr, first, count, action_pool, width_,
            c10::cuda::getCurrentCUDAStream(), action_condition_width(model_->architecture));
        cuda_check(cudaGetLastError());
        return result;
    }
    size_t resident_bytes() const {
        size_t total = 0;
        for (auto* tensor : {&x_, &flags_, &conditions_, &labels_, &placements_})
            if (tensor->defined())
                total += size_t(tensor->nbytes());
        return total;
    }
};
} // namespace blitz::neural::training
