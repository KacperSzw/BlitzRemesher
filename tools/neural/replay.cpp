#include "training/network.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
using namespace blitz;
using namespace blitz::neural;
using namespace blitz::neural::training;

int main(int argc, char** argv) {
    try {
        if (argc < 4)
            throw std::invalid_argument("blitz-neural-replay MODEL INPUT.bin REPORT.json "
                                        "[--example N] [--core N] [--cpu] [--strict]");
        uint32_t example = 0, core = UINT32_MAX;
        bool cpu = false, strict = false;
        for (int i = 4; i < argc; ++i) {
            std::string k = argv[i];
            if (k == "--cpu")
                cpu = true;
            else if (k == "--strict")
                strict = true;
            else if ((k == "--example" || k == "--core") && i + 1 < argc) {
                auto x = std::stoull(argv[++i]);
                if (x > UINT32_MAX)
                    throw std::invalid_argument("replay index overflow");
                (k == "--example" ? example : core) = uint32_t(x);
            } else
                throw std::invalid_argument("unknown replay option");
        }
        if (strict)
            ieee_fp32();
        torch::set_num_threads(4);
        torch::NoGradGuard guard;
        const auto weights = load_weights(argv[1]);
        auto asset = load_asset(argv[2]);
        if (example >= asset.examples.size())
            throw std::invalid_argument("example outside shard");
        auto g = core == UINT32_MAX ? std::move(asset.graph)
                                    : patch(asset.graph, {&core, 1}, 32768).graph;
        if (g.size() > 32768)
            throw std::length_error(
                "FP64 replay cap is 32768 vertices; select --core for a dataset shard");
        const auto c = asset.examples[example].condition;
        validate_graph(g);
        auto oracle = reference_fp64(g, c, weights);
        torch::Device device(cpu ? torch::kCPU : torch::kCUDA);
        if (!cpu && !torch::cuda::is_available())
            throw NeuralUnavailable("CUDA replay unavailable; use --cpu for oracle checks");
        Network net;
        net->to(device);
        net->eval();
        import_weights(net, weights, device);
        std::vector<int64_t> from, to;
        std::vector<float> degree;
        for (uint32_t i = 0; i < g.size(); ++i) {
            degree.push_back(float(std::max(1u, g.offsets[i + 1] - g.offsets[i])));
            for (auto k = g.offsets[i]; k < g.offsets[i + 1]; ++k) {
                from.push_back(i);
                to.push_back(g.neighbors[k]);
            }
        }
        auto copy = [&](auto& v, std::vector<int64_t> sizes, torch::ScalarType dtype) {
            return torch::from_blob(v.data(), sizes, dtype).clone().to(device);
        };
        auto x = copy(g.x, {int64_t(g.size()), features}, torch::kFloat32),
             src = copy(from, {int64_t(from.size())}, torch::kInt64),
             dst = copy(to, {int64_t(to.size())}, torch::kInt64),
             deg = copy(degree, {int64_t(g.size()), 1}, torch::kFloat32);
        auto cond = torch::from_blob(const_cast<float*>(c.data()), {1, conditions}, torch::kFloat32)
                        .clone()
                        .to(device)
                        .expand({int64_t(g.size()), conditions});
        NumericTrace torch_trace, native_trace;
        auto capture = [&](std::string name, const torch::Tensor& t) {
            torch_trace.push_back({std::move(name), uint32_t(t.size(1)), doubles(t)});
        };
        for (unsigned i = 0; i < 5; ++i) {
            auto prefix = "layer" + std::to_string(i) + "/";
            torch::Tensor input;
            if (i < 3) {
                auto sum = torch::zeros_like(x);
                sum.index_add_(0, src, x.index_select(0, dst));
                capture(prefix + "sum", sum);
                input = torch::cat({x, sum / deg}, 1);
            } else
                input = i == 3 ? torch::cat({x, cond}, 1) : x;
            capture(prefix + "input", input);
            capture(prefix + "gemm", torch::mm(input, net->layer[i]->weight.t()));
            auto pre = net->layer[i]->forward(input);
            capture(prefix + "bias", pre);
            x = i == 4 ? pre : torch::relu(pre);
            capture(prefix + "output", x);
        }
        json report = {{"training_started", false},
                       {"strict_fp32_requested", strict},
                       {"device", cpu ? "cpu" : "cuda"},
                       {"libtorch", TORCH_VERSION},
                       {"model_sha256", file_sha256(argv[1])},
                       {"input_sha256", file_sha256(argv[2])},
                       {"vertices", g.size()},
                       {"example", example},
                       {"layers", json::array()}};
        if (!cpu) {
            int runtime = 0, driver = 0;
            cudaRuntimeGetVersion(&runtime);
            cudaDriverGetVersion(&driver);
            report["cuda_runtime"] = runtime;
            report["cuda_driver"] = driver;
            auto encoded = encode_cuda(g, weights, {}, &native_trace);
            predict_cuda(encoded, c, weights, {}, &native_trace);
        }
        bool finite = true;
        for (const auto& layer : oracle) {
            auto find = [&](const NumericTrace& t) -> const NumericLayer* {
                for (auto& l : t)
                    if (l.name == layer.name)
                        return &l;
                return nullptr;
            };
            json row = {{"operation", layer.name}, {"width", layer.width}};
            auto a = find(torch_trace), b = find(native_trace);
            if (a) {
                auto d = numeric_difference(a->values, layer.values);
                finite &= d.finite && d.same_shape;
                row["torch_vs_fp64"] = difference_json(d);
            }
            if (b) {
                auto d = numeric_difference(b->values, layer.values);
                finite &= d.finite && d.same_shape;
                row["native_vs_fp64"] = difference_json(d);
            }
            if (a && b)
                row["torch_vs_native"] = difference_json(numeric_difference(a->values, b->values));
            report["layers"].push_back(row);
        }
        bool agreement = true;
        if (!cpu) {
            auto d = numeric_difference(torch_trace.back().values, native_trace.back().values);
            agreement = legacy_numeric_pass(d);
            report["legacy_gate_passed"] = agreement;
            report["output_difference"] = difference_json(d);
        }
        report["finite"] = finite;
        report["complete"] = true;
        write_json(argv[3], report);
        std::cout << report.dump(2) << '\n';
        return finite && agreement ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "neural replay: " << e.what() << '\n';
        return 1;
    }
}
