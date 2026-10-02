#pragma once
#include "training/update.hpp"
#include <fcntl.h>
#include <future>
#include <map>
#include <system_error>
#include <unistd.h>
namespace blitz::neural::training {
inline ModelUse checkpoint_model_use(torch::serialize::InputArchive& all) {
    torch::serialize::InputArchive optimizer;
    all.read("optimizer", optimizer);
    torch::Tensor version, scope, use;
    optimizer.read("version", version);
    const auto v = version.item<int64_t>();
    if (v != update_checkpoint_version && v != endpoint_checkpoint_version)
        throw std::invalid_argument("unsupported model checkpoint optimizer version");
    const bool endpoint = v == endpoint_checkpoint_version;
    if (endpoint) {
        optimizer.read("trainable_scope", scope);
        if (scope.item<int64_t>() != int64_t(TrainableScope::EndpointScorer))
            throw std::invalid_argument("invalid endpoint checkpoint scope");
    }
    if (!all.try_read("model_use", use))
        return endpoint ? ModelUse::EndpointReuseOnly : ModelUse::Unrestricted;
    const auto value = use.item<int64_t>();
    if (value < 0 || value > int64_t(ModelUse::EndpointReuseOnly) || (endpoint && value != 1))
        throw std::invalid_argument("invalid checkpoint model use");
    return ModelUse(value);
}
inline void restore_model(torch::serialize::InputArchive& archive, ActionNetwork& model,
                          torch::Device device) {
    ActionNetwork restored(model->architecture, model->hidden_width);
    restored->to(device);
    restored->load(archive);
    torch::NoGradGuard guard;
    auto current = model->parameters(), loaded = restored->parameters();
    for (size_t i = 0; i < current.size(); ++i) {
        if (current[i].sizes() != loaded[i].sizes())
            throw std::invalid_argument("checkpoint model dimensions");
        current[i].copy_(loaded[i]);
    }
}
inline void load_checkpoint_model(const fs::path& path, ActionNetwork& model,
                                  torch::Device device) {
    torch::serialize::InputArchive all, m;
    all.load_from(path.string(), device);
    const auto use = checkpoint_model_use(all);
    all.read("model", m);
    restore_model(m, model, device);
    model->use = use;
}
inline void save_state(const fs::path& path, ActionNetwork& model, DeviceAdam& optimizer,
                       uint64_t step) {
    torch::serialize::OutputArchive all, m, o;
    model->save(m);
    optimizer.save(o);
    all.write("model", m);
    all.write("optimizer", o);
    all.write("model_use",
              torch::tensor(int64_t(trained_model_use(model, optimizer.settings.scope))));
    all.write("step", torch::tensor(int64_t(step)));
    auto temp = path;
    temp += ".part";
    all.save_to(temp.string());
    fs::rename(temp, path);
}
inline uint64_t load_state(const fs::path& path, ActionNetwork& model, DeviceAdam& optimizer,
                           torch::Device device) {
    torch::serialize::InputArchive all, m, o;
    all.load_from(path.string(), device);
    const auto use = checkpoint_model_use(all);
    if ((optimizer.settings.scope == TrainableScope::Joint && use == ModelUse::EndpointReuseOnly) ||
        (optimizer.settings.scope == TrainableScope::EndpointScorer &&
         use != ModelUse::EndpointReuseOnly))
        throw std::invalid_argument("checkpoint model use differs from update scope");
    all.read("model", m);
    all.read("optimizer", o);
    restore_model(m, model, device);
    optimizer.load(o);
    model->use = use;
    torch::Tensor step;
    all.read("step", step);
    auto n = step.item<int64_t>();
    if (n < 0 || uint64_t(n) != optimizer.state().step)
        throw std::invalid_argument("checkpoint counter mismatch");
    return uint64_t(n);
}
inline void sync_checkpoint_file(const fs::path& path) {
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0)
        throw std::system_error(errno, std::generic_category(), "open checkpoint for durability");
    int status = ::fsync(fd), error = errno;
    ::close(fd);
    if (status)
        throw std::system_error(error, std::generic_category(), "sync checkpoint");
}
// One immutable CPU snapshot in flight. Local readiness and durable publication
// are separate: teacher refresh never waits on network storage. Only this worker
// publishes recovery journals while the cycle is active.
class CheckpointWriter {
    std::future<json> pending_;
    std::shared_future<fs::path> local_ready_;
    fs::path scratch_, last_local_;
    std::function<void()> before_publish_;
    json totals_ = {{"snapshot_seconds", 0.}, {"serialize_seconds", 0.},
                    {"verify_seconds", 0.},   {"publish_seconds", 0.},
                    {"wait_seconds", 0.},     {"local_wait_seconds", 0.},
                    {"published", 0},         {"bytes", 0}};
    static double elapsed(std::chrono::steady_clock::time_point start) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }

  public:
    explicit CheckpointWriter(fs::path scratch = {}, std::function<void()> before_publish = {})
        : scratch_(std::move(scratch)), before_publish_(std::move(before_publish)) {}
    ~CheckpointWriter() {
        if (pending_.valid())
            pending_.wait();
    }
    void join() {
        if (pending_.valid()) {
            auto t = std::chrono::steady_clock::now();
            auto done = pending_.get();
            totals_["wait_seconds"] = totals_["wait_seconds"].get<double>() + elapsed(t);
            for (auto key : {"serialize_seconds", "verify_seconds", "publish_seconds"})
                totals_[key] = totals_[key].get<double>() + done.at(key).get<double>();
            totals_["published"] = totals_["published"].get<uint64_t>() + 1;
            totals_["bytes"] = totals_["bytes"].get<uint64_t>() + done.at("bytes").get<uint64_t>();
        }
    }
    fs::path local_checkpoint() {
        if (!local_ready_.valid())
            return {};
        auto t = std::chrono::steady_clock::now();
        auto p = local_ready_.get();
        totals_["local_wait_seconds"] = totals_["local_wait_seconds"].get<double>() + elapsed(t);
        return p;
    }
    json stats() const {
        return totals_;
    }
    bool busy() {
        if (!pending_.valid())
            return false;
        if (pending_.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
            return true;
        join();
        return false;
    }
    void submit(const fs::path& path, ActionNetwork& model, DeviceAdam& optimizer,
                const json& provenance, torch::Tensor input = {}, torch::Tensor expected = {},
                fs::path journal_path = {}, json journal = {}, torch::Tensor native = {}) {
        join();
        if (!scratch_.empty() && !last_local_.empty())
            fs::remove_all(last_local_);
        auto began = std::chrono::steady_clock::now();
        torch::NoGradGuard guard;
        ActionNetwork frozen(model->architecture, model->hidden_width);
        frozen->use = trained_model_use(model, optimizer.settings.scope);
        auto source = model->parameters(), target = frozen->parameters();
        for (size_t i = 0; i < source.size(); ++i)
            target[i].copy_(source[i].detach().cpu());
        auto control = optimizer.control.cpu().clone();
        std::vector<torch::Tensor> mean, variance;
        for (auto& v : optimizer.mean)
            mean.push_back(v.cpu().clone());
        for (auto& v : optimizer.variance)
            variance.push_back(v.cpu().clone());
        auto step = optimizer.state().step;
        if (input.defined())
            input = input.detach().cpu().clone();
        if (expected.defined())
            expected = expected.detach().cpu().clone();
        if (native.defined())
            native = native.detach().cpu().clone();
        totals_["snapshot_seconds"] = totals_["snapshot_seconds"].get<double>() + elapsed(began);
        auto ready = std::make_shared<std::promise<fs::path>>();
        local_ready_ = ready->get_future().share();
        last_local_ = scratch_.empty() ? path : scratch_ / path.filename();
        auto local = last_local_;
        auto hook = before_publish_;
        pending_ = std::async(
            std::launch::async,
            [path, local, hook, ready, frozen = std::move(frozen), control = std::move(control),
             mean = std::move(mean), variance = std::move(variance), step, provenance,
             input = std::move(input), expected = std::move(expected), native = std::move(native),
             scope = optimizer.settings.scope, journal_path = std::move(journal_path),
             journal = std::move(journal)]() mutable -> json {
                try {
                    auto start = std::chrono::steady_clock::now();
                    torch::NoGradGuard guard;
                    fs::create_directories(local);
                    torch::serialize::OutputArchive all, m, o;
                    frozen->save(m);
                    write_update_header(o, scope);
                    o.write("control", control);
                    for (size_t i = 0; i < mean.size(); ++i) {
                        o.write("mean" + std::to_string(i), mean[i]);
                        o.write("variance" + std::to_string(i), variance[i]);
                    }
                    all.write("model", m);
                    all.write("optimizer", o);
                    all.write("model_use", torch::tensor(int64_t(frozen->use)));
                    all.write("step", torch::tensor(int64_t(step)));
                    all.save_to((local / "checkpoint.pt.part").string());
                    fs::rename(local / "checkpoint.pt.part", local / "checkpoint.pt");
                    auto weights = export_actions(frozen, provenance);
                    save_weights(local / "model.blzn", weights);
                    if (input.defined()) {
                        torch::save(input, local / "input.pt");
                        torch::save(expected, local / "expected.pt");
                    }
                    if (native.defined())
                        torch::save(native, local / "native.pt");
                    double serialize = elapsed(start);
                    start = std::chrono::steady_clock::now();
                    // Keep the exact immutable state before any numerical verdict. A
                    // rejected snapshot is forensic evidence, never a recovery index.
                    if (input.defined()) {
                        auto flat = input.contiguous();
                        auto reference =
                            action_oracle({flat.data_ptr<float>(), size_t(flat.numel())}, weights);
                        auto difference = numeric_difference(doubles(expected), reference);
                        json verification = {{"fp64", difference_json(difference)},
                                             {"threshold", 2e-4}};
                        bool passed = legacy_numeric_pass(difference);
                        if (native.defined()) {
                            auto parity = numeric_difference(doubles(expected), doubles(native));
                            auto native_oracle = numeric_difference(doubles(native), reference);
                            verification["native"] = difference_json(parity);
                            verification["native_fp64"] = difference_json(native_oracle);
                            passed = passed && legacy_numeric_pass(parity) &&
                                     legacy_numeric_pass(native_oracle);
                        }
                        verification["passed"] = passed;
                        write_json(local / "verification.json", verification);
                        if (!passed)
                            throw std::runtime_error(
                                "checkpoint inference verification failed; exact state retained");
                    }
                    std::map<std::string, std::string> hashes;
                    uint64_t bytes = 0;
                    for (const auto& entry : fs::directory_iterator(local))
                        if (entry.is_regular_file() && entry.path().filename() != "index.json") {
                            hashes[entry.path().filename().string()] = file_sha256(entry.path());
                            bytes += entry.file_size();
                        }
                    auto checksum = hashes.at("checkpoint.pt");
                    write_json(local / "index.json", {{"complete", true},
                                                      {"step", step},
                                                      {"checkpoint_sha256", checksum},
                                                      {"model_sha256", hashes.at("model.blzn")},
                                                      {"files", hashes}});
                    double verify = elapsed(start);
                    ready->set_value(local);
                    if (hook)
                        hook();
                    start = std::chrono::steady_clock::now();
                    if (local != path) {
                        fs::create_directories(path);
                        for (const auto& [name, hash] : hashes) {
                            auto destination = path / name;
                            auto temporary = destination;
                            temporary += ".part";
                            fs::copy_file(local / name, temporary,
                                          fs::copy_options::overwrite_existing);
                            if (file_sha256(temporary) != hash)
                                throw std::runtime_error(
                                    "checkpoint publication checksum mismatch");
                            sync_checkpoint_file(temporary);
                            fs::rename(temporary, destination);
                        }
                        write_json(path / "index.json", read_json(local / "index.json"));
                    } else
                        for (const auto& [name, hash] : hashes)
                            sync_checkpoint_file(path / name);
                    sync_checkpoint_file(path / "index.json");
                    sync_checkpoint_file(path);
                    sync_checkpoint_file(path.parent_path());
                    if (!journal_path.empty()) {
                        journal["step"] = step;
                        journal["checkpoint"] =
                            fs::relative(path, journal_path.parent_path()).string();
                        journal["checkpoint_sha256"] = checksum;
                        auto temporary = journal_path;
                        temporary += ".durable";
                        write_json(temporary, journal);
                        sync_checkpoint_file(temporary);
                        fs::rename(temporary, journal_path);
                        sync_checkpoint_file(journal_path.parent_path());
                    }
                    return {{"serialize_seconds", serialize},
                            {"verify_seconds", verify},
                            {"publish_seconds", elapsed(start)},
                            {"bytes", bytes}};
                } catch (...) {
                    // Scratch is outside the collected cloud results. Retain failed
                    // numerical probes there too, without publishing a recovery index.
                    auto failure = std::current_exception();
                    std::string reason = "unknown checkpoint failure";
                    try {
                        std::rethrow_exception(failure);
                    } catch (const std::exception& e) {
                        reason = e.what();
                    } catch (...) {
                    }
                    try {
                        auto evidence = (journal_path.empty() ? path.parent_path()
                                                              : journal_path.parent_path()) /
                                        "checkpoint-failures" / path.filename();
                        fs::create_directories(evidence);
                        if (fs::exists(local))
                            for (const auto& file : fs::directory_iterator(local))
                                if (file.is_regular_file()) {
                                    auto destination = evidence / file.path().filename();
                                    fs::copy_file(file.path(), destination,
                                                  fs::copy_options::overwrite_existing);
                                    sync_checkpoint_file(destination);
                                }
                        write_json(evidence / "failure.json", {{"complete", false},
                                                               {"step", step},
                                                               {"error", reason},
                                                               {"recoverable", false}});
                        sync_checkpoint_file(evidence / "failure.json");
                        sync_checkpoint_file(evidence);
                    } catch (const std::exception& e) {
                        reason += "; retaining evidence failed: ";
                        reason += e.what();
                    }
                    try {
                        ready->set_exception(failure);
                    } catch (const std::future_error&) {
                    }
                    throw std::runtime_error(reason);
                }
            });
    }
};
} // namespace blitz::neural::training
