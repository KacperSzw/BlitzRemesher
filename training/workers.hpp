#pragma once
#include "neural/timeline.hpp"
#include "training/placement_teacher.hpp"
#include <condition_variable>
#include <deque>
#include <thread>
namespace blitz::neural::training {
struct TeacherJob {
    uint32_t id{};
    std::string asset, name;
    fs::path directory;
    PlacementRequest request;
    uint64_t submitted_ns{};
    bool allow_recovery{true};
};
struct TeacherResult {
    TeacherJob job;
    PlacementResult result;
    double seconds{}, device_seconds{};
    uint64_t started_ns{}, completed_ns{}, consumed_ns{}, wave{};
    uint8_t worker{};
    bool recovered{};
    std::exception_ptr error;
};
// A wave owns one immutable policy and at most two jobs per worker. Completion
// order never determines replay order. Threads own all their native resources.
class TeacherWorkers {
    NeuralOptions options_;
    MemoryBudget& budget_;
    uint32_t width_, architecture_;
    std::vector<std::thread> threads_;
    std::mutex mutex_;
    std::condition_variable changed_;
    std::deque<TeacherJob> queue_;
    std::map<uint32_t, TeacherResult> ready_;
    size_t in_flight_{};
    uint32_t ready_workers_{};
    bool stop_{};
    std::atomic<bool> cancelled_{};
    std::exception_ptr startup_error_;
    const float* weights_{};
    size_t weight_count_{};
    cudaEvent_t policy_ready_{};
    uint64_t wave_{};
    const std::chrono::steady_clock::time_point origin_{std::chrono::steady_clock::now()};
    uint64_t timestamp() const {
        return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - origin_)
                            .count());
    }
    void worker(uint8_t worker_id) {
        try {
            gpu::check(cudaSetDevice(options_.device));
            gpu::StreamScope stream;
            MemoryScope memory(budget_);
            AuditSession session(options_);
            WeightsData shape;
            shape.architecture = architecture_;
            shape.hidden_width = width_;
            shape.values.resize(policy_weights(shape.architecture, width_));
            ActionCuda policy(shape, options_, 1024);
            {
                std::lock_guard lock(mutex_);
                ++ready_workers_;
            }
            changed_.notify_all();
            uint64_t installed = 0;
            for (;;) {
                TeacherJob job;
                uint64_t wave;
                {
                    std::unique_lock lock(mutex_);
                    changed_.wait(lock, [&] { return stop_ || !queue_.empty(); });
                    if (stop_)
                        break;
                    job = std::move(queue_.front());
                    queue_.pop_front();
                    wave = wave_;
                }
                Timeline range(("teacher-" + std::to_string(job.id)).c_str());
                TeacherResult done;
                done.job = job;
                done.worker = worker_id;
                done.wave = wave;
                done.started_ns = timestamp();
                auto start = std::chrono::steady_clock::now();
                cudaEvent_t begin{}, end{};
                try {
                    if (installed != wave) {
                        gpu::check(cudaStreamWaitEvent(stream.value, policy_ready_));
                        policy.refresh_device(weights_, weight_count_);
                        installed = wave;
                    }
                    gpu::check(cudaEventCreate(&begin));
                    gpu::check(cudaEventCreate(&end));
                    gpu::check(cudaEventRecord(begin, stream.value));
                    auto cancel = job.request.cancelled;
                    job.request.cancelled = [&, cancel] {
                        return cancelled_ || (cancel && cancel());
                    };
                    bool recovered = false;
                    if (!job.allow_recovery && fs::exists(job.directory))
                        throw std::invalid_argument("fresh teacher job directory already exists");
                    if (fs::exists(job.directory / "index.json")) {
                        auto index = read_json(job.directory / "index.json"),
                             contract = read_json(job.directory / "contract.json");
                        if (index.at("complete") == true &&
                            index.at("reference_confirmed") == true &&
                            contract.at("policy_payload_sha256") == job.request.policy_sha256 &&
                            contract.at("seed") == job.request.seed &&
                            contract.at("states_requested") == job.request.states &&
                            contract.at("pixels") == job.request.pixels &&
                            contract.at("asset").at("id") == job.asset &&
                            contract.at("binary_sha256") == job.request.cache->binary_hash &&
                            contract.at("schema") == job.request.architecture &&
                            contract.at("source_limit") == job.request.source_limit &&
                            contract.at("adjacent_limit") == job.request.adjacent_limit &&
                            contract.at("previous_steps") == job.request.previous_steps &&
                            contract.at("preserve_uv") == job.request.preserve_uv &&
                            contract.at("policy_action_candidates") ==
                                job.request.policy_candidates &&
                            contract.at("policy_rollout_trials") ==
                                job.request.policy_rollout_trials &&
                            contract.at("previous_pixels") ==
                                (job.request.previous_steps
                                     ? (job.request.previous_pixels
                                            ? job.request.previous_pixels
                                            : std::min(512., job.request.pixels * 2))
                                     : job.request.pixels) &&
                            contract.at("pool") == job.request.pool &&
                            contract.at("training_profile") ==
                                training_profile_name(job.request.profile) &&
                            contract.at("episode").at("target_retained") == job.request.retained &&
                            contract.at("episode").at("kind") ==
                                (job.request.policy_rollout_trials ? "audited_policy_rollout"
                                 : job.request.simplifier          ? "audited_simplifier"
                                 : job.request.episode.empty()     ? "source"
                                                               : "audited_teacher_trajectory") &&
                            contract.at("episode").at("parent_sha256") ==
                                (job.request.episode.empty() ? ""
                                                             : file_sha256(job.request.episode))) {
                            if (index.at("contract_sha256") !=
                                    file_sha256(job.directory / "contract.json") ||
                                index.at("sha256") != file_sha256(job.directory / "actions.bin") ||
                                index.at("episode_sha256") !=
                                    file_sha256(job.directory / "episode.bin"))
                                throw std::invalid_argument(
                                    "completed teacher recovery payload changed");
                            done.result = {true, load_actions(job.directory / "actions.bin"), index,
                                           load_compact_actions(job.directory / "actions.bin")};
                            recovered = true;
                        }
                    }
                    if (!recovered) {
                        if (fs::exists(job.directory)) {
                            auto evidence = job.directory;
                            evidence +=
                                ".unpublished-" +
                                std::to_string(
                                    std::chrono::steady_clock::now().time_since_epoch().count());
                            fs::rename(job.directory, evidence);
                        }
                        done.result = prepare_placements(job.asset, job.directory, job.request,
                                                         options_, &policy);
                    }
                    done.recovered = recovered;
                    gpu::check(cudaEventRecord(end, stream.value));
                    gpu::check(cudaEventSynchronize(end));
                    float ms;
                    gpu::check(cudaEventElapsedTime(&ms, begin, end));
                    done.device_seconds = ms / 1000.;
                } catch (...) {
                    done.error = std::current_exception();
                }
                if (begin)
                    cudaEventDestroy(begin);
                if (end)
                    cudaEventDestroy(end);
                done.seconds =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                done.completed_ns = timestamp();
                {
                    std::lock_guard lock(mutex_);
                    ready_.emplace(job.id, std::move(done));
                }
                changed_.notify_all();
            }
        } catch (...) {
            std::lock_guard lock(mutex_);
            startup_error_ = std::current_exception();
            changed_.notify_all();
        }
    }

  public:
    TeacherWorkers(uint32_t count, NeuralOptions options, MemoryBudget& budget, uint32_t width,
                   uint32_t architecture = placement_schema)
        : options_(options), budget_(budget), width_(width), architecture_(architecture) {
        if (count < 1 || count > 3)
            throw std::invalid_argument("teacher worker count outside 1..3");
        try {
            for (uint32_t i = 0; i < count; ++i)
                threads_.emplace_back([this, i] { worker(uint8_t(i)); });
        } catch (...) {
            shutdown();
            throw;
        }
    }
    ~TeacherWorkers() {
        shutdown();
    }
    void wait_ready() {
        std::unique_lock lock(mutex_);
        changed_.wait(lock, [&] { return startup_error_ || ready_workers_ == threads_.size(); });
        if (startup_error_)
            std::rethrow_exception(startup_error_);
    }
    void shutdown() {
        cancelled_ = true;
        {
            std::lock_guard lock(mutex_);
            stop_ = true;
            queue_.clear();
        }
        changed_.notify_all();
        for (auto& t : threads_)
            if (t.joinable())
                t.join();
    }
    void wave(const float* weights, size_t count, cudaEvent_t ready) {
        std::lock_guard lock(mutex_);
        if (in_flight_)
            throw std::logic_error("policy changed while teacher jobs still borrow it");
        weights_ = weights;
        weight_count_ = count;
        policy_ready_ = ready;
        ++wave_;
    }
    void submit(TeacherJob job) {
        std::lock_guard lock(mutex_);
        if (startup_error_)
            std::rethrow_exception(startup_error_);
        if (!wave_ || in_flight_ >= threads_.size() * 2)
            throw std::length_error("teacher queue capacity");
        ++in_flight_;
        job.submitted_ns = timestamp();
        queue_.push_back(std::move(job));
        changed_.notify_one();
    }
    TeacherResult take(uint32_t id) {
        std::unique_lock lock(mutex_);
        changed_.wait(lock, [&] { return startup_error_ || ready_.contains(id); });
        if (startup_error_)
            std::rethrow_exception(startup_error_);
        auto node = ready_.extract(id);
        node.mapped().consumed_ns = timestamp();
        --in_flight_;
        lock.unlock();
        changed_.notify_all();
        if (node.mapped().error)
            std::rethrow_exception(node.mapped().error);
        return std::move(node.mapped());
    }
};
} // namespace blitz::neural::training
