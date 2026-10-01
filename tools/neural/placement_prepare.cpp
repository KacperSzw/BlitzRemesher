#include "neural/action_gpu.hpp"
#include "neural/memory.hpp"
#include "tools/neural/teacher_benchmark.hpp"
#include "training/json.hpp"
#include "training/placement_teacher.hpp"
#include <csignal>
#include <iostream>
using namespace blitz;
using namespace blitz::neural;
using namespace blitz::neural::training;
static volatile std::sig_atomic_t stopped = 0;
static void stop(int) {
    stopped = 1;
}
int main(int argc, char** argv) {
    try {
        if (argc > 1 && std::string_view(argv[1]) == "--jobs") {
            if (argc != 4)
                throw std::invalid_argument("blitz-neural-placement-prepare --jobs JSON OUTPUT");
            std::signal(SIGINT, stop);
            std::signal(SIGTERM, stop);
            return prepare_teacher_jobs(argv[2], argv[3], [] { return bool(stopped); });
        }
        if (argc < 3)
            throw std::invalid_argument(
                "blitz-neural-placement-prepare ASSET OUTPUT [--states N] [--pool N] [--pixels N] "
                "[--previous-steps N] [--source-limit N] [--adjacent-limit N] [--minutes N] "
                "[--gpu-memory-mib N] [--model MODEL] [--seed N] [--corpus JSON] "
                "[--training-selection JSON] [--mesh-cache DIR] [--episode FILE] "
                "[--simplifier-seed on|off] [--retained FRACTION] [--architecture 3|4] "
                "[--preserve-uv on|off] [--previous-pixels N] [--teacher-selection "
                "geometric-random|policy-mixed] [--policy-rollout-trials 0..4096] "
                "[--teacher-strategy exhaustive|coverage-core-first]");
        uint32_t states = 32, pool = 16, previous_steps = 0, seed = 101;
        double pixels = 128, minutes = 5, source_limit = 3, adjacent_limit = 3;
        NeuralOptions options;
        fs::path model;
        bool sparse = false;
        Profile profile = Profile::Attributes;
        PlacementRequest episode;
        auto integer = [](const char* value) {
            std::string s = value;
            if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos)
                throw std::invalid_argument("invalid integer option");
            auto x = std::stoull(s);
            if (x > 65536)
                throw std::invalid_argument("integer option too large");
            return uint32_t(x);
        };
        for (int i = 3; i < argc; i += 2) {
            if (i + 1 == argc)
                throw std::invalid_argument("missing preparation option");
            std::string k = argv[i];
            if (k == "--corpus") {
                episode.corpus = argv[i + 1];
                continue;
            }
            if (k == "--training-selection") {
                episode.selection = argv[i + 1];
                continue;
            }
            if (k == "--mesh-cache") {
                episode.mesh_cache = argv[i + 1];
                continue;
            }
            if (k == "--episode") {
                episode.episode = argv[i + 1];
                continue;
            }
            if (k == "--retained") {
                episode.retained = std::stod(argv[i + 1]);
                continue;
            }
            if (k == "--architecture") {
                episode.architecture = integer(argv[i + 1]);
                continue;
            }
            if (k == "--policy-rollout-trials") {
                episode.policy_rollout_trials = integer(argv[i + 1]);
                continue;
            }
            if (k == "--previous-pixels") {
                episode.previous_pixels = std::stod(argv[i + 1]);
                continue;
            }
            if (k == "--teacher-selection") {
                std::string_view v = argv[i + 1];
                if (v != "geometric-random" && v != "policy-mixed")
                    throw std::invalid_argument(
                        "teacher selection must be geometric-random or policy-mixed");
                episode.policy_candidates = v == "policy-mixed";
                continue;
            }
            if (k == "--teacher-strategy") {
                episode.strategy = teacher_strategy_option(argv[i + 1]);
                continue;
            }
            if (k == "--preserve-uv") {
                std::string_view v = argv[i + 1];
                if (v != "on" && v != "off")
                    throw std::invalid_argument("preserve UV must be on or off");
                episode.preserve_uv = v == "on";
                continue;
            }
            if (k == "--simplifier-seed") {
                std::string_view v = argv[i + 1];
                if (v != "on" && v != "off")
                    throw std::invalid_argument("simplifier seed must be on or off");
                episode.simplifier = v == "on";
                continue;
            }
            if (k == "--training-profile")
                profile = training_profile_option(argv[i + 1]);
            else if (k == "--mask-only-coverage") {
                std::string_view v = argv[i + 1];
                if (v != "on" && v != "off")
                    throw std::invalid_argument("mask-only coverage must be on or off");
                options.mask_only_coverage = v == "on";
            } else if (k == "--states")
                states = integer(argv[i + 1]);
            else if (k == "--seed")
                seed = integer(argv[i + 1]);
            else if (k == "--pool")
                pool = integer(argv[i + 1]);
            else if (k == "--previous-steps")
                previous_steps = integer(argv[i + 1]);
            else if (k == "--pixels")
                pixels = std::stod(argv[i + 1]);
            else if (k == "--source-limit")
                source_limit = std::stod(argv[i + 1]);
            else if (k == "--adjacent-limit")
                adjacent_limit = std::stod(argv[i + 1]);
            else if (k == "--raster-backend")
                options.raster_backend = raster_option(argv[i + 1]);
            else if (k == "--candidate-batch") {
                auto n = integer(argv[i + 1]);
                if (n != 1 && n != 2 && n != 4 && n != 8)
                    throw std::invalid_argument("candidate batch must be 1, 2, 4 or 8");
                options.candidate_batch = uint8_t(n);
            } else if (k == "--vertex-storage")
                options.vertex_storage = storage_option(argv[i + 1]);
            else if (k == "--audit-mode") {
                std::string mode = argv[i + 1];
                if (mode != "sparse" && mode != "exact")
                    throw std::invalid_argument("audit mode must be sparse or exact");
                sparse = mode == "sparse";
            } else if (k == "--model")
                model = argv[i + 1];
            else if (k == "--minutes")
                minutes = std::stod(argv[i + 1]);
            else if (k == "--gpu-memory-mib")
                options.memory_mib = integer(argv[i + 1]);
            else
                throw std::invalid_argument("unknown preparation option");
        }
        if (!states || states > 4096 || previous_steps > 4096 || !pool || pool > 16 ||
            !std::isfinite(pixels) || pixels < 16 || pixels > 512 || !std::isfinite(minutes) ||
            minutes <= 0 || minutes > 50 || !std::isfinite(source_limit) || source_limit <= 0 ||
            source_limit > 16 || !std::isfinite(adjacent_limit) || adjacent_limit <= 0 ||
            adjacent_limit > 16)
            throw std::invalid_argument("preparation bounds");
        std::signal(SIGINT, stop);
        std::signal(SIGTERM, stop);
        fs::path output = argv[2];
        std::string policy_hash;
        std::unique_ptr<ActionCuda> policy;
        if (!model.empty()) {
            auto w = load_weights(model);
            if (w.architecture != episode.architecture)
                throw std::invalid_argument(
                    "placement teacher policy architecture differs from requested schema");
            policy_hash = sha256(std::as_bytes(std::span(w.values)));
            policy = std::make_unique<ActionCuda>(w, options);
        }
        PlacementRequest request{states,
                                 pool,
                                 previous_steps,
                                 seed,
                                 pixels,
                                 minutes,
                                 source_limit,
                                 adjacent_limit,
                                 sparse,
                                 policy_hash,
                                 [] { return bool(stopped); }};
        request.architecture = episode.architecture;
        request.strategy = episode.strategy;
        request.policy_candidates = episode.policy_candidates;
        request.policy_rollout_trials = episode.policy_rollout_trials;
        request.previous_pixels = episode.previous_pixels;
        request.preserve_uv = episode.preserve_uv;
        request.profile = profile;
        request.corpus = episode.corpus;
        request.selection = episode.selection;
        request.mesh_cache = episode.mesh_cache;
        request.episode = episode.episode;
        request.simplifier = episode.simplifier;
        request.retained = episode.retained;
        return prepare_placements(argv[1], output, request, options, policy.get()).complete ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "placement preparation: " << e.what() << '\n';
        return 1;
    }
}
