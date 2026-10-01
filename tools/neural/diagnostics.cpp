#include "tools/neural/cycle_replay.hpp"
#include "tools/neural/model_audit.hpp"
#include "tools/neural/numeric_diagnosis.hpp"
#include "tools/neural/resident_tests.hpp"
#include "tools/neural/storage_proof.hpp"
#include "tools/neural/update_benchmark.hpp"
#include <iostream>

using namespace blitz;
using namespace blitz::neural;
using namespace blitz::neural::training;
int main(int argc, char** argv) {
    try {
        ieee_fp32();
        torch::set_num_threads(1);
        if (argc == 4 && std::string_view(argv[1]) == "--diagnose-checkpoint") {
            diagnose_checkpoint(argv[2], argv[3]);
            return 0;
        }
        if (!torch::cuda::is_available())
            return 77;
        if (argc == 6 && std::string_view(argv[1]) == "--audit-model") {
            audit_model(argv[2], argv[3], argv[4], argv[5]);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--check") {
            resident_contracts();
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--check-checkpoint") {
            checkpoint_forensics_contract();
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--check-schema") {
            c10::cuda::CUDAStreamGuard learner(c10::cuda::getStreamFromPool());
            gpu::StreamScope native_stream(learner.current_stream());
            placement_label_contract();
            resident_contracts(false, UpdateBackend::Reference, 64, conditioned_placement_schema);
            resident_contracts(true, UpdateBackend::Fused, 64, conditioned_placement_schema);
            return 0;
        }
        if (argc == 6 && std::string_view(argv[1]) == "--migrate-policy") {
            if (fs::exists(argv[4]))
                throw std::invalid_argument("choose a fresh migrated policy output");
            auto original = load_weights(argv[2]);
            torch::Tensor probe;
            torch::load(probe, argv[3]);
            probe = probe.cpu().to(torch::kFloat32).contiguous();
            json verification;
            try {
                auto migrated = condition_policy(
                    original, {probe.data_ptr<float>(), size_t(probe.numel())}, verification);
                auto conditioned = probe.reshape({-1, placement_features}).clone();
                conditioned.select(1, 79).fill_(1);
                ActionCuda native(migrated, {}, 32);
                auto values =
                    native.predict({conditioned.data_ptr<float>(), size_t(conditioned.numel())});
                auto difference = numeric_difference(
                    std::vector<double>(values.begin(), values.end()),
                    action_oracle({probe.data_ptr<float>(), size_t(probe.numel())}, original));
                verification["native"] = difference_json(difference);
                verification["passed"] = legacy_numeric_pass(difference);
                if (!legacy_numeric_pass(difference))
                    throw std::runtime_error("native migration failed unchanged numerical gate");
                verification["source_sha256"] = file_sha256(argv[2]);
                verification["probe_sha256"] = file_sha256(argv[3]);
                save_weights(argv[4], migrated);
                write_json(argv[5], verification);
                return 0;
            } catch (...) {
                verification["passed"] = false;
                write_json(argv[5], verification);
                throw;
            }
        }
        if (argc == 3 && std::string_view(argv[1]) == "--check-width") {
            gpu::StreamScope native_stream;
            c10::cuda::CUDAStreamGuard learner(
                c10::cuda::getStreamFromExternal(native_stream.value, 0));
            resident_contracts(true, UpdateBackend::Fused, neural_unsigned(argv[2]));
            return 0;
        }
        if (argc == 5 && std::string_view(argv[1]) == "--compare-storage") {
            storage_proof(argv[2], argv[3], argv[4]);
            return 0;
        }
        if (argc == 5 && std::string_view(argv[1]) == "--benchmark-update") {
            benchmark_updates(argv[2], argv[3], argv[4]);
            return 0;
        }
        if (argc == 5 && std::string_view(argv[1]) == "--replay-checkpoint")
            return replay_cycle_checkpoint(argv[2], argv[3], neural_unsigned(argv[4]));
        throw std::invalid_argument(
            "blitz-neural-diagnostics: --check | --check-checkpoint | --check-schema | "
            "--check-width N | --diagnose-checkpoint CHECKPOINT OUTPUT | --audit-model MANIFEST "
            "MODEL SETTINGS OUTPUT | --migrate-policy MODEL PROBE OUTPUT REPORT | "
            "--compare-storage SHARDS OUTPUT MODEL | --benchmark-update SHARDS OUTPUT MODEL | "
            "--replay-checkpoint RUN OUTPUT UPDATES");
    } catch (const std::exception& e) {
        std::cerr << "neural diagnostics: " << e.what() << '\n';
        return 1;
    }
}
