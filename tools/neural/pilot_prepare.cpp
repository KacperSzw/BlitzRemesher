#include "training/placement_teacher.hpp"
#include <csignal>
using namespace blitz;
using namespace blitz::neural;
using namespace blitz::neural::training;
static volatile std::sig_atomic_t interrupted = 0;
int main(int argc, char** argv) {
    try {
        if (argc != 3 && argc != 5)
            throw std::invalid_argument("blitz-neural-pilot-prepare FRESH_DIRECTORY GPU_MEMORY_MIB "
                                        "[--training-profile coverage|attributes]");
        fs::path root = argv[1];
        if (fs::exists(root))
            throw std::invalid_argument("choose a fresh preparation directory");
        fs::create_directories(root);
        std::signal(SIGINT, [](int) { interrupted = 1; });
        std::signal(SIGTERM, [](int) { interrupted = 1; });
        auto start = std::chrono::steady_clock::now();
        auto elapsed = [&] {
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        };
        auto cancelled = [&] { return interrupted || elapsed() >= 600; };
        NeuralOptions options;
        options.raster_backend = NeuralRasterBackend::Vulkan;
        options.memory_mib = neural_unsigned(argv[2]);
        options.candidate_batch = 2;
        options.view_batch = 1;
        AuditSession session(options);
        const fs::path selection = "research/neural/prepared-pilot/selection.json",
                       curriculum = "research/neural/prepared-pilot/curriculum.json";
        auto conditions = read_json(curriculum).at("conditions");
        json report = {{"complete", false},
                       {"score", nullptr},
                       {"optimizer_updates", 0},
                       {"selection_sha256", file_sha256(selection)},
                       {"curriculum_sha256", file_sha256(curriculum)},
                       {"binary_sha256", file_sha256("/proc/self/exe")},
                       {"conditions", json::array()}};
        for (size_t i = 0; i < conditions.size() && !cancelled(); ++i) {
            auto c = conditions[i];
            PlacementRequest r;
            if (argc == 5) {
                if (std::string_view(argv[3]) != "--training-profile")
                    throw std::invalid_argument("unknown pilot option");
                r.profile = training_profile_option(argv[4]);
            }
            r.states = 2;
            r.selection = selection;
            r.mesh_cache = root / "mesh-cache";
            r.pixels = c.at("pixels");
            r.previous_steps = c.at("previous_steps");
            r.minutes = std::min(.5, (600 - elapsed()) / 60);
            r.cancelled = cancelled;
            auto directory = root / ("condition-" + std::to_string(i));
            json row = {{"condition", c}, {"complete", false}};
            try {
                auto result = prepare_placements(c.at("asset"), directory, r, options);
                row["complete"] = result.complete;
                row["result"] = result.index;
            } catch (const std::exception& e) {
                row["error"] = e.what();
                row["status"] = "preparation_failure";
            }
            report["conditions"].push_back(row);
            report["seconds"] = elapsed();
            write_json(root / "report.json", report);
        }
        report["complete"] =
            report["conditions"].size() == conditions.size() &&
            std::all_of(report["conditions"].begin(), report["conditions"].end(),
                        [](const auto& r) { return r.at("complete").template get<bool>(); });
        report["seconds"] = elapsed();
        write_json(root / "report.json", report);
        return report["complete"].get<bool>() ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
