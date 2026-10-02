#include "training/action_dataset.hpp"
#include "training/mesh_cache.hpp"
#include <chrono>
#include <iostream>
using namespace blitz;
using namespace blitz::neural;
using namespace blitz::neural::training;
namespace {
void require(bool value, const char* reason) {
    if (!value)
        throw std::runtime_error(reason);
}
template <class F> void rejects(F call) {
    bool rejected = false;
    try {
        call();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "unverified endpoint teaching was admitted");
}
struct Temporary {
    fs::path path = fs::temp_directory_path() /
                    ("blitz-action-dataset-" +
                     std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    Temporary() {
        fs::create_directory(path);
    }
    ~Temporary() {
        fs::remove_all(path);
    }
};
} // namespace
int main() try {
    Temporary directory;
    const std::string collector(64, 'a');
    for (uint32_t count : {3u, 9u})
        for (bool preserve : {false, true}) {
            ActionData data;
            data.architecture = conditioned_placement_schema;
            data.offsets = {0, count, 2 * count, 3 * count};
            data.progress = {0, .5, 1};
            data.x.resize(3 * count * placement_features);
            data.targets.resize(3 * count * 9);
            data.labels.resize(3 * count, 24);
            data.from.resize(3 * count);
            data.to.resize(3 * count, 1);
            for (size_t row = 0; row < 3 * count; ++row)
                data.x[row * placement_features + 79] = float(preserve);
            data.labels[0] = data.labels[count] = 31;
            std::fill(data.labels.begin() + 2 * count, data.labels.end(),
                      31); // No informative pair in the third state.
            save_actions(directory.path / "actions.bin", data, false);
            {
                std::ofstream stream(directory.path / "geometry-rejected.bin", std::ios::binary);
                stream.write("BLZRANK1", 8);
                write_vector(stream, std::vector<uint8_t>((data.labels.size() + 7) / 8));
            }
            Mesh mesh;
            mesh.positions = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
            mesh.indices = {0, 1, 2};
            {
                std::ofstream stream(directory.path / "source-reference.bin", std::ios::binary);
                write_reference(stream, mesh);
            }
            write_json(directory.path / "requests.json", json::array());
            write_json(directory.path / "trajectory.json", json::array());
            json contract = {{"teacher_target", "runtime-endpoint-v3"},
                             {"teacher_version", 10},
                             {"data_storage", "fp32"},
                             {"policy_payload_sha256", collector}};
            write_json(directory.path / "contract.json", contract);
            json index = {
                {"schema", data.architecture},
                {"complete", true},
                {"path", "actions.bin"},
                {"asset", "fixture"},
                {"category", "rocks"},
                {"sha256", file_sha256(directory.path / "actions.bin")},
                {"contract_sha256", file_sha256(directory.path / "contract.json")},
                {"geometry_rejected_sha256", file_sha256(directory.path / "geometry-rejected.bin")},
                {"requests_sha256", file_sha256(directory.path / "requests.json")},
                {"trajectory_sha256", file_sha256(directory.path / "trajectory.json")},
                {"source_augmentation",
                 {{"reference_sha256", file_sha256(directory.path / "source-reference.bin")}}},
                {"verification",
                 {{"same_meshes_and_errors", true},
                  {"same_action_counts", true},
                  {"same_trajectory_iterations", true},
                  {"unknown_observation", false}}}};
            const auto reset = [&] {
                write_json(directory.path / "contract.json", contract);
                write_json(directory.path / "index.json", index);
            };
            reset();
            auto actual =
                load_action_dataset(directory.path, TrainableScope::EndpointScorer, collector);
            require(actual.states == 2 && actual.observed_states == 3 &&
                        actual.uninformative_states == 1,
                    "dataset changed informative-state admission");
            require(actual.x.size() == 2 * action_pool * placement_features &&
                        actual.labels.size() == 2 * action_pool,
                    "pool padding differs");
            require(actual.asset_bins.size() == 1 && actual.categories.size() == 1,
                    "asset-balanced sampling hierarchy differs");
            require(actual.provenance.at(0).at("index_sha256") ==
                        file_sha256(directory.path / "index.json"),
                    "sampler metadata is not bound to training provenance");
            auto changed_category = index;
            changed_category["category"] = "organic";
            write_json(directory.path / "index.json", changed_category);
            require(load_action_dataset(directory.path, TrainableScope::EndpointScorer, collector)
                            .provenance != actual.provenance,
                    "changed sampler categories retain the previous training contract");
            reset();
            rejects([&] {
                load_action_dataset(directory.path, TrainableScope::EndpointScorer,
                                    std::string(64, 'b'));
            });
            rejects([&] { load_action_dataset(directory.path, TrainableScope::Joint, collector); });
            for (const char* field : {"same_meshes_and_errors", "same_action_counts",
                                      "same_trajectory_iterations", "unknown_observation"}) {
                auto bad = index;
                bad["verification"][field] = std::string_view(field) == "unknown_observation";
                write_json(directory.path / "index.json", bad);
                rejects([&] {
                    load_action_dataset(directory.path, TrainableScope::EndpointScorer, collector);
                });
                reset();
            }
            for (const char* field :
                 {"requests_sha256", "trajectory_sha256", "geometry_rejected_sha256"}) {
                auto bad = index;
                bad[field] = std::string(64, '0');
                write_json(directory.path / "index.json", bad);
                rejects([&] {
                    load_action_dataset(directory.path, TrainableScope::EndpointScorer, collector);
                });
                reset();
            }
            for (auto [target, version] :
                 {std::pair{"runtime-endpoint-v1", 8}, std::pair{"runtime-endpoint-v2", 9},
                  std::pair{"policy-placement-v1", 7}}) {
                auto legacy = contract;
                legacy["teacher_target"] = target;
                legacy["teacher_version"] = version;
                write_json(directory.path / "contract.json", legacy);
                auto legacy_index = index;
                legacy_index["contract_sha256"] = file_sha256(directory.path / "contract.json");
                write_json(directory.path / "index.json", legacy_index);
                require(load_action_dataset(directory.path, TrainableScope::RankingRow, collector)
                                .states == 2,
                        "legacy row-scoring data compatibility changed");
                rejects([&] {
                    load_action_dataset(directory.path, TrainableScope::EndpointScorer, collector);
                });
                reset();
            }
        }
    std::cout << "action dataset scope and teaching integrity contracts passed\n";
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
