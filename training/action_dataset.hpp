#pragma once
#include "training/action_data.hpp"
#include "training/policy_ranking.hpp"
#include "training/trainable_scope.hpp"
namespace blitz::neural::training {
struct ActionDataset {
    std::vector<float> x, targets;
    std::vector<uint8_t> labels;
    std::vector<std::array<std::vector<uint32_t>, 4>> asset_bins;
    std::vector<std::vector<uint32_t>> categories;
    json provenance = json::array();
    std::string rank_target;
    uint32_t states{}, architecture{}, observed_states{}, uninformative_states{};
};
inline ActionDataset load_action_dataset(const fs::path& directory, TrainableScope scope,
                                         const std::string& policy_hash) {
    if (scope > TrainableScope::EndpointScorer)
        throw std::invalid_argument("invalid action dataset training scope");
    const bool ranking_only = scope != TrainableScope::Joint;
    ActionDataset out;
    auto index = read_json(directory / "index.json");
    std::vector<fs::path> paths{directory};
    if (index.contains("datasets")) {
        paths.clear();
        for (auto& p : index.at("datasets")) {
            fs::path relative = p.get<std::string>();
            if (relative.is_absolute() || relative.string().find("..") != std::string::npos)
                throw std::invalid_argument("invalid dataset path");
            paths.push_back(directory / relative);
        }
    }
    std::vector<std::string> names, asset_names, asset_categories;
    for (auto& path : paths) {
        auto j = read_json(path / "index.json");
        if (!j.at("complete").get<bool>() ||
            (j.at("schema") != action_schema &&
             !is_placement_schema(j.at("schema").get<uint32_t>())) ||
            j.at("path") != "actions.bin" || j.at("sha256") != file_sha256(path / "actions.bin") ||
            j.at("contract_sha256") != file_sha256(path / "contract.json"))
            throw std::invalid_argument("incomplete or changed action dataset");
        auto data = load_actions(path / "actions.bin");
        const auto contract = read_json(path / "contract.json");
        const auto target = contract.value("teacher_target", "oracle");
        const bool runtime_endpoint = target == "runtime-endpoint-v1" ||
                                      target == "runtime-endpoint-v2" ||
                                      target == "runtime-endpoint-v3";
        if (ranking_only != (target == "policy-placement-v1" || runtime_endpoint))
            throw std::invalid_argument("teacher labels do not match the training objective");
        if (scope == TrainableScope::EndpointScorer && target != "runtime-endpoint-v3")
            throw std::invalid_argument("endpoint scorer requires runtime-endpoint-v3 teaching");
        if (scope == TrainableScope::EndpointScorer &&
            (!j.at("verification").at("same_meshes_and_errors").get<bool>() ||
             !j.at("verification").at("same_action_counts").get<bool>() ||
             !j.at("verification").at("same_trajectory_iterations").get<bool>() ||
             j.at("verification").at("unknown_observation").get<bool>() ||
             j.at("requests_sha256") != file_sha256(path / "requests.json") ||
             j.at("trajectory_sha256") != file_sha256(path / "trajectory.json") ||
             j.at("source_augmentation").at("reference_sha256") !=
                 file_sha256(path / "source-reference.bin")))
            throw std::invalid_argument("endpoint teaching trajectory is not verified");
        if (ranking_only) {
            if (!out.rank_target.empty() && out.rank_target != target)
                throw std::invalid_argument("v4 ranking cannot mix endpoint and placement targets");
            out.rank_target = target;
            if (contract.at("policy_payload_sha256") != policy_hash ||
                contract.at("teacher_version") != (target == "runtime-endpoint-v3"   ? 10
                                                   : target == "runtime-endpoint-v2" ? 9
                                                   : runtime_endpoint                ? 8
                                                                                     : 7) ||
                contract.at("data_storage") != "fp32" ||
                data.architecture != conditioned_placement_schema ||
                j.at("geometry_rejected_sha256") != file_sha256(path / "geometry-rejected.bin"))
                throw std::invalid_argument("frozen placement policy/data identity differs");
            std::ifstream f(path / "geometry-rejected.bin", std::ios::binary);
            char magic[8];
            f.read(magic, 8);
            if (!f || std::memcmp(magic, "BLZRANK1", 8))
                throw std::invalid_argument("invalid geometry rejection bitmap");
            auto bits = read_vector<uint8_t>(f, (data.labels.size() + 7) / 8);
            if (f.peek() != EOF)
                throw std::invalid_argument("geometry rejection bitmap tail");
            apply_policy_rank_masks(data.labels, bits);
        }
        if (j.at("schema") != data.architecture ||
            (out.architecture && out.architecture != data.architecture))
            throw std::invalid_argument("mixed action policy datasets");
        out.architecture = data.architecture;
        auto width = policy_inputs(data.architecture);
        if (!data.states() && !runtime_endpoint)
            throw std::invalid_argument("action dataset has no states");
        out.provenance.push_back({{"asset", j.at("asset")},
                                  {"index_sha256", file_sha256(path / "index.json")},
                                  {"sha256", j.at("sha256")},
                                  {"contract_sha256", j.at("contract_sha256")}});
        if (ranking_only)
            out.provenance.back()["geometry_rejected_sha256"] = j.at("geometry_rejected_sha256");
        std::vector<uint32_t> eligible;
        for (uint32_t s = 0; s < data.states(); ++s) {
            ++out.observed_states;
            if (ranking_only && !ranking_pairs(std::span(data.labels)
                                                   .subspan(data.offsets[s],
                                                            data.offsets[s + 1] - data.offsets[s])))
                ++out.uninformative_states;
            else
                eligible.push_back(s);
        }
        if (eligible.empty())
            continue;
        if (uint64_t(out.states) + data.states() >
            (8ull << 30) / (action_pool * width * sizeof(float)))
            throw std::length_error("resident action dataset exceeds 8 GiB");
        const auto category = j.at("category").get<std::string>();
        auto found = std::find(names.begin(), names.end(), category);
        if (found == names.end()) {
            names.push_back(category);
            out.categories.emplace_back();
            found = std::prev(names.end());
        }
        auto asset = j.at("asset").get<std::string>();
        auto asset_it = std::find(asset_names.begin(), asset_names.end(), asset);
        size_t asset_id = size_t(asset_it - asset_names.begin());
        if (asset_it == asset_names.end()) {
            asset_names.push_back(asset);
            asset_categories.push_back(category);
            out.categories[size_t(found - names.begin())].push_back(uint32_t(asset_id));
            out.asset_bins.emplace_back();
        } else if (asset_categories[asset_id] != category)
            throw std::invalid_argument("asset appears in multiple categories");
        for (uint32_t s : eligible) {
            auto count = data.offsets[s + 1] - data.offsets[s];
            out.asset_bins[asset_id][std::min(3u, uint32_t(data.progress[s] * 4))].push_back(
                out.states++);
            out.x.insert(out.x.end(), data.x.begin() + size_t(data.offsets[s]) * width,
                         data.x.begin() + size_t(data.offsets[s + 1]) * width);
            out.x.resize(size_t(out.states) * action_pool * width);
            out.labels.insert(out.labels.end(), data.labels.begin() + data.offsets[s],
                              data.labels.begin() + data.offsets[s + 1]);
            out.labels.resize(size_t(out.states) * action_pool);
            if (is_placement_schema(data.architecture)) {
                out.targets.insert(out.targets.end(),
                                   data.targets.begin() + size_t(data.offsets[s]) * 9,
                                   data.targets.begin() + size_t(data.offsets[s + 1]) * 9);
                out.targets.resize(size_t(out.states) * action_pool * 9);
            }
        }
    }
    if (!out.states)
        throw std::invalid_argument("action dataset collection is empty");
    if (out.x.size() * sizeof(float) > 8ull * 1024 * 1024 * 1024)
        throw std::length_error("resident action dataset exceeds 8 GiB");
    return out;
}
} // namespace blitz::neural::training
