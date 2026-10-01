#include "blitz/io.hpp"
#include "tools/foliage.hpp"
#include <archive.h>
#include <archive_entry.h>
#include <chrono>
#include <curl/curl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <regex>
#include <set>
#include <sstream>
#include <thread>
using json = nlohmann::json;
namespace fs = std::filesystem;
#include "tools/acquisition/acquisition.hpp"
#include "tools/acquisition/neural_corpus.hpp"
using namespace blitz::acquisition;
int main(int argc, char** argv) try {
    if (argc > 1 && std::string_view(argv[1]) == "neural-expand")
        return expand_neural_corpus(argc - 1, argv + 1);
    if (argc > 1 && std::string_view(argv[1]) == "neural-cache")
        return prepare_neural_corpus_cache(argc - 1, argv + 1);
    if (argc > 1 && std::string_view(argv[1]).starts_with("foliage-"))
        return foliage::main(argc - 1, argv + 1);
    fs::path root = argc > 1 ? argv[1] : "data";
    const auto start = std::chrono::steady_clock::now();
    curl_global_init(CURL_GLOBAL_DEFAULT);
    fs::create_directories(root);
    auto mp = root / "manifest.json";
    if (argc > 2) {
        auto frozen = json::parse(read(argv[2]));
        size_t complete = 0;
        uint64_t total = 0;
        for (auto& asset : frozen["assets"]) {
            for (auto& file : asset["files"]) {
                total += file.at("bytes").get<uint64_t>();
                if (total > 50ull * 1024 * 1024 * 1024)
                    throw std::runtime_error("frozen corpus exceeds size budget");
                fs::path original = file.at("path").get<std::string>();
                if (original.is_absolute())
                    throw std::runtime_error("absolute corpus path");
                fs::path relative;
                bool first = true;
                for (auto part : original) {
                    if (part == "..")
                        throw std::runtime_error("unsafe corpus path");
                    if (first) {
                        first = false;
                        continue;
                    }
                    relative /= part;
                }
                if (relative.empty())
                    throw std::runtime_error("empty corpus path");
                auto destination = root / relative;
                auto expected = file.at("sha256").get<std::string>();
                if (fs::exists(destination) && hash(read(destination)) == expected)
                    continue;
                if (std::chrono::steady_clock::now() - start > std::chrono::minutes(50)) {
                    std::cout << "Replay checkpoint; rerun the same command\n";
                    return 2;
                }
                std::string bytes;
                if (file.contains("url"))
                    bytes = request(file.at("url"));
                else if (asset.at("provider") == "smithsonian")
                    bytes = extract_obj(request(asset.at("source_url")));
                else
                    throw std::runtime_error("missing frozen file URL");
                if (hash(bytes) != expected)
                    throw std::runtime_error(
                        "upstream bytes changed; refusing nonreproducible corpus");
                write(destination, bytes);
            }
            ++complete;
            std::cout << "Verified " << complete << "/" << frozen["assets"].size() << std::endl;
        }
        std::cout << "Frozen corpus replay complete\n";
        curl_global_cleanup();
        return 0;
    }
    json manifest = fs::exists(mp) ? json::parse(read(mp))
                                   : json{{"version", 1},
                                          {"seed", 0xB1172026u},
                                          {"assets", json::array()},
                                          {"rejected", json::array()}};
    std::set<std::string> seen;
    std::map<std::string, size_t> counts;
    uint64_t acquired_bytes = 0;
    for (auto& a : manifest["assets"]) {
        seen.insert(a["id"]);
        counts[a["category"].get<std::string>()]++;
    }
    for (auto& a : manifest["assets"])
        for (auto& f : a["files"])
            acquired_bytes += f.value("bytes", uint64_t(0));
    auto checkpoint = [&] { write(mp, manifest.dump(2)); };
    auto add = [&](json a) {
        for (auto& f : a["files"])
            acquired_bytes += f.value("bytes", uint64_t(0));
        std::string cat = a["category"];
        counts[cat]++;
        seen.insert(a["id"]);
        manifest["assets"].push_back(std::move(a));
        checkpoint();
        std::cout << manifest["assets"].size() << " downloaded, " << cat << " " << counts[cat]
                  << std::endl;
    };
    auto expired = [&] {
        return std::chrono::steady_clock::now() - start > std::chrono::minutes(50) ||
               acquired_bytes + max_file > 50ull * 1024 * 1024 * 1024;
    };
    auto assets =
        cached("https://api.polyhaven.com/assets?t=models", root / "catalog-polyhaven.json");
    std::map<std::string, std::vector<std::string>> candidates;
    for (auto it = assets.begin(); it != assets.end(); ++it) {
        auto cats = it.value()["categories"];
        auto id = it.key();
        if (id == "long_life_food" || id == "russian_food_cans_01" ||
            id.starts_with("coastal_cliff"))
            continue; // QC: packaging and cliffs are not organic exemplars.
        if (contains(cats, "rocks"))
            candidates["rocks"].push_back(id);
        else if (contains(cats, "nature") || contains(cats, "food"))
            candidates["organic"].push_back(id);
        else
            candidates["manufactured"].push_back(id);
    }
    for (auto& [cat, ids] : candidates)
        std::stable_sort(ids.begin(), ids.end(), [&](auto& a, auto& b) {
            if (cat == "organic" && woody(a) != woody(b))
                return woody(a);
            return hash("B1172026" + a) < hash("B1172026" + b);
        });
    for (auto [cat, quota] :
         std::map<std::string, size_t>{{"rocks", 30}, {"organic", 30}, {"manufactured", 40}})
        for (auto& id : candidates[cat]) {
            if (counts[cat] >= quota || expired())
                break;
            if (seen.contains("ph_" + id))
                continue;
            try {
                add(poly(id, cat, root));
            } catch (const std::exception& e) {
                manifest["rejected"].push_back({{"id", "ph_" + id}, {"reason", e.what()}});
                checkpoint();
                std::cerr << id << ": " << e.what() << '\n';
            }
        }
    for (unsigned page = 0; counts["stress"] < 20 && page < 100 && !expired(); ++page) {
        auto url =
            "https://3d-api.si.edu/api/v1.0/content/file/search?model_type=obj&rows=100&start=" +
            std::to_string(page * 100);
        auto list = cached(url, root / ("catalog-si-" + std::to_string(page) + ".json"));
        auto rows = list.value("rows", json::array());
        if (rows.empty())
            break;
        for (auto& row : rows) {
            auto& c = row["content"];
            auto id = "si_" + safe(c["model_url"]);
            if (seen.contains(id) || counts["stress"] >= 20 || expired())
                continue;
            if (c.value("quality", "") != "Low_resolution")
                continue;
            try {
                add(smithsonian(row, root));
            } catch (const std::exception& e) {
                manifest["rejected"].push_back({{"id", id}, {"reason", e.what()}});
                checkpoint();
                std::cerr << id << ": " << e.what() << '\n';
            }
        }
    }
    // Deterministic category-stratified assignment; frozen only once all quotas exist.
    bool complete = counts["rocks"] == 30 && counts["organic"] == 30 &&
                    counts["manufactured"] == 40 && counts["stress"] == 20;
    size_t woody_count = 0;
    for (auto& a : manifest["assets"])
        woody_count += a.value("woody", false);
    complete = complete && woody_count >= 10;
    if (complete) {
        for (auto cat : {"rocks", "organic", "manufactured", "stress"}) {
            std::vector<size_t> order;
            for (size_t i = 0; i < manifest["assets"].size(); ++i)
                if (manifest["assets"][i]["category"] == cat)
                    order.push_back(i);
            std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                return hash("split" + manifest["assets"][a]["source_group"].get<std::string>()) <
                       hash("split" + manifest["assets"][b]["source_group"].get<std::string>());
            });
            size_t train = std::string(cat) == "manufactured" ? 26
                           : std::string(cat) == "stress"     ? 14
                                                              : 20;
            size_t val = std::string(cat) == "manufactured" ? 7
                         : std::string(cat) == "stress"     ? 3
                                                            : 5;
            for (size_t j = 0; j < order.size(); ++j)
                manifest["assets"][order[j]]["split"] = j < train         ? "development"
                                                        : j < train + val ? "validation"
                                                                          : "held-out";
        }
    }
    manifest["acquisition_complete"] = complete;
    checkpoint();
    curl_global_cleanup();
    std::cout << "Acquisition " << (complete ? "complete" : "checkpointed") << ": "
              << manifest["assets"].size() << " assets\n";
    return complete ? 0 : 2;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
