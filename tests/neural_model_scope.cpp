#include "neural/internal.hpp"
#include "neural/placement.hpp"
#include "training/policy_ranking.hpp"
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
using namespace blitz;
using namespace blitz::neural;
namespace {
void require(bool value, const char* reason) {
    if (!value)
        throw std::runtime_error(reason);
}
template <class F> void rejects(F action) {
    bool rejected = false;
    try {
        action();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "invalid model scope accepted");
}
struct Temporary {
    std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("blitz-model-scope-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    Temporary() {
        std::filesystem::create_directory(path);
    }
    ~Temporary() {
        std::filesystem::remove_all(path);
    }
};
} // namespace
int main() try {
    Temporary directory;
    const auto a = directory.path / "a.blzn", b = directory.path / "b.blzn";
    for (uint32_t width : {64u, 128u, 256u}) {
        WeightsData weights;
        weights.architecture = conditioned_placement_schema;
        weights.hidden_width = width;
        weights.values.resize(policy_weights(weights.architecture, width), .013f);
        weights.provenance = width == 256 ? std::string(65536, 'x') : "scope fixture";
        save_weights(a, weights);
        auto legacy = load_weights(a);
        require(legacy.use == ModelUse::Unrestricted && legacy.hidden_width == width &&
                    legacy.values == weights.values,
                "legacy model meaning changed");
        save_weights(b, legacy);
        require(file_sha256(a) == file_sha256(b), "canonical legacy bytes changed");
        const auto legacy_size = std::filesystem::file_size(a);
        weights.use = ModelUse::EndpointReuseOnly;
        save_weights(a, weights);
        auto endpoint = load_weights(a);
        require(endpoint.use == ModelUse::EndpointReuseOnly && endpoint.values == weights.values &&
                    endpoint.hidden_width == width,
                "endpoint model scope did not roundtrip");
        require(std::filesystem::file_size(a) == legacy_size + (width == 64 ? 8 : 4),
                "scoped header layout differs");
        save_weights(b, endpoint);
        require(file_sha256(a) == file_sha256(b), "endpoint model bytes changed on roundtrip");
        validate_model_output(endpoint, OutputMode::Reuse);
        rejects([&] { validate_model_output(endpoint, OutputMode::Rebuild); });
        rejects([&] { validate_model_output(endpoint, {}); });
        rejects([&] { training::blend_ranking(legacy, endpoint, .5); });
        weights.use = ModelUse(2);
        rejects([&] { save_weights(a, weights); });
        weights.use = ModelUse::EndpointReuseOnly;
        weights.architecture = placement_schema;
        weights.values.resize(policy_weights(weights.architecture, width));
        rejects([&] { save_weights(a, weights); });
    }
    // Unknown serialized scope must fail even with a correct payload checksum.
    std::ifstream input(b, std::ios::binary | std::ios::ate);
    std::vector<std::byte> bytes(size_t(input.tellg()));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(bytes.data()), std::streamsize(bytes.size()));
    bytes[24] = std::byte{2};
    bytes[25] = bytes[26] = bytes[27] = std::byte{0};
    const auto sum = sha256(std::span(bytes).first(bytes.size() - 64));
    std::memcpy(bytes.data() + bytes.size() - 64, sum.data(), 64);
    std::ofstream output(a, std::ios::binary);
    output.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
    output.close();
    rejects([&] { load_weights(a); });
    std::cout << "model use scope contracts passed\n";
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
