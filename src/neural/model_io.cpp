#include "neural/internal.hpp"
#include "neural/placement.hpp"
#include <atomic>
#include <bit>
#include <chrono>
#include <fstream>
#include <openssl/evp.h>
namespace blitz::neural {
std::string sha256(std::span<const std::byte> bytes) {
    unsigned char hash[32];
    unsigned n = 0;
    if (!EVP_Digest(bytes.data(), bytes.size(), hash, &n, EVP_sha256(), nullptr) || n != 32)
        throw std::runtime_error("SHA-256 failed");
    std::string out;
    out.reserve(64);
    for (auto c : hash) {
        out.push_back("0123456789abcdef"[c >> 4]);
        out.push_back("0123456789abcdef"[c & 15]);
    }
    return out;
}
std::string file_sha256(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::invalid_argument("cannot open hash input: " + path.string());
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (!ctx || !EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr))
        throw std::runtime_error("SHA-256 initialization failed");
    char buffer[65536];
    while (f) {
        f.read(buffer, sizeof(buffer));
        if (!EVP_DigestUpdate(ctx.get(), buffer, size_t(f.gcount())))
            throw std::runtime_error("SHA-256 failed");
    }
    if (!f.eof())
        throw std::runtime_error("hash input read failed");
    unsigned char hash[32];
    unsigned n = 0;
    if (!EVP_DigestFinal_ex(ctx.get(), hash, &n) || n != 32)
        throw std::runtime_error("SHA-256 failed");
    std::string out;
    for (auto c : hash) {
        out.push_back("0123456789abcdef"[c >> 4]);
        out.push_back("0123456789abcdef"[c & 15]);
    }
    return out;
}
namespace {
void put_u32(std::vector<std::byte>& out, uint32_t x) {
    for (unsigned i = 0; i < 4; ++i)
        out.push_back(std::byte((x >> (i * 8)) & 255));
}
uint32_t get_u32(std::span<const std::byte> a, size_t at) {
    uint32_t x = 0;
    for (unsigned i = 0; i < 4; ++i)
        x |= uint32_t(a[at + i]) << (i * 8);
    return x;
}
constexpr char magic[8] = {'B', 'L', 'Z', 'N', 'E', 'T', '0', '1'};
} // namespace
WeightsData load_weights(const std::filesystem::path& file, std::string* hash) {
    std::ifstream f(file, std::ios::binary | std::ios::ate);
    if (!f)
        throw std::invalid_argument("cannot open neural model: " + file.string());
    auto size = f.tellg();
    if (size < 84 ||
        size > std::streamoff(92 + policy_weights(conditioned_placement_schema, 256) * 4 + 65536))
        throw std::invalid_argument("neural model size invalid");
    std::vector<std::byte> data(static_cast<size_t>(size));
    f.seekg(0);
    if (!f.read(reinterpret_cast<char*>(data.data()), size))
        throw std::invalid_argument("truncated neural model");
    const bool scoped = !std::memcmp(data.data(), "BLZNET03", 8);
    bool shaped = scoped || !std::memcmp(data.data(), "BLZNET02", 8);
    uint32_t width = shaped ? get_u32(data, 20) : 64;
    size_t header = scoped ? 28 : shaped ? 24 : 20;
    auto architecture = get_u32(data, 8);
    const auto use = scoped ? get_u32(data, 24) : uint32_t(ModelUse::Unrestricted);
    if (scoped && (use != uint32_t(ModelUse::EndpointReuseOnly) ||
                   architecture != conditioned_placement_schema))
        throw std::invalid_argument("incompatible neural model use scope");
    size_t count = architecture == schema ? (width == 64 ? weight_count : 0)
                                          : policy_weights(architecture, width);
    if ((!shaped && std::memcmp(data.data(), magic, 8)) || !count || get_u32(data, 12) != count)
        throw std::invalid_argument("incompatible neural model schema/architecture");
    size_t provenance = get_u32(data, 16), expected = header + provenance + count * 4 + 64;
    if (provenance > 65536 || data.size() != expected)
        throw std::invalid_argument("invalid neural model payload");
    auto payload = std::span(data).first(data.size() - 64);
    auto checksum = sha256(payload);
    if (std::memcmp(checksum.data(), data.data() + data.size() - 64, 64))
        throw std::invalid_argument("neural model checksum mismatch");
    WeightsData w;
    w.architecture = architecture;
    w.hidden_width = width;
    w.use = ModelUse(use);
    w.provenance.assign(reinterpret_cast<const char*>(data.data() + header), provenance);
    w.values.resize(count);
    for (size_t i = 0; i < count; ++i) {
        w.values[i] = std::bit_cast<float>(get_u32(data, header + provenance + i * 4));
        if (!std::isfinite(w.values[i]))
            throw std::invalid_argument("nonfinite model weights");
    }
    if (hash)
        *hash = sha256(data);
    return w;
}
void save_weights(const std::filesystem::path& file, const WeightsData& w) {
    size_t count = w.architecture == schema ? (w.hidden_width == 64 ? weight_count : 0)
                                            : policy_weights(w.architecture, w.hidden_width);
    if (!count || w.values.size() != count || w.provenance.size() > 65536)
        throw std::invalid_argument("invalid model weight dimensions or provenance");
    if (w.use > ModelUse::EndpointReuseOnly ||
        (w.use == ModelUse::EndpointReuseOnly && w.architecture != conditioned_placement_schema))
        throw std::invalid_argument("incompatible neural model use scope");
    std::vector<std::byte> data;
    for (char c : magic)
        data.push_back(std::byte(c));
    put_u32(data, w.architecture);
    put_u32(data, uint32_t(count));
    put_u32(data, uint32_t(w.provenance.size()));
    if (w.use != ModelUse::Unrestricted || w.hidden_width != 64) {
        data[7] = std::byte(w.use == ModelUse::Unrestricted ? '2' : '3');
        put_u32(data, w.hidden_width);
        if (w.use != ModelUse::Unrestricted)
            put_u32(data, uint32_t(w.use)); // Explicit four-byte little-endian header field.
    }
    for (char c : w.provenance)
        data.push_back(std::byte(c));
    for (float v : w.values) {
        if (!std::isfinite(v))
            throw std::invalid_argument("nonfinite export");
        put_u32(data, std::bit_cast<uint32_t>(v));
    }
    auto sum = sha256(data);
    for (char c : sum)
        data.push_back(std::byte(c));
    static std::atomic<uint64_t> serial{};
    auto temp = file;
    temp += ".tmp." + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
            "." + std::to_string(serial++);
    try {
        std::ofstream f(temp, std::ios::binary | std::ios::trunc);
        f.write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size()));
        f.close();
        if (!f)
            throw std::runtime_error("model write failed");
        std::filesystem::rename(temp, file);
    } catch (...) {
        std::error_code ec;
        std::filesystem::remove(temp, ec);
        throw;
    }
}
} // namespace blitz::neural
