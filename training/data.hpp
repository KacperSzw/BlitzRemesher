#pragma once
#include "blitz/io.hpp"
#include "neural/internal.hpp"
#include <bit>
#include <chrono>
#include <fstream>
namespace blitz::neural::training {
using json = nlohmann::json;
namespace fs = std::filesystem;
inline json read_json(const fs::path& p) {
    std::ifstream f(p);
    if (!f)
        throw std::runtime_error("cannot read " + p.string());
    json j;
    f >> j;
    return j;
}
inline void write_json(const fs::path& p, const json& j) {
    if (!p.parent_path().empty())
        fs::create_directories(p.parent_path());
    auto temp = p;
    temp += ".part";
    {
        std::ofstream f(temp);
        f << j.dump(2) << '\n';
        if (!f)
            throw std::runtime_error("cannot write " + temp.string());
    }
    fs::rename(temp, p);
}
template <class T> void write_vector(std::ostream& f, const std::vector<T>& a) {
    uint64_t n = a.size();
    f.write(reinterpret_cast<char*>(&n), 8);
    f.write(reinterpret_cast<const char*>(a.data()), std::streamsize(n * sizeof(T)));
}
template <class T> std::vector<T> read_vector(std::istream& f, uint64_t maximum) {
    uint64_t n = 0;
    f.read(reinterpret_cast<char*>(&n), 8);
    if (!f || n > maximum)
        throw std::invalid_argument("invalid training shard length");
    std::vector<T> a(n);
    f.read(reinterpret_cast<char*>(a.data()), std::streamsize(n * sizeof(T)));
    if (!f)
        throw std::invalid_argument("truncated training shard");
    return a;
}
struct Example {
    std::array<float, conditions> condition;
    std::vector<uint32_t> representative;
    std::vector<uint8_t> retained;
};
struct Asset {
    Graph graph;
    std::vector<Example> examples;
};
inline void save_asset(const fs::path& path, const Asset& a) {
    static_assert(std::endian::native == std::endian::little);
    auto temp = path;
    temp += ".part";
    {
        std::ofstream f(temp, std::ios::binary);
        f.write("BLZDATA1", 8);
        write_vector(f, a.graph.x);
        write_vector(f, a.graph.offsets);
        write_vector(f, a.graph.neighbors);
        write_vector(f, a.graph.flags);
        uint32_t n = uint32_t(a.examples.size());
        f.write(reinterpret_cast<char*>(&n), 4);
        for (auto& e : a.examples) {
            f.write(reinterpret_cast<const char*>(e.condition.data()), sizeof(e.condition));
            write_vector(f, e.representative);
            write_vector(f, e.retained);
        }
        f.close();
        if (!f)
            throw std::runtime_error("training shard write failed");
    }
    fs::rename(temp, path);
}
inline Asset load_asset(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    char magic[8];
    f.read(magic, 8);
    if (!f || std::memcmp(magic, "BLZDATA1", 8))
        throw std::invalid_argument("invalid training shard");
    Asset a;
    a.graph.x = read_vector<float>(f, 1000000000);
    a.graph.offsets = read_vector<uint32_t>(f, 40000001);
    a.graph.neighbors = read_vector<uint32_t>(f, 240000000);
    a.graph.flags = read_vector<uint8_t>(f, 40000000);
    size_t n = a.graph.size();
    if (a.graph.x.size() != n * features || a.graph.offsets.size() != n + 1 ||
        a.graph.offsets[0] != 0 || a.graph.offsets.back() != a.graph.neighbors.size() ||
        !std::is_sorted(a.graph.offsets.begin(), a.graph.offsets.end()))
        throw std::invalid_argument("invalid training graph dimensions");
    for (auto i : a.graph.neighbors)
        if (i >= n)
            throw std::invalid_argument("invalid training neighbor");
    uint32_t count = 0;
    f.read(reinterpret_cast<char*>(&count), 4);
    if (!f || !count || count > 64)
        throw std::invalid_argument("invalid example count");
    for (uint32_t i = 0; i < count; ++i) {
        Example e;
        f.read(reinterpret_cast<char*>(e.condition.data()), sizeof(e.condition));
        e.representative = read_vector<uint32_t>(f, n);
        e.retained = read_vector<uint8_t>(f, n);
        if (e.representative.size() != n || e.retained.size() != n)
            throw std::invalid_argument("example dimensions differ");
        for (auto id : e.representative)
            if (id >= n)
                throw std::invalid_argument("invalid representative");
        a.examples.push_back(std::move(e));
    }
    if (f.peek() != EOF)
        throw std::invalid_argument("unexpected training shard tail");
    return a;
}
} // namespace blitz::neural::training
