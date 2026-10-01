#pragma once
#include "blitz/remesher.hpp"
#include <stdexcept>
namespace blitz::detail {
inline void reserve_vertices(Mesh& out, MeshView schema, size_t count) {
    out.positions.reserve(count);
    if (schema.normals)
        out.normals.reserve(count);
    if (schema.uv)
        out.uv.reserve(count);
    if (schema.colors)
        out.colors.reserve(count);
    if (schema.tangents)
        out.tangents.reserve(count);
}
template <class T>
void append_stream(std::vector<T>& out, Stream<T> input, size_t first, size_t count) {
    if (!input)
        return;
    auto offset = out.size();
    out.resize(offset + count);
    if (input.stride == sizeof(T))
        std::memcpy(out.data() + offset, input.data + first * input.stride, count * sizeof(T));
    else
        for (size_t i = 0; i < count; ++i)
            out[offset + i] = input[first + i];
}
inline void append_vertices(Mesh& out, MeshView input, size_t first, size_t count) {
    const size_t offset = out.positions.size();
    append_stream(out.positions, input.positions, first, count);
    append_stream(out.normals, input.normals, first, count);
    append_stream(out.uv, input.uv, first, count);
    append_stream(out.colors, input.colors, first, count);
    append_stream(out.tangents, input.tangents, first, count);
    if (!out.exact_position_bits.empty() || !input.exact_position_bits.empty()) {
        out.exact_position_bits.resize((offset + count + 31) / 32);
        for (size_t i = 0; i < count; ++i)
            if (input.exact_position(first + i))
                out.exact_position_bits[(offset + i) / 32] |= 1u << ((offset + i) % 32);
    }
}
class SourceVertices {
    struct Entry {
        uint32_t hash, id;
    }; // Collisions are verified against every supplied stream.
    MeshView source;
    std::vector<Entry> entries;
    static uint32_t hash(MeshView v, uint32_t i) {
        uint32_t h = 2166136261u;
        auto add = [&](auto stream) {
            if (!stream)
                return;
            auto value = stream[i];
            auto p = reinterpret_cast<const uint8_t*>(&value);
            for (size_t k = 0; k < sizeof(value); ++k)
                h = (h ^ p[k]) * 16777619u;
        };
        add(v.positions);
        add(v.normals);
        add(v.uv);
        add(v.colors);
        add(v.tangents);
        return (h ^ uint32_t(v.exact_position(i))) * 16777619u;
    }
    static bool same(MeshView a, uint32_t i, MeshView b, uint32_t j) {
        auto equal = [&](auto x, auto y) {
            if (!x)
                return true;
            auto u = x[i], v = y[j];
            return std::memcmp(&u, &v, sizeof(u)) == 0;
        };
        return equal(a.positions, b.positions) && equal(a.normals, b.normals) &&
               equal(a.uv, b.uv) && equal(a.colors, b.colors) && equal(a.tangents, b.tangents) &&
               a.exact_position(i) == b.exact_position(j);
    }

  public:
    SourceVertices(MeshView v, bool enabled) : source(v) {
        if (!enabled)
            return;
        entries.reserve(v.positions.count);
        for (uint32_t i = 0; i < v.positions.count; ++i)
            entries.push_back({hash(v, i), i});
        std::sort(entries.begin(), entries.end(),
                  [](auto a, auto b) { return std::pair{a.hash, a.id} < std::pair{b.hash, b.id}; });
    }
    void share(Lod& lod) const {
        if (entries.empty() || lod.shared_vertices || lod.vertex_pool)
            return;
        auto v = lod.data.view();
        if (!validate(v).empty())
            return;
        if (bool(v.normals) != bool(source.normals) || bool(v.uv) != bool(source.uv) ||
            bool(v.colors) != bool(source.colors) || bool(v.tangents) != bool(source.tangents))
            return;
        if (v.double_sided.size() != source.double_sided.size() ||
            !std::equal(v.double_sided.begin(), v.double_sided.end(), source.double_sided.begin()))
            return;
        constexpr uint32_t none = UINT32_MAX;
        std::vector<uint32_t> map(v.positions.count, none);
        size_t added = 0;
        for (uint32_t i = 0; i < v.positions.count; ++i) {
            auto h = hash(v, i);
            auto found = std::lower_bound(entries.begin(), entries.end(), h,
                                          [](auto e, uint32_t key) { return e.hash < key; });
            for (; found != entries.end() && found->hash == h; ++found)
                if (same(v, i, source, found->id)) {
                    map[i] = found->id;
                    break;
                }
            added += map[i] == none;
        }
        if (added == v.positions.count || source.positions.count + added >= UINT32_MAX)
            return;
        std::shared_ptr<Mesh> pool;
        if (added) {
            pool = std::make_shared<Mesh>();
            reserve_vertices(*pool, source, source.positions.count + added);
            append_vertices(*pool, source, 0, source.positions.count);
            pool->double_sided.assign(source.double_sided.begin(), source.double_sided.end());
            for (uint32_t i = 0; i < v.positions.count; ++i)
                if (map[i] == none) {
                    map[i] = uint32_t(pool->positions.size());
                    append_vertices(*pool, v, i, 1);
                }
        }
        for (auto& i : lod.data.indices)
            i = map[i];
        Mesh payload;
        payload.indices = std::move(lod.data.indices);
        payload.materials = std::move(lod.data.materials);
        payload.double_sided = std::move(lod.data.double_sided);
        lod.data = std::move(payload);
        lod.shared_vertices = !added;
        lod.source_prefix_vertices = added ? uint32_t(source.positions.count) : 0;
        lod.vertex_pool = std::move(pool);
    }
};
// Consolidate only the selected chain. Proposals own immutable pools separately;
// selected source-prefix levels and LOD0 share one contiguous runtime allocation.
inline void share_result_vertices(Result& r) {
    std::vector<std::shared_ptr<const Mesh>> pools;
    std::vector<uint32_t> offsets;
    const auto prefix = uint32_t(r.source.positions.count);
    size_t count = prefix;
    for (auto& l : r.lods)
        if (l.source_prefix_vertices &&
            std::find(pools.begin(), pools.end(), l.vertex_pool) == pools.end()) {
            offsets.push_back(uint32_t(count));
            pools.push_back(l.vertex_pool);
            count += l.vertex_pool->positions.size() - prefix;
            if (count >= UINT32_MAX)
                throw std::length_error("shared vertex pool exceeds 32-bit IDs");
        }
    if (pools.empty())
        return;
    auto pool = std::make_shared<Mesh>();
    reserve_vertices(*pool, r.source, count);
    append_vertices(*pool, r.source, 0, prefix);
    pool->double_sided.assign(r.source.double_sided.begin(), r.source.double_sided.end());
    for (auto& input : pools)
        append_vertices(*pool, input->view(), prefix, input->positions.size() - prefix);
    // All allocations complete before publishing or changing any index.
    for (auto& l : r.lods)
        if (l.source_prefix_vertices) {
            auto id = size_t(std::find(pools.begin(), pools.end(), l.vertex_pool) - pools.begin());
            for (auto& index : l.data.indices)
                if (index >= prefix)
                    index = offsets[id] + (index - prefix);
            l.vertex_pool = pool;
        } else if (l.shared_vertices)
            l.vertex_pool = pool;
    auto view = pool->view();
    view.positions.count = prefix;
    if (view.normals)
        view.normals.count = prefix;
    if (view.uv)
        view.uv.count = prefix;
    if (view.colors)
        view.colors.count = prefix;
    if (view.tangents)
        view.tangents.count = prefix;
    // Borrow the original prefix bitmap just like its indices/materials. The
    // combined bitmap can contain suffix flags in the prefix's final word.
    view.exact_position_bits = r.source.exact_position_bits;
    view.indices = r.source.indices;
    view.materials = r.source.materials;
    r.source = view;
}
} // namespace blitz::detail
