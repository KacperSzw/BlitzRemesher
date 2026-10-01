#pragma once
#include "blitz/remesher.hpp"
#include <numeric>
namespace blitz::detail {
// Corner connectivity preserves vertex-only contacts and index seams. Boundary
// fans are anchors: never join two of them, even at an original seam endpoint.
// All IDs address source corners (<UINT32_MAX); UINT32_MAX means no live output.
inline bool merge_wedges(MeshView source, std::span<const uint32_t> positions,
                         std::span<const uint32_t> clusters, std::span<const uint32_t> faces,
                         Mesh& emitted, ReductionStats* stats) {
    constexpr uint32_t none = UINT32_MAX;
    const auto count = source.indices.size();
    std::vector<uint32_t> parent(count);
    std::iota(parent.begin(), parent.end(), 0);
    std::vector<uint8_t> boundary(count);
    auto root = [&](uint32_t i) {
        while (parent[i] != i) {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        return i;
    };
    auto next = [](uint32_t c) { return c / 3 * 3 + (c % 3 + 1) % 3; };
    auto join = [&](uint32_t a, uint32_t b) {
        a = root(a);
        b = root(b);
        if (a == b)
            return;
        if (boundary[a] && boundary[b])
            return;
        auto x = source.indices[a], y = source.indices[b];
        if ((source.colors && source.colors[x].a != source.colors[y].a) ||
            (source.tangents && source.tangents[x].w != source.tangents[y].w))
            return;
        if (boundary[b] || (!boundary[a] && b < a))
            std::swap(a, b);
        parent[b] = a;
    };
    // Three u32 values avoid the padding of a u64 key plus corner ID.
    struct Edge {
        uint32_t a, b, corner;
    };
    std::vector<Edge> edges;
    edges.reserve(count);
    for (uint32_t c = 0; c < count; ++c) {
        auto a = positions[source.indices[c]], b = positions[source.indices[next(c)]];
        edges.push_back({std::min(a, b), std::max(a, b), c});
    }
    std::sort(edges.begin(), edges.end(), [](auto a, auto b) {
        return std::array{a.a, a.b, a.corner} < std::array{b.a, b.b, b.corner};
    });
    for (size_t i = 0; i < edges.size();) {
        size_t j = i + 1;
        while (j < edges.size() && edges[j].a == edges[i].a && edges[j].b == edges[i].b)
            ++j;
        bool continuous = false;
        if (j == i + 2 && edges[i].a != edges[i].b) {
            auto a = edges[i].corner, b = edges[i + 1].corner;
            continuous = source.indices[a] == source.indices[next(b)] &&
                         source.indices[next(a)] == source.indices[b] &&
                         source.material(a / 3) == source.material(b / 3);
            if (continuous)
                for (auto pair : {std::pair{a, next(b)}, std::pair{next(a), b}}) {
                    auto x = root(pair.first), y = root(pair.second);
                    parent[y] = x;
                }
        }
        if (!continuous)
            for (size_t k = i; k < j; ++k) {
                boundary[edges[k].corner] = 1;
                boundary[next(edges[k].corner)] = 1;
            }
        i = j;
    }
    // Mark anchors only after ordinary corner fans are connected.
    for (uint32_t c = 0; c < count; ++c)
        if (boundary[c])
            boundary[root(c)] = 1;
    std::vector<uint32_t> fan(count);
    for (uint32_t c = 0; c < count; ++c)
        fan[c] = root(c);
    const uint64_t edge_bytes = edges.capacity() * sizeof(Edge);
    std::vector<Edge>().swap(edges);
    for (uint32_t c = 0; c < count; ++c) {
        auto d = next(c);
        if (clusters[positions[source.indices[c]]] == clusters[positions[source.indices[d]]])
            join(c, d);
    }
    for (uint32_t c = 0; c < count; ++c)
        parent[c] = root(c);
    std::vector<uint32_t> output(count, none);
    uint32_t vertices = 0;
    for (auto f : faces)
        for (unsigned k = 0; k < 3; ++k) {
            auto r = parent[3 * f + k];
            if (output[r] == none)
                output[r] = vertices++;
        }
    // Accumulate only live output fields: area plus present normal/RGB/UV values.
    const uint8_t channels = uint8_t(3 * bool(source.normals) + 3 * bool(source.colors) +
                                     2 * bool(source.uv)),
                  stride = uint8_t(channels + 1);
    std::vector<double> sums(size_t(vertices) * stride);
    if (stats)
        stats->appearance_bytes +=
            (parent.capacity() + fan.capacity()) * 4 + boundary.capacity() +
            std::max(edge_bytes, output.capacity() * 8 + sums.capacity() * 8);
    auto values = [&](uint32_t i, double* a) {
        unsigned c = 0;
        if (source.normals) {
            auto n = normalized(source.normals[i]);
            a[c++] = n.x;
            a[c++] = n.y;
            a[c++] = n.z;
        }
        if (source.colors) {
            auto v = source.colors[i];
            a[c++] = double(v.r) / 255;
            a[c++] = double(v.g) / 255;
            a[c++] = double(v.b) / 255;
        }
        if (source.uv) {
            auto v = source.uv[i];
            a[c++] = v.x;
            a[c++] = v.y;
        }
    };
    const auto b = bounds(source);
    const double scale = b.diameter();
    for (uint32_t f = 0; f < source.triangles(); ++f) {
        double p[3][3], a[3][8];
        for (unsigned k = 0; k < 3; ++k) {
            auto id = source.indices[3 * f + k];
            auto v = source.positions[id];
            p[k][0] = (double(v.x) - b.center.x) / scale;
            p[k][1] = (double(v.y) - b.center.y) / scale;
            p[k][2] = (double(v.z) - b.center.z) / scale;
            values(id, a[k]);
        }
        double e[3], z[3], ee = 0, ez = 0, zz = 0;
        for (unsigned k = 0; k < 3; ++k) {
            e[k] = p[1][k] - p[0][k];
            z[k] = p[2][k] - p[0][k];
            ee += e[k] * e[k];
            ez += e[k] * z[k];
            zz += z[k] * z[k];
        }
        const double det = ee * zz - ez * ez;
        if (!(det > 1e-14 * ee * zz) || det <= 1e-48)
            continue;
        const double area = std::sqrt(det);
        for (unsigned k = 0; k < 3; ++k) {
            auto id = output[parent[3 * f + k]];
            if (id == none)
                continue;
            auto v = emitted.positions[source.indices[3 * f + k]];
            double d[3] = {(double(v.x) - b.center.x) / scale - p[0][0],
                           (double(v.y) - b.center.y) / scale - p[0][1],
                           (double(v.z) - b.center.z) / scale - p[0][2]};
            double de = 0, dz = 0;
            for (unsigned j = 0; j < 3; ++j) {
                de += d[j] * e[j];
                dz += d[j] * z[j];
            }
            const double u = (de * zz - dz * ez) / det, w = (dz * ee - de * ez) / det;
            auto s = sums.data() + size_t(id) * stride;
            s[0] += area;
            for (unsigned c = 0; c < channels; ++c)
                s[c + 1] += area * (a[0][c] + u * (a[1][c] - a[0][c]) + w * (a[2][c] - a[0][c]));
        }
    }
    Mesh fused;
    fused.positions.resize(vertices);
    fused.double_sided = emitted.double_sided;
    fused.materials = emitted.materials;
    if (!emitted.exact_position_bits.empty()) {
        fused.exact_position_bits.resize((size_t(vertices) + 31) / 32);
        // A fitted wedge inherits exact storage if any merged contributor
        // required it, including contributors whose original face was removed.
        for (uint32_t c = 0; c < count; ++c) {
            auto id = output[parent[c]];
            if (id != none && emitted.view().exact_position(source.indices[c]))
                fused.exact_position_bits[id / 32] |= 1u << (id % 32);
        }
    }
    if (source.normals)
        fused.normals.resize(vertices);
    if (source.colors)
        fused.colors.resize(vertices);
    if (source.uv)
        fused.uv.resize(vertices);
    if (source.tangents)
        fused.tangents.resize(vertices);
    for (uint32_t r = 0; r < count; ++r)
        if (output[r] != none) {
            auto id = output[r], original = source.indices[r];
            fused.positions[id] = emitted.positions[original];
            double a[8];
            values(original, a);
            auto sum = sums.data() + size_t(id) * stride;
            if (sum[0] > 0)
                for (unsigned c = 0; c < channels; ++c)
                    if (std::isfinite(sum[c + 1] / sum[0]))
                        a[c] = sum[c + 1] / sum[0];
            unsigned c = 0;
            if (source.normals) {
                Vec3 n{float(a[0]), float(a[1]), float(a[2])};
                fused.normals[id] = finite(n) && length(n) > 1e-20
                                        ? normalized(n)
                                        : normalized(source.normals[original]);
                c = 3;
            }
            if (source.colors) {
                auto col = source.colors[original];
                fused.colors[id] =
                    quantize_color(std::clamp(a[c], 0., 1.), std::clamp(a[c + 1], 0., 1.),
                                   std::clamp(a[c + 2], 0., 1.), double(col.a) / 255);
                c += 3;
            }
            if (source.uv) {
                Vec2 uv{float(a[c]), float(a[c + 1])};
                if (boundary[r] || !std::isfinite(uv.x) || !std::isfinite(uv.y))
                    uv = source.uv[original];
                fused.uv[id] = uv;
            }
            if (source.tangents) {
                auto t = source.tangents[original];
                Vec3 v{t.x, t.y, t.z};
                if (source.normals) {
                    auto n = fused.normals[id];
                    v = v - n * dot(v, n);
                    if (length(v) < 1e-20)
                        v = cross(n, std::abs(n.x) < .8 ? Vec3{1, 0, 0} : Vec3{0, 1, 0});
                }
                v = normalized(v);
                fused.tangents[id] = {v.x, v.y, v.z, t.w};
            }
        }
    fused.indices.reserve(faces.size() * 3);
    for (auto f : faces)
        for (unsigned k = 0; k < 3; ++k)
            fused.indices.push_back(output[parent[3 * f + k]]);
    Mesh repaired;
    std::vector<uint32_t> remap(count, none);
    for (unsigned round = 0; round < 4; ++round) {
        auto& candidate = round ? repaired : fused;
        bool valid = true, changed = false;
        if (source.uv)
            for (size_t f = 0; f < faces.size(); ++f) {
                Vec2 old[3], now[3];
                for (unsigned k = 0; k < 3; ++k) {
                    old[k] = source.uv[source.indices[3 * faces[f] + k]];
                    now[k] = candidate.uv[candidate.indices[3 * f + k]];
                }
                auto area = [](Vec2* p) {
                    return (double(p[1].x) - p[0].x) * (double(p[2].y) - p[0].y) -
                           (double(p[1].y) - p[0].y) * (double(p[2].x) - p[0].x);
                };
                auto before = area(old), after = area(now);
                if (std::abs(before) > 1e-20 && before * after <= 0) {
                    valid = false;
                    if (stats)
                        ++stats->uv_rejections;
                    for (unsigned k = 0; k < 3; ++k) {
                        auto r = parent[3 * faces[f] + k];
                        changed |= !(boundary[r] & 2);
                        boundary[r] |= 2;
                    }
                }
            }
        if (valid) {
            emitted = std::move(candidate);
            return true;
        }
        if (!changed || round == 3)
            return false;
        // Restore only affected original fans and attributes. Recheck adjacent
        // UV triangles after every repair; a bounded failure keeps old emission.
        std::fill(remap.begin(), remap.end(), none);
        repaired = {};
        repaired.double_sided = emitted.double_sided;
        repaired.materials = emitted.materials;
        for (auto f : faces)
            for (unsigned k = 0; k < 3; ++k) {
                auto corner = 3 * f + k, r = parent[corner];
                bool split = boundary[r] & 2;
                auto key = split ? fan[corner] : r;
                if (remap[key] == none) {
                    remap[key] = uint32_t(repaired.positions.size());
                    const auto& from = split ? emitted : fused;
                    auto id = split ? source.indices[key] : output[r];
                    repaired.positions.push_back(from.positions[id]);
                    if (source.normals)
                        repaired.normals.push_back(from.normals[id]);
                    if (source.colors)
                        repaired.colors.push_back(from.colors[id]);
                    if (source.uv)
                        repaired.uv.push_back(from.uv[id]);
                    if (source.tangents)
                        repaired.tangents.push_back(from.tangents[id]);
                    if (!repaired.exact_position_bits.empty() ||
                        !from.exact_position_bits.empty()) {
                        auto output_id = remap[key];
                        repaired.exact_position_bits.resize((repaired.positions.size() + 31) / 32);
                        if (from.view().exact_position(id))
                            repaired.exact_position_bits[output_id / 32] |= 1u << (output_id % 32);
                    }
                }
                repaired.indices.push_back(remap[key]);
            }
    }
    return false;
}
} // namespace blitz::detail
