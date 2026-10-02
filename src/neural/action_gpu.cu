#include "neural/action_gpu.hpp"
#include "neural/cuda.cuh"
#include "neural/upload.cuh"
#include "neural/vertex_storage.hpp"
#include <atomic>
#include <chrono>
#include <climits>
#include <cub/cub.cuh>
#include <math_constants.h>
#include <numeric>

namespace blitz::neural {
namespace {
using namespace gpu;
constexpr uint32_t none = UINT32_MAX;
__device__ Vec3 add3(Vec3 a, Vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
__device__ Vec3 sub3(Vec3 a, Vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
__device__ Vec3 mul3(Vec3 a, double s) {
    return {float(a.x * s), float(a.y * s), float(a.z * s)};
}
__device__ double dot3(Vec3 a, Vec3 b) {
    return double(a.x) * b.x + double(a.y) * b.y + double(a.z) * b.z;
}
__device__ Vec3 cross3(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
__device__ double len3(Vec3 a) {
    return sqrt(dot3(a, a));
}
__device__ Vec3 norm3(Vec3 a) {
    double n = len3(a);
    return n > 0 ? mul3(a, 1 / n) : Vec3{};
}
__device__ double area2(Vec2 a, Vec2 b, Vec2 c) {
    return (double(b.x) - a.x) * (double(c.y) - a.y) - (double(b.y) - a.y) * (double(c.x) - a.x);
}
__device__ uint32_t bits(float f) {
    return __float_as_uint(f == 0 ? 0.f : f);
}
__device__ bool same_position(Vec3 a, Vec3 b) {
    return bits(a.x) == bits(b.x) && bits(a.y) == bits(b.y) && bits(a.z) == bits(b.z);
}
template <class T> __device__ bool same_item(const T* data, uint32_t a, uint32_t b) {
    if (!data)
        return true;
    auto x = reinterpret_cast<const unsigned char*>(data + a),
         y = reinterpret_cast<const unsigned char*>(data + b);
    for (unsigned i = 0; i < sizeof(T); ++i)
        if (x[i] != y[i])
            return false;
    return true;
}
__device__ bool same_attributes(DeviceMeshView m, uint32_t a, uint32_t b, bool preserve_uv = true) {
    return same_item(m.normals, a, b) && same_item(m.colors, a, b) &&
           (!preserve_uv || (same_item(m.uv, a, b) && same_item(m.tangents, a, b)));
}
__device__ uint32_t hash_position(Vec3 p) {
    uint32_t h = bits(p.x) * 0x9e3779b9u;
    h = (h ^ bits(p.y)) * 0x85ebca6bu;
    return (h ^ bits(p.z)) * 0xc2b2ae35u;
}
struct DeviceMesh {
    Buffer<Vec3> positions, normals;
    Buffer<Vec2> uv;
    Buffer<ColorRGBA8> colors;
    Buffer<Vec4> tangents;
    Buffer<uint32_t> indices, precision;
    Buffer<uint16_t> materials;
    Buffer<uint8_t> sided;
    DeviceMeshView view;
    static uint64_t identity() {
        static std::atomic<uint64_t> serial{1};
        return serial++;
    }
    template <class T> static Buffer<T> clone(Device& d, const Buffer<T>& source) {
        Buffer<T> out(d, source.n);
        if (source.n)
            check(gpu::copy(out.p, source.p, source.n * sizeof(T), cudaMemcpyDeviceToDevice));
        return out;
    }
    void bind() {
        view.positions = positions.p;
        view.normals = normals.p;
        view.uv = uv.p;
        view.colors = colors.p;
        view.tangents = tangents.p;
        view.indices = indices.p;
        view.materials = materials.p;
        view.double_sided = sided.p;
        view.exact_position_bits = precision.p;
        view.identity = identity();
    }
    DeviceMesh(Device& d, const DeviceMesh& m)
        : positions(clone(d, m.positions)), normals(clone(d, m.normals)), uv(clone(d, m.uv)),
          colors(clone(d, m.colors)), tangents(clone(d, m.tangents)), indices(clone(d, m.indices)),
          precision(clone(d, m.precision)), materials(clone(d, m.materials)),
          sided(clone(d, m.sided)), view(m.view) {
        bind();
    }
    DeviceMesh(Device& d, MeshView m, bool packed = false, const VertexBounds* domain = nullptr)
        : positions(packed ? Buffer<Vec3>(d, m.positions.count) : upload_stream(d, m.positions)),
          normals(packed ? Buffer<Vec3>(d, m.normals.count) : upload_stream(d, m.normals)),
          uv(packed ? Buffer<Vec2>(d, m.uv.count) : upload_stream(d, m.uv)),
          colors(upload_stream(d, m.colors)),
          tangents(packed ? Buffer<Vec4>(d, m.tangents.count) : upload_stream(d, m.tangents)),
          indices(d, m.indices.size()), precision(d, m.exact_position_bits.size()),
          materials(d, m.materials.size()), sided(d, m.double_sided.size()) {
        if (packed)
            upload_working_mesh(d, m, domain ? *domain : vertex_bounds(m), positions.p, normals.p,
                                uv.p, tangents.p);
        indices.upload(m.indices);
        precision.upload(m.exact_position_bits);
        materials.upload(m.materials);
        sided.upload(m.double_sided);
        view = {positions.p,
                normals.p,
                uv.p,
                colors.p,
                tangents.p,
                indices.p,
                materials.p,
                sided.p,
                uint32_t(m.positions.count),
                uint32_t(m.triangles()),
                uint32_t(m.double_sided.size()),
                identity(),
                0};
        view.exact_position_bits = precision.p;
    }
};
struct State {
    uint64_t ranked;
    uint32_t faces, revision, actions, trials, accepted, rejected, error, position, first,
        invalid_placement;
    uint8_t selected, accepted_trial, exhausted, padding;
};
struct Edit {
    uint32_t from, to, source[2], target[2];
    uint8_t wedges, removed, valid, padding;
};
struct TopologyView {
    const uint64_t* edges;
    const uint32_t *counts, *length, *offsets, *neighbors, *face_offsets, *faces;
    const uint32_t* invalid;
};
struct Topology {
    Buffer<uint64_t> raw, sorted, edges;
    Buffer<uint32_t> counts, length, offsets, neighbors, face_offsets, faces, next;
    Buffer<uint32_t> invalid;
    Buffer<uint8_t> visited;
    std::unique_ptr<Buffer<std::byte>> temp;
    size_t sort_bytes{}, encode_bytes{}, scan_bytes{};
    uint32_t vertices, entries;
    Topology(Device& d, uint32_t nv, uint32_t nf)
        : raw(d, size_t(nf) * 3), sorted(d, size_t(nf) * 3), edges(d, size_t(nf) * 3),
          counts(d, size_t(nf) * 3), length(d, 1), offsets(d, size_t(nv) + 1),
          neighbors(d, size_t(nf) * 6), face_offsets(d, size_t(nv) + 1), faces(d, size_t(nf) * 3),
          next(d, size_t(nv) + 1), invalid(d, nv), visited(d, size_t(nf) * 6), vertices(nv),
          entries(nf * 3) {
        check(cub::DeviceRadixSort::SortKeys(nullptr, sort_bytes, raw.p, sorted.p, int(entries), 0,
                                             64, gpu::stream()));
        check(cub::DeviceRunLengthEncode::Encode(nullptr, encode_bytes, sorted.p, edges.p, counts.p,
                                                 length.p, int(entries), gpu::stream()));
        check(cub::DeviceScan::InclusiveSum(nullptr, scan_bytes, offsets.p, offsets.p, int(nv + 1),
                                            gpu::stream()));
        temp = std::make_unique<Buffer<std::byte>>(
            d, std::max({sort_bytes, encode_bytes, scan_bytes}));
    }
    TopologyView view() const {
        return {edges.p,     counts.p,       length.p, offsets.p,
                neighbors.p, face_offsets.p, faces.p,  invalid.p};
    }
};
__global__ void insert_vertices(DeviceMeshView m, uint32_t* slots, uint32_t capacity,
                                bool attributes, bool preserve_uv) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m.vertices)
        return;
    uint32_t at = hash_position(m.positions[i]) & (capacity - 1);
    for (;;) {
        uint32_t old = atomicCAS(slots + at, none, i);
        if (old == none)
            return;
        if (same_position(m.positions[i], m.positions[old]) &&
            (!attributes || same_attributes(m, i, old, preserve_uv))) {
            atomicMin(slots + at, i);
            return;
        }
        at = (at + 1) & (capacity - 1);
    }
}
__global__ void lookup_vertices(DeviceMeshView m, const uint32_t* slots, uint32_t capacity,
                                uint32_t* ids, bool attributes, bool preserve_uv) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m.vertices)
        return;
    uint32_t at = hash_position(m.positions[i]) & (capacity - 1);
    for (;;) {
        uint32_t id = slots[at];
        if (same_position(m.positions[i], m.positions[id]) &&
            (!attributes || same_attributes(m, i, id, preserve_uv))) {
            ids[i] = id;
            return;
        }
        at = (at + 1) & (capacity - 1);
    }
}
__global__ void weld_uv_indices(uint32_t* indices, const uint32_t* canonical, uint32_t count) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        indices[i] = canonical[indices[i]];
}
__global__ void edge_keys(DeviceMeshView m, const State* state, const uint32_t* ids, uint64_t* keys,
                          uint32_t* incident, uint32_t* invalid, uint32_t capacity) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= capacity)
        return;
    keys[i] = UINT64_MAX;
    if (i >= state->faces * 3)
        return;
    uint32_t face = i / 3, k = i % 3, a = ids[m.indices[i]],
             b = ids[m.indices[face * 3 + (k + 1) % 3]];
    atomicAdd(incident + a + 1, 1u);
    if (a == b) {
        for (unsigned j = 0; j < 3; ++j)
            atomicOr(invalid + ids[m.indices[face * 3 + j]], 1u);
        return;
    }
    keys[i] = (uint64_t(min(a, b)) << 32) | max(a, b);
}
__global__ void edge_degrees(const uint64_t* edges, const uint32_t* counts, const uint32_t* size,
                             uint32_t* offsets, uint32_t* invalid, uint32_t capacity) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= capacity || i >= *size || edges[i] == UINT64_MAX)
        return;
    uint32_t a = uint32_t(edges[i] >> 32), b = uint32_t(edges[i]);
    atomicAdd(offsets + a + 1, 1u);
    atomicAdd(offsets + b + 1, 1u);
    if (counts[i] > 2) {
        atomicOr(invalid + a, 1u);
        atomicOr(invalid + b, 1u);
    }
}
__global__ void scatter_edges(TopologyView t, uint32_t* next, uint32_t* neighbors,
                              uint32_t capacity) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= capacity || i >= *t.length || t.edges[i] == UINT64_MAX)
        return;
    uint32_t a = uint32_t(t.edges[i] >> 32), b = uint32_t(t.edges[i]);
    neighbors[atomicAdd(next + a, 1u)] = b;
    neighbors[atomicAdd(next + b, 1u)] = a;
}
__global__ void scatter_faces(DeviceMeshView m, const State* state, const uint32_t* ids,
                              uint32_t* next, uint32_t* faces, uint32_t capacity) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= capacity || i >= state->faces * 3)
        return;
    faces[atomicAdd(next + ids[m.indices[i]], 1u)] = i / 3;
}
__global__ void sort_lists(const uint32_t* offsets, uint32_t* values, uint32_t vertices) {
    uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= vertices)
        return;
    for (uint32_t i = offsets[v] + 1; i < offsets[v + 1]; ++i) {
        uint32_t value = values[i], j = i;
        while (j > offsets[v] && values[j - 1] > value) {
            values[j] = values[j - 1];
            --j;
        }
        values[j] = value;
    }
}
__device__ uint32_t find_neighbor(TopologyView t, uint32_t u, uint32_t v) {
    uint32_t lo = t.offsets[u], hi = t.offsets[u + 1];
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (t.neighbors[mid] < v)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo < t.offsets[u + 1] && t.neighbors[lo] == v ? lo : none;
}
__global__ void connected_fans(DeviceMeshView m, TopologyView t, const uint32_t* geometry,
                               uint32_t* invalid, uint8_t* visited) {
    uint32_t u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= m.vertices || invalid[u] || t.face_offsets[u] == t.face_offsets[u + 1])
        return;
    uint32_t start = t.offsets[u], end = t.offsets[u + 1];
    for (uint32_t j = start; j < end; ++j)
        visited[j] = 0;
    uint32_t first = t.faces[t.face_offsets[u]], seed = none;
    for (unsigned j = 0; j < 3; ++j) {
        auto v = geometry[m.indices[first * 3 + j]];
        if (v != u) {
            seed = v;
            break;
        }
    }
    auto index = find_neighbor(t, u, seed);
    if (index == none) {
        invalid[u] = 1;
        return;
    }
    visited[index] = 1;
    bool changed = true;
    while (changed) {
        changed = false;
        for (uint32_t k = t.face_offsets[u]; k < t.face_offsets[u + 1]; ++k) {
            uint32_t f = t.faces[k], other[2], n = 0;
            for (unsigned j = 0; j < 3; ++j) {
                uint32_t v = geometry[m.indices[f * 3 + j]];
                if (v != u && n < 2)
                    other[n++] = v;
            }
            if (n != 2) {
                invalid[u] = 1;
                return;
            }
            auto a = find_neighbor(t, u, other[0]), b = find_neighbor(t, u, other[1]);
            if (a == none || b == none) {
                invalid[u] = 1;
                return;
            }
            if (visited[a] && !visited[b]) {
                visited[b] = 1;
                changed = true;
            }
            if (visited[b] && !visited[a]) {
                visited[a] = 1;
                changed = true;
            }
        }
    }
    for (uint32_t j = start; j < end; ++j)
        if (!visited[j]) {
            invalid[u] = 1;
            return;
        }
}
__device__ uint16_t material(DeviceMeshView m, uint32_t f) {
    return m.materials ? m.materials[f] : 0;
}
__device__ bool boundary(TopologyView t, uint32_t u) {
    for (uint32_t k = t.offsets[u]; k < t.offsets[u + 1]; ++k) {
        uint32_t v = t.neighbors[k];
        uint64_t key = (uint64_t(min(u, v)) << 32) | max(u, v);
        uint32_t lo = 0, hi = *t.length;
        while (lo < hi) {
            uint32_t mid = lo + (hi - lo) / 2;
            if (t.edges[mid] < key)
                lo = mid + 1;
            else
                hi = mid;
        }
        if (lo < *t.length && t.counts[lo] == 1)
            return true;
    }
    return false;
}
__device__ Vec3 scaled(DeviceMeshView m, uint32_t i, Bounds b) {
    auto p = m.positions[i];
    double scale = b.radius * 2;
    return {float((double(p.x) - b.center.x) / scale), float((double(p.y) - b.center.y) / scale),
            float((double(p.z) - b.center.z) / scale)};
}
__device__ bool exact_position(DeviceMeshView m, uint32_t i) {
    return m.exact_position_bits && ((m.exact_position_bits[i / 32] >> (i % 32)) & 1);
}
__device__ bool exact_edit(DeviceMeshView m, Edit e) {
    for (unsigned j = 0; j < e.wedges; ++j)
        if (exact_position(m, e.source[j]) || exact_position(m, e.target[j]))
            return true;
    return false;
}
__device__ Vec3 grid_position(DeviceMeshView m, Vec3 p) {
    if (!m.fixed_quantization)
        return p;
    auto lo = m.quant_low, e = m.quant_extent;
    if (p.x < lo.x || p.y < lo.y || p.z < lo.z || p.x > lo.x + e.x || p.y > lo.y + e.y ||
        p.z > lo.z + e.z)
        return p;
    return {unpack_unorm16(pack_unorm16(p.x, lo.x, e.x), lo.x, e.x),
            unpack_unorm16(pack_unorm16(p.y, lo.y, e.y), lo.y, e.y),
            unpack_unorm16(pack_unorm16(p.z, lo.z, e.z), lo.z, e.z)};
}
__global__ void vertex_features(DeviceMeshView m, TopologyView t, const uint32_t* canonical,
                                const uint32_t* geometry, const uint32_t* seams, Bounds b,
                                float* features) {
    uint32_t u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= m.vertices)
        return;
    Vec3 normal{};
    bool junction = false;
    uint16_t first_material = 0;
    bool seen = false;
    for (uint32_t k = t.face_offsets[u]; k < t.face_offsets[u + 1]; ++k) {
        auto f = t.faces[k];
        auto p = scaled(m, canonical[m.indices[f * 3]], b),
             q = scaled(m, canonical[m.indices[f * 3 + 1]], b),
             r = scaled(m, canonical[m.indices[f * 3 + 2]], b);
        normal = add3(normal, norm3(cross3(sub3(q, p), sub3(r, p))));
        auto mat = material(m, f);
        if (seen && mat != first_material)
            junction = true;
        first_material = mat;
        seen = true;
    }
    normal = norm3(m.normals ? m.normals[u] : normal);
    auto p = scaled(m, u, b);
    Vec4 color{1, 1, 1, 1};
    if (m.colors) {
        auto c = m.colors[u];
        color = {c.r / 255.f, c.g / 255.f, c.b / 255.f, c.a / 255.f};
    }
    Vec2 uv = m.uv ? m.uv[u] : Vec2{};
    Vec4 tangent = m.tangents ? m.tangents[u] : Vec4{};
    auto direction = norm3({tangent.x, tangent.y, tangent.z});
    double edge = 0;
    for (uint32_t k = t.offsets[u]; k < t.offsets[u + 1]; ++k)
        edge += len3(sub3(scaled(m, t.neighbors[k], b), p));
    uint32_t degree = t.offsets[u + 1] - t.offsets[u];
    edge /= max(1u, degree);
    float values[24] = {p.x,
                        p.y,
                        p.z,
                        normal.x,
                        normal.y,
                        normal.z,
                        color.x,
                        color.y,
                        color.z,
                        fminf(16, fmaxf(-16, uv.x)) / 16,
                        fminf(16, fmaxf(-16, uv.y)) / 16,
                        direction.x,
                        direction.y,
                        direction.z,
                        tangent.w,
                        float(bool(m.normals)),
                        float(bool(m.uv)),
                        float(bool(m.colors)),
                        float(bool(m.tangents)),
                        float(seams[geometry[u]] != 0),
                        float(boundary(t, u)),
                        float(junction),
                        float(log2(double(1 + degree)) / 8),
                        float(log2(1 + edge * 1024) / 10)};
    for (unsigned j = 0; j < 24; ++j)
        features[size_t(u) * 24 + j] = values[j];
}
__global__ void mark_seams(DeviceMeshView m, const uint32_t* geometry, uint32_t* seams,
                           bool preserve_uv) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m.vertices && !same_attributes(m, i, geometry[i], preserve_uv))
        atomicOr(seams + geometry[i], 1u);
}
__device__ uint32_t map_vertex(const Edit& edit, uint32_t id) {
    for (unsigned j = 0; j < edit.wedges; ++j)
        if (id == edit.source[j])
            return edit.target[j];
    return id;
}
__device__ void triple(uint32_t* x) {
    if (x[0] > x[1]) {
        auto t = x[0];
        x[0] = x[1];
        x[1] = t;
    }
    if (x[1] > x[2]) {
        auto t = x[1];
        x[1] = x[2];
        x[2] = t;
    }
    if (x[0] > x[1]) {
        auto t = x[0];
        x[0] = x[1];
        x[1] = t;
    }
}
__device__ bool face_equal(const uint32_t* a, const uint32_t* b) {
    return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
}
__device__ bool legal(DeviceMeshView m, const State* state, TopologyView t,
                      const uint32_t* geometry, Edit& edit, bool placement, bool preserve_uv) {
    uint32_t u = edit.from, v = edit.to;
    if (t.invalid[u] || t.invalid[v])
        return false;
    uint32_t opposite[2], edge_faces = 0;
    for (uint32_t k = t.face_offsets[u]; k < t.face_offsets[u + 1]; ++k) {
        uint32_t f = t.faces[k], a = none, b = none, w = none;
        for (unsigned j = 0; j < 3; ++j) {
            auto id = m.indices[f * 3 + j], g = geometry[id];
            if (g == u)
                a = id;
            else if (g == v)
                b = id;
            else
                w = g;
        }
        if (a == none)
            return false;
        if (b != none) {
            if (edge_faces == 2 || w == none)
                return false;
            opposite[edge_faces++] = w;
            bool found = false;
            for (unsigned j = 0; j < edit.wedges; ++j)
                found |= edit.source[j] == a && edit.target[j] == b;
            if (!found) {
                if (edit.wedges == 2)
                    return false;
                edit.source[edit.wedges] = a;
                edit.target[edit.wedges++] = b;
            }
        }
    }
    if (!edge_faces || edge_faces >= state->faces ||
        (edge_faces == 2 && opposite[0] == opposite[1]))
        return false;
    if (boundary(t, u) && (!boundary(t, v) || edge_faces != 1))
        return false;
    uint32_t common = 0;
    for (uint32_t k = t.offsets[u]; k < t.offsets[u + 1]; ++k) {
        auto w = t.neighbors[k];
        if (find_neighbor(t, v, w) != none) {
            if (w != opposite[0] && (edge_faces < 2 || w != opposite[1]))
                return false;
            ++common;
        }
    }
    if (common != edge_faces)
        return false;
    if (edit.wedges == 2) {
        if (edit.source[0] == edit.source[1])
            return false;
        if (edit.source[0] > edit.source[1]) {
            auto a = edit.source[0];
            edit.source[0] = edit.source[1];
            edit.source[1] = a;
            a = edit.target[0];
            edit.target[0] = edit.target[1];
            edit.target[1] = a;
        }
    }
    uint16_t first = material(m, t.faces[t.face_offsets[u]]);
    bool junction = false;
    for (uint32_t k = t.face_offsets[u]; k < t.face_offsets[u + 1]; ++k) {
        auto f = t.faces[k];
        junction |= material(m, f) != first;
        for (unsigned j = 0; j < 3; ++j) {
            auto id = m.indices[f * 3 + j];
            if (geometry[id] == u) {
                bool mapped = false;
                for (unsigned l = 0; l < edit.wedges; ++l)
                    mapped |= edit.source[l] == id;
                if (!mapped)
                    return false;
            }
        }
    }
    if (junction)
        for (unsigned direction = 0; direction < 2; ++direction) {
            auto a = direction ? v : u, b = direction ? u : v;
            for (uint32_t k = t.face_offsets[a]; k < t.face_offsets[a + 1]; ++k) {
                auto mat = material(m, t.faces[k]);
                bool found = false;
                for (uint32_t l = t.face_offsets[b]; l < t.face_offsets[b + 1]; ++l)
                    found |= mat == material(m, t.faces[l]);
                if (!found)
                    return false;
            }
        }
    for (uint32_t k = t.face_offsets[u]; k < t.face_offsets[u + 1]; ++k) {
        uint32_t f = t.faces[k], old[3], next[3], sorted[3];
        bool removed = false;
        for (unsigned j = 0; j < 3; ++j) {
            old[j] = m.indices[f * 3 + j];
            next[j] = map_vertex(edit, old[j]);
            sorted[j] = geometry[next[j]];
            removed |= geometry[old[j]] == v;
        }
        if (removed)
            continue;
        triple(sorted);
        if (sorted[0] == sorted[1] || sorted[1] == sorted[2])
            return false;
        for (unsigned direction = 0; direction < 2; ++direction) {
            auto a = direction ? u : v;
            for (uint32_t l = t.face_offsets[a]; l < (direction ? k : t.face_offsets[a + 1]); ++l) {
                auto g = t.faces[l];
                uint32_t other[3];
                bool skip = false;
                for (unsigned j = 0; j < 3; ++j) {
                    auto id = m.indices[g * 3 + j];
                    skip |= direction && geometry[id] == v;
                    other[j] = geometry[direction ? map_vertex(edit, id) : id];
                }
                triple(other);
                if (!skip && face_equal(sorted, other))
                    return false;
            }
        }
        if (placement)
            continue;
        auto before = cross3(sub3(m.positions[old[1]], m.positions[old[0]]),
                             sub3(m.positions[old[2]], m.positions[old[0]]));
        auto after = cross3(sub3(m.positions[next[1]], m.positions[next[0]]),
                            sub3(m.positions[next[2]], m.positions[next[0]]));
        if (len3(after) <= 1e-15 || dot3(before, after) <= .05 * len3(before) * len3(after))
            return false;
        if (preserve_uv && m.uv) {
            double x = area2(m.uv[old[0]], m.uv[old[1]], m.uv[old[2]]),
                   y = area2(m.uv[next[0]], m.uv[next[1]], m.uv[next[2]]);
            if (fabs(x) > 1e-20 && x * y <= 0)
                return false;
        }
    }
    edit.removed = uint8_t(edge_faces);
    return true;
}
__global__ void enumerate(DeviceMeshView m, State* state, TopologyView t, const uint32_t* geometry,
                          Edit* edits, uint8_t* valid, uint32_t capacity, bool placement,
                          bool preserve_uv) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= capacity)
        return;
    valid[i] = 0;
    edits[i] = {};
    auto edge = i / 2;
    if (edge >= *t.length || t.edges[edge] == UINT64_MAX)
        return;
    auto a = uint32_t(t.edges[edge] >> 32), b = uint32_t(t.edges[edge]);
    Edit edit{};
    edit.from = i % 2 ? b : a;
    edit.to = i % 2 ? a : b;
    edit.valid = legal(m, state, t, geometry, edit, placement, preserve_uv);
    edits[i] = edit;
    valid[i] = edit.valid;
}
__global__ void fill_action_features(DeviceMeshView m, const State* state, TopologyView t,
                                     const Edit* edits, const float* vertices,
                                     const float* condition, Bounds b, uint32_t original_faces,
                                     float* rows, uint32_t capacity, uint32_t width,
                                     const uint32_t* selection, uint32_t first,
                                     uint32_t architecture, bool preserve_uv) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= capacity)
        return;
    auto* x = rows + size_t(i) * width;
    for (unsigned j = 0; j < width; ++j)
        x[j] = 0;
    uint32_t action = selection ? selection[i] : first + i;
    if (action >= state->actions)
        return;
    auto a = edits[action];
    uint32_t u = a.from, v = a.to, s = a.source[0], z = a.target[0];
    double scale = b.radius * 2;
    for (unsigned j = 0; j < 24; ++j) {
        x[j] = vertices[size_t(s) * 24 + j];
        x[24 + j] = vertices[size_t(z) * 24 + j];
    }
    for (unsigned j = 0; j < 8; ++j)
        x[48 + j] = condition[j];
    auto d = mul3(sub3(m.positions[v], m.positions[u]), 1 / scale);
    x[56] = d.x;
    x[57] = d.y;
    x[58] = d.z;
    x[59] = float(len3(d));
    for (unsigned j = 0; j < 3; ++j)
        x[60] += x[3 + j] * x[27 + j];
    if (m.uv) {
        x[61] = fminf(16, fmaxf(-16, m.uv[z].x - m.uv[s].x)) / 16;
        x[62] = fminf(16, fmaxf(-16, m.uv[z].y - m.uv[s].y)) / 16;
    }
    x[63] = float(t.offsets[u + 1] - t.offsets[u]) / 64;
    x[64] = float(t.offsets[v + 1] - t.offsets[v]) / 64;
    x[65] = a.removed / 2.f;
    x[66] = float(double(state->faces) / original_faces);
    x[67] = boundary(t, u);
    x[68] = boundary(t, v);
    x[69] = a.wedges / 8.f;
    x[70] = float(t.face_offsets[u + 1] - t.face_offsets[u]) / 64;
    x[71] = float(t.face_offsets[v + 1] - t.face_offsets[v]) / 64;
    Vec3 mean{}, target_mean{};
    double minimum = 1, maximum = 0;
    for (unsigned side = 0; side < 2; ++side) {
        auto vertex = side ? v : u;
        Vec3 sum{};
        for (uint32_t k = t.face_offsets[vertex]; k < t.face_offsets[vertex + 1]; ++k) {
            auto f = t.faces[k];
            auto p = m.positions[m.indices[f * 3]], q = m.positions[m.indices[f * 3 + 1]],
                 r = m.positions[m.indices[f * 3 + 2]];
            sum = add3(sum, norm3(cross3(sub3(q, p), sub3(r, p))));
        }
        if (side)
            target_mean = norm3(sum);
        else
            mean = norm3(sum);
    }
    for (uint32_t k = t.face_offsets[u]; k < t.face_offsets[u + 1]; ++k) {
        auto f = t.faces[k];
        auto p = m.positions[m.indices[f * 3]], q = m.positions[m.indices[f * 3 + 1]],
             r = m.positions[m.indices[f * 3 + 2]];
        auto normal = cross3(sub3(q, p), sub3(r, p));
        minimum = fmin(minimum, dot3(mean, norm3(normal)));
        maximum = fmax(maximum, len3(normal) / (scale * scale));
    }
    x[72] = float(minimum);
    x[73] = float(dot3(mean, target_mean));
    x[74] = float(maximum);
    x[75] = a.wedges > 1;
    x[76] = bool(m.materials);
    if (m.colors) {
        auto a = m.colors[s], b = m.colors[z];
        x[77] = float(len3(
            {a.r / 255.f - b.r / 255.f, a.g / 255.f - b.g / 255.f, a.b / 255.f - b.b / 255.f}));
    }
    x[78] = vertices[size_t(s) * 24 + 23];
    x[79] = architecture == conditioned_placement_schema ? float(preserve_uv)
                                                         : vertices[size_t(z) * 24 + 23];
    if (width == placement_features && a.wedges == 2)
        for (unsigned j = 0; j < 24; ++j) {
            x[80 + j] = vertices[size_t(a.source[1]) * 24 + j];
            x[104 + j] = vertices[size_t(a.target[1]) * 24 + j];
        }
}
__global__ void single_edit(State* state, const Edit* edits, uint32_t from, uint32_t to,
                            uint32_t revision, uint32_t* selected) {
    uint32_t match = none;
    if (revision == state->revision)
        for (uint32_t i = threadIdx.x; i < state->actions; i += blockDim.x)
            if (edits[i].from == from && edits[i].to == to) {
                match = i;
                break;
            }
    __shared__ cub::BlockReduce<uint32_t, 256>::TempStorage storage;
    auto found = cub::BlockReduce<uint32_t, 256>(storage).Reduce(match, cub::Min());
    if (threadIdx.x == 0) {
        state->error = found == none;
        state->selected = found != none;
        selected[0] = found;
    }
}
__global__ void apply_map(const Edit* edits, const uint32_t* selected, const State* state,
                          uint32_t* remap) {
    uint32_t i = threadIdx.x;
    if (i >= state->selected)
        return;
    auto e = edits[selected[i]];
    for (unsigned j = 0; j < e.wedges; ++j)
        remap[e.source[j]] = e.target[j];
}
__global__ void keep_faces(DeviceMeshView m, const State* state, const uint32_t* geometry,
                           const uint32_t* remap, uint32_t* keep, uint32_t capacity) {
    uint32_t f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f > capacity)
        return;
    keep[f] = 0;
    if (f >= state->faces)
        return;
    uint32_t ids[3];
    bool changed = false;
    for (unsigned j = 0; j < 3; ++j) {
        auto i = m.indices[f * 3 + j];
        if (remap[i] != none) {
            i = remap[i];
            changed = true;
        }
        ids[j] = geometry[i];
    }
    keep[f] = !(changed && (ids[0] == ids[1] || ids[1] == ids[2] || ids[0] == ids[2]));
}
__global__ void copy_trial(DeviceMeshView m, const State* state, const uint32_t* remap,
                           const uint32_t* keep, const uint32_t* offsets, uint32_t* indices,
                           uint16_t* materials, uint32_t capacity) {
    uint32_t f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= capacity || f >= state->faces || !keep[f])
        return;
    auto out = offsets[f];
    for (unsigned j = 0; j < 3; ++j) {
        auto i = m.indices[f * 3 + j];
        indices[out * 3 + j] = remap[i] == none ? i : remap[i];
    }
    if (materials)
        materials[out] = m.materials[f];
}
__global__ void commit_faces(State* state, const uint32_t* offsets, uint32_t capacity) {
    state->faces = offsets[capacity];
    ++state->revision;
    state->accepted += state->selected;
    state->accepted_trial = 1;
    state->position = 0;
}
__global__ void copy_commit(uint32_t* indices, uint16_t* materials, const uint32_t* trial_indices,
                            const uint16_t* trial_materials, const uint32_t* offsets,
                            uint32_t capacity) {
    uint32_t f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= offsets[capacity])
        return;
    for (unsigned j = 0; j < 3; ++j)
        indices[f * 3 + j] = trial_indices[f * 3 + j];
    if (materials)
        materials[f] = trial_materials[f];
}
__global__ void rank_actions(DeviceMeshView m, State* state, TopologyView t, const Edit* edits,
                             const float* prediction, float* scores, uint32_t* order,
                             NeuralRanking ranking, double scale, uint32_t capacity,
                             uint32_t outputs) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= capacity)
        return;
    order[i] = i;
    scores[i] = -CUDART_INF_F;
    if (i >= state->actions)
        return;
    float score = 0;
    if (ranking == NeuralRanking::Learned || ranking == NeuralRanking::Shuffled)
        score = prediction[size_t(i) * outputs];
    else if (ranking == NeuralRanking::ShortestEdge) {
        auto e = edits[i];
        score = -float(len3(mul3(sub3(m.positions[e.to], m.positions[e.from]), 1 / scale)));
    } else if (ranking == NeuralRanking::CurrentPlane) {
        scale = fmax(1e-20, scale);
        auto e = edits[i];
        double cost = 0;
        for (uint32_t k = t.face_offsets[e.from]; k < t.face_offsets[e.from + 1]; ++k) {
            auto f = t.faces[k];
            auto p = m.positions[m.indices[f * 3]], q = m.positions[m.indices[f * 3 + 1]],
                 r = m.positions[m.indices[f * 3 + 2]];
            auto normal = cross3(sub3(q, p), sub3(r, p));
            double weight = len3(normal),
                   distance = dot3(norm3(normal), sub3(m.positions[e.to], p)) / scale;
            cost += distance * distance * weight / (scale * scale);
        }
        score = float(-cost);
    }
    if (!isfinite(score))
        atomicExch(&state->error, 2u);
    scores[i] = score;
}
__device__ uint32_t random32(uint64_t& x) {
    x += 0x9e3779b97f4a7c15ull;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return uint32_t((z ^ (z >> 31)) >> 32);
}
__global__ void shuffle_scores(State* state, float* scores, uint32_t seed) {
    uint64_t x = (uint64_t(seed) << 32) | state->revision;
    for (uint32_t n = state->actions; n > 1; --n) {
        uint32_t threshold = uint32_t(-n) % n, r;
        do {
            r = random32(x);
        } while (r < threshold);
        auto j = r % n;
        float a = scores[n - 1];
        scores[n - 1] = scores[j];
        scores[j] = a;
    }
}
__global__ void select_batch(State* state, TopologyView t, const Edit* edits, const uint32_t* order,
                             uint32_t* selected, uint8_t* used, uint32_t vertices, uint32_t target,
                             uint8_t maximum, uint32_t ranked_count) {
    state->selected = 0;
    state->accepted_trial = 0;
    for (uint32_t v = 0; v < vertices; ++v)
        used[v] = 0;
    uint32_t remaining = state->faces > target ? state->faces - target : 0;
    for (uint32_t k = state->position; k < ranked_count && remaining; ++k) {
        auto i = order[k];
        auto e = edits[i];
        if (e.removed > remaining)
            continue;
        bool overlaps = false;
        for (auto u : {e.from, e.to}) {
            overlaps |= used[u] != 0;
            for (uint32_t j = t.offsets[u]; j < t.offsets[u + 1]; ++j)
                overlaps |= used[t.neighbors[j]] != 0;
        }
        if (overlaps)
            continue;
        for (auto u : {e.from, e.to}) {
            used[u] = 1;
            for (uint32_t j = t.offsets[u]; j < t.offsets[u + 1]; ++j)
                used[t.neighbors[j]] = 1;
        }
        if (!state->selected)
            state->first = k;
        selected[state->selected++] = i;
        remaining -= e.removed;
        if (state->selected == maximum)
            break;
    }
    state->exhausted = !state->selected;
}
__global__ void record_rank(State* state) {
    state->ranked += state->actions;
    state->position = 0;
}
__global__ void verdict(State* state, bool accepted) {
    ++state->trials;
    if (!accepted) {
        state->rejected += state->selected;
        if (state->selected > 1)
            state->selected /= 2;
        else {
            state->selected = 0;
            state->position = state->first + 1;
        }
    }
}
__device__ bool finite3(Vec3 p) {
    return isfinite(p.x) && isfinite(p.y) && isfinite(p.z);
}
__device__ Vec3 barycentric(Vec3 p, Vec3 a, Vec3 b, Vec3 c) {
    auto ab = sub3(b, a), ac = sub3(c, a), ap = sub3(p, a);
    double d1 = dot3(ab, ap), d2 = dot3(ac, ap);
    if (d1 <= 0 && d2 <= 0)
        return {1, 0, 0};
    auto bp = sub3(p, b);
    double d3 = dot3(ab, bp), d4 = dot3(ac, bp);
    if (d3 >= 0 && d4 <= d3)
        return {0, 1, 0};
    double vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) {
        double v = d1 / (d1 - d3);
        return {float(1 - v), float(v), 0};
    }
    auto cp = sub3(p, c);
    double d5 = dot3(ab, cp), d6 = dot3(ac, cp);
    if (d6 >= 0 && d5 <= d6)
        return {0, 0, 1};
    double vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) {
        double w = d2 / (d2 - d6);
        return {float(1 - w), 0, float(w)};
    }
    double va = d3 * d6 - d5 * d4;
    if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
        double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return {0, float(1 - w), float(w)};
    }
    double sum = va + vb + vc;
    if (fabs(sum) < 1e-30)
        return {1, 0, 0};
    double v = vb / sum, w = vc / sum;
    return {float(1 - v - w), float(v), float(w)};
}
__device__ Vec3 blend3(Vec3 a, Vec3 b, Vec3 c, Vec3 w) {
    return add3(add3(mul3(a, w.x), mul3(b, w.y)), mul3(c, w.z));
}
__device__ uint8_t blend_byte(uint8_t a, uint8_t b, uint8_t c, Vec3 w) {
    return uint8_t(
        fmin(255., fmax(0., floor(double(a) * w.x + double(b) * w.y + double(c) * w.z + .5))));
}
// UV/color/tangent correspondence stays on the original incident chart. It is
// never obtained by welding coincident positions or by projecting across seams.
__global__ void place_vertices(DeviceMeshView current, DeviceMeshView original, const State* state,
                               const Edit* edits, const Placement* proposals,
                               const uint32_t* selected, const uint32_t* geometry,
                               const uint32_t* canonical, const uint32_t* original_offsets,
                               const uint32_t* original_faces, Vec3* positions, Vec3* normals,
                               Vec2* uv, ColorRGBA8* colors, Vec4* tangents, bool packed_uv,
                               uint32_t* invalid) {
    uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= current.vertices)
        return;
    positions[v] = current.positions[v];
    if (normals)
        normals[v] = current.normals[v];
    if (uv)
        uv[v] = current.uv[v];
    if (colors)
        colors[v] = current.colors[v];
    if (tangents)
        tangents[v] = current.tangents[v];
    for (unsigned k = 0; k < state->selected; ++k) {
        auto index = selected[k];
        auto edit = edits[index];
        if (geometry[v] != edit.to)
            continue;
        auto proposal = proposals[index];
        positions[v] = proposal.position;
        uint32_t roots[2] = {canonical[v], none};
        for (unsigned w = 0; w < edit.wedges; ++w)
            if (canonical[v] == canonical[edit.target[w]]) {
                roots[1] = canonical[edit.source[w]];
                if (normals) {
                    if (!finite3(proposal.normals[w]))
                        atomicExch(invalid, 1u);
                    normals[v] = norm3(proposal.normals[w]);
                }
            }
        double best = dot3(sub3(proposal.position, original.positions[v]),
                           sub3(proposal.position, original.positions[v]));
        uint32_t ids[3] = {v, v, v};
        Vec3 weights{1, 0, 0};
        for (unsigned side = 0; side < 2; ++side) {
            auto root = roots[side];
            if (root == none)
                continue;
            for (uint32_t j = original_offsets[root]; j < original_offsets[root + 1]; ++j) {
                auto f = original_faces[j];
                uint32_t a = original.indices[f * 3], b = original.indices[f * 3 + 1],
                         c = original.indices[f * 3 + 2];
                auto w = barycentric(proposal.position, original.positions[a],
                                     original.positions[b], original.positions[c]);
                auto delta =
                    sub3(proposal.position, blend3(original.positions[a], original.positions[b],
                                                   original.positions[c], w));
                double distance = dot3(delta, delta);
                if (distance < best) {
                    best = distance;
                    ids[0] = a;
                    ids[1] = b;
                    ids[2] = c;
                    weights = w;
                }
            }
        }
        if (uv) {
            auto a = original.uv[ids[0]], b = original.uv[ids[1]], c = original.uv[ids[2]];
            uv[v] = {
                float(double(a.x) * weights.x + double(b.x) * weights.y + double(c.x) * weights.z),
                float(double(a.y) * weights.x + double(b.y) * weights.y + double(c.y) * weights.z)};
        }
        if (colors) {
            auto a = original.colors[ids[0]], b = original.colors[ids[1]],
                 c = original.colors[ids[2]];
            colors[v] = {blend_byte(a.r, b.r, c.r, weights), blend_byte(a.g, b.g, c.g, weights),
                         blend_byte(a.b, b.b, c.b, weights), blend_byte(a.a, b.a, c.a, weights)};
        }
        if (tangents) {
            auto a = original.tangents[ids[0]], b = original.tangents[ids[1]],
                 c = original.tangents[ids[2]];
            auto t = blend3({a.x, a.y, a.z}, {b.x, b.y, b.z}, {c.x, c.y, c.z}, weights);
            if (normals) {
                t = sub3(t, mul3(normals[v], dot3(t, normals[v])));
                if (len3(t) < 1e-15)
                    t = cross3(normals[v],
                               fabsf(normals[v].x) < .8f ? Vec3{1, 0, 0} : Vec3{0, 1, 0});
            }
            t = norm3(t);
            tangents[v] = {t.x, t.y, t.z, original.tangents[v].w};
        }
        break;
    }
    // The packed streams retain discarded vertex slots until compaction. A
    // triangle-only check misses an invalid edit when all its faces disappear.
    // Validate the streams here, in the kernel that writes them, for both serial
    // and indirect trials. Rejected candidates never reach packing or commit.
    auto p = positions[v], lo = current.quant_low, e = current.quant_extent;
    if (!finite3(p) || (normals && !finite3(normals[v])) ||
        (current.fixed_quantization && (p.x < lo.x || p.y < lo.y || p.z < lo.z ||
                                        p.x > lo.x + e.x || p.y > lo.y + e.y || p.z > lo.z + e.z)))
        atomicExch(invalid, 1u);
    if (uv) {
        auto t = uv[v];
        if (!isfinite(t.x) || !isfinite(t.y) ||
            (packed_uv && (t.x < -8 || t.x > 8 || t.y < -8 || t.y > 8)))
            atomicExch(invalid, 1u);
    }
}
__global__ void trial_precision(DeviceMeshView m, const State* state, const Edit* edits,
                                const uint32_t* selected, const uint32_t* geometry, uint32_t* out,
                                bool transfer) {
    uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    bool exact = v < m.vertices && exact_position(m, v);
    if (transfer && v < m.vertices)
        for (unsigned k = 0; k < state->selected; ++k) {
            auto e = edits[selected[k]];
            if (geometry[v] == e.to)
                exact |= exact_edit(m, e);
        }
    uint32_t word = __ballot_sync(0xffffffffu, exact);
    if ((threadIdx.x & 31) == 0 && v < m.vertices)
        out[v / 32] = word;
}
__global__ void precision_references(const uint32_t* indices, const uint32_t* count,
                                     uint32_t capacity, uint32_t* used) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= capacity * 3 || i >= *count * 3)
        return;
    auto v = indices[i];
    atomicOr(used + v / 32, 1u << (v % 32));
}
__global__ void precision_cap(uint32_t* bits, const uint32_t* used, uint32_t words, uint16_t bps,
                              uint32_t* invalid) {
    uint64_t counts = 0;
    for (uint32_t i = threadIdx.x; i < words; i += blockDim.x) {
        bits[i] &= used[i];
        counts += (uint64_t(__popc(bits[i])) << 32) | uint32_t(__popc(used[i]));
    }
    __shared__ cub::BlockReduce<uint64_t, 256>::TempStorage temp;
    auto sum = cub::BlockReduce<uint64_t, 256>(temp).Sum(counts);
    if (threadIdx.x == 0 && (sum >> 32) * 10000 > uint64_t(uint32_t(sum)) * bps)
        atomicExch(invalid, 1u);
}
__global__ void validate_placement(DeviceMeshView current, const State* state,
                                   const uint32_t* remap, const uint32_t* keep,
                                   const Vec3* positions, const Vec2* uv, uint32_t* invalid,
                                   bool preserve_uv, const uint32_t* geometry, uint32_t* slots,
                                   uint32_t hash_capacity) {
    uint32_t f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= state->faces || !keep[f])
        return;
    uint32_t old[3], next[3];
    for (unsigned j = 0; j < 3; ++j) {
        old[j] = current.indices[f * 3 + j];
        next[j] = remap[old[j]] == none ? old[j] : remap[old[j]];
    }
    auto before = cross3(sub3(current.positions[old[1]], current.positions[old[0]]),
                         sub3(current.positions[old[2]], current.positions[old[0]]));
    auto after = cross3(sub3(positions[next[1]], positions[next[0]]),
                        sub3(positions[next[2]], positions[next[0]]));
    if (len3(after) <= 1e-15 || dot3(before, after) <= .05 * len3(before) * len3(after)) {
        atomicExch(invalid, 1u);
        return;
    }
    if (preserve_uv && uv) {
        double x = area2(current.uv[old[0]], current.uv[old[1]], current.uv[old[2]]),
               y = area2(uv[next[0]], uv[next[1]], uv[next[2]]);
        if (!isfinite(y) || (fabs(x) > 1e-20 && x * y <= 0))
            atomicExch(invalid, 1u);
    }
    // Geometry IDs also anchor the original attribute charts. A placement must
    // not silently merge two of these classes: rebuilding from its snapshot
    // would then weld them and discover a different topology. Hash only live
    // trial vertices; discarded source IDs and existing seam wedges are valid.
    for (auto v : next) {
        // Fold high FP32 bits before masking: integer-grid coordinates have
        // zero low mantissa bits and otherwise cluster in a single bucket.
        uint32_t hash = hash_position(positions[v]);
        hash ^= hash >> 16;
        hash *= 0x7feb352du;
        hash ^= hash >> 15;
        hash *= 0x846ca68bu;
        hash ^= hash >> 16;
        uint32_t at = hash & (hash_capacity - 1);
        for (;;) {
            uint32_t old = atomicCAS(slots + at, none, v);
            if (old == none)
                break;
            if (same_position(positions[v], positions[old])) {
                if (geometry[v] != geometry[old])
                    atomicExch(invalid, 1u);
                break;
            }
            at = (at + 1) & (hash_capacity - 1);
        }
    }
}
__global__ void decode_placements(DeviceMeshView m, State* state, const Edit* edits,
                                  const float* prediction, Placement* proposals,
                                  const uint32_t* selection = nullptr, uint32_t count = UINT32_MAX,
                                  bool supervise_normals = true) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= state->actions || i >= count)
        return;
    auto id = selection ? selection[i] : i;
    auto e = edits[id];
    auto* y = prediction + size_t(i) * placement_outputs;
    for (unsigned j = 0; j < placement_outputs; ++j)
        if (!isfinite(y[j])) {
            atomicExch(&state->error, 2u);
            return;
        }
    auto middle = mul3(add3(m.positions[e.from], m.positions[e.to]), .5);
    double length = len3(sub3(m.positions[e.from], m.positions[e.to]));
    Placement p{};
    p.position =
        add3(middle,
             mul3({fminf(1, fmaxf(-1, y[3])), fminf(1, fmaxf(-1, y[4])), fminf(1, fmaxf(-1, y[5]))},
                  length));
    if (!exact_edit(m, e))
        p.position = grid_position(m, p.position);
    if (m.normals)
        for (unsigned j = 0; j < e.wedges; ++j)
            p.normals[j] = supervise_normals
                               ? norm3(add3(norm3(m.normals[e.target[j]]),
                                            {y[6 + j * 3], y[7 + j * 3], y[8 + j * 3]}))
                               : norm3(m.normals[e.target[j]]);
    proposals[id] = p;
}
__global__ void default_placements(DeviceMeshView m, const State* state, const Edit* edits,
                                   Placement* proposals) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= state->actions)
        return;
    auto e = edits[i];
    Placement p{};
    p.position = m.positions[e.to];
    if (m.normals)
        for (unsigned j = 0; j < e.wedges; ++j)
            p.normals[j] = norm3(m.normals[e.target[j]]);
    proposals[i] = p;
}
__global__ void assign_placement(const uint32_t* selected, Placement* proposals, Placement p) {
    proposals[selected[0]] = p;
}
__global__ void capture_policy_placements(const uint32_t* selected, const Placement* proposals,
                                          GpuActionState::Proposal* output, uint32_t count) {
    if (threadIdx.x < count)
        output[threadIdx.x] = {proposals[selected[threadIdx.x]], {}, 0};
}
__global__ void assign_trial(const uint32_t* selected, Placement* proposals, Placement p,
                             DeviceTrialStatus* status, const uint32_t* faces) {
    proposals[selected[0]] = p;
    status->faces = *faces;
    status->invalid = 0;
}
__global__ void teacher_selection(const State* state, const uint32_t* order, uint32_t* selected,
                                  uint32_t count, uint32_t seed) {
    count = min(count, state->actions);
    uint64_t random = (uint64_t(seed) << 32) | state->revision;
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t pick;
        if (i < (count + 1) / 2)
            pick = order[i];
        else {
            bool duplicate;
            do {
                pick = uint32_t((uint64_t(random32(random)) * state->actions) >> 32);
                duplicate = false;
                for (uint32_t j = 0; j < i; ++j)
                    duplicate |= selected[j] == pick;
            } while (duplicate);
        }
        selected[i] = pick;
    }
}
__global__ void teacher_mixed_selection(const State* state, const uint32_t* ranked,
                                        uint32_t* selected, uint32_t count, uint32_t seed) {
    uint64_t random = (uint64_t(seed) << 32) | state->revision;
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t pick = none, kind = i % 4;
        if (kind < 3)
            for (uint32_t j = 0; j < count; ++j) {
                auto candidate = ranked[kind * 16 + j];
                bool used = false;
                for (uint32_t k = 0; k < i; ++k)
                    used |= selected[k] == candidate;
                if (!used) {
                    pick = candidate;
                    break;
                }
            }
        if (pick == none) {
            bool used;
            do {
                pick = uint32_t((uint64_t(random32(random)) * state->actions) >> 32);
                used = false;
                for (uint32_t k = 0; k < i; ++k)
                    used |= selected[k] == pick;
            } while (used);
        }
        selected[i] = pick;
    }
}
__global__ void teacher_records(const State* state, const Edit* edits, const uint32_t* selected,
                                Action* actions, uint32_t count) {
    uint32_t i = threadIdx.x;
    if (i >= count)
        return;
    auto id = selected[i];
    actions[i] = {edits[id].from, edits[id].to, state->revision};
}
// The sorted global scores are diagnostic; only collector-supported action IDs
// remain selectable. Read the complete order before overwriting its prefix.
__global__ void restrict_ranking(State* state, const Edit* edits, uint32_t* order,
                                 const float* scores, const Action* support, uint32_t count,
                                 uint32_t seed, SupportedRanking* result) {
    uint32_t selected[16], used = 0;
    const auto first = order[0];
    result->global = {edits[first].from, edits[first].to, state->revision};
    result->global_score = scores[0];
    result->seed = seed;
    result->pool = uint8_t(count);
    result->global_in_support = false;
    for (uint32_t i = 0; i < state->actions && used < count; ++i) {
        const auto id = order[i];
        for (uint32_t j = 0; j < count; ++j)
            if (support[j].revision == state->revision && support[j].from == edits[id].from &&
                support[j].to == edits[id].to) {
                selected[used++] = id;
                result->global_in_support |= i == 0;
                if (used == 1) {
                    result->supported = {edits[id].from, edits[id].to, state->revision};
                    result->supported_score = scores[i];
                }
                break;
            }
    }
    if (used != count)
        state->error = 1;
    else
        for (uint32_t i = 0; i < count; ++i)
            order[i] = selected[i];
}
__global__ void teacher_placements(DeviceMeshView m, TopologyView topology, const Edit* edits,
                                   const uint32_t* selected, GpuActionState::Proposal* proposals,
                                   const Placement* learned, bool supervise_normals) {
    uint32_t output = threadIdx.x, index = supervise_normals ? output : output * 2;
    if (index > 20 || (index == 20 && !learned))
        return;
    auto e = edits[selected[0]];
    unsigned position = index / 2, normal = index % 2;
    auto middle = mul3(add3(m.positions[e.from], m.positions[e.to]), .5);
    double scale = len3(sub3(m.positions[e.from], m.positions[e.to]));
    Placement p{};
    p.position = middle;
    if (position < 2)
        p.position = m.positions[position ? e.to : e.from];
    if (position == 3) {
        double matrix[3][4]{};
        for (unsigned side = 0; side < 2; ++side) {
            auto vertex = side ? e.to : e.from;
            for (uint32_t k = topology.face_offsets[vertex]; k < topology.face_offsets[vertex + 1];
                 ++k) {
                auto f = topology.faces[k];
                auto a = m.positions[m.indices[f * 3]], b = m.positions[m.indices[f * 3 + 1]],
                     c = m.positions[m.indices[f * 3 + 2]];
                auto cross = cross3(sub3(b, a), sub3(c, a));
                auto n = norm3(cross);
                double length = len3(cross), v[3] = {n.x, n.y, n.z}, d = dot3(n, sub3(a, middle));
                for (unsigned i = 0; i < 3; ++i) {
                    for (unsigned j = 0; j < 3; ++j)
                        matrix[i][j] += v[i] * v[j] * length;
                    matrix[i][3] += v[i] * d * length;
                }
            }
        }
        bool solved = true;
        for (unsigned column = 0; column < 3; ++column) {
            unsigned pivot = column;
            for (unsigned row = column + 1; row < 3; ++row)
                if (fabs(matrix[row][column]) > fabs(matrix[pivot][column]))
                    pivot = row;
            if (fabs(matrix[pivot][column]) < 1e-15) {
                solved = false;
                break;
            }
            for (unsigned j = column; j < 4; ++j) {
                double t = matrix[column][j];
                matrix[column][j] = matrix[pivot][j];
                matrix[pivot][j] = t;
            }
            double d = matrix[column][column];
            for (unsigned j = column; j < 4; ++j)
                matrix[column][j] /= d;
            for (unsigned row = 0; row < 3; ++row)
                if (row != column) {
                    double factor = matrix[row][column];
                    for (unsigned j = column; j < 4; ++j)
                        matrix[row][j] -= factor * matrix[column][j];
                }
        }
        if (solved)
            p.position = add3(middle, {float(fmin(scale, fmax(-scale, matrix[0][3]))),
                                       float(fmin(scale, fmax(-scale, matrix[1][3]))),
                                       float(fmin(scale, fmax(-scale, matrix[2][3])))});
    }
    if (position >= 4 && index < 20) {
        float delta[3]{};
        delta[(position - 4) / 2] = float(((position - 4) % 2 ? -.25 : .25) * scale);
        p.position = add3(middle, {delta[0], delta[1], delta[2]});
    }
    if (index == 20)
        p = learned[selected[0]];
    if (!exact_edit(m, e))
        p.position = grid_position(m, p.position);
    GpuActionState::Proposal out{};
    out.placement = p;
    auto offset = mul3(sub3(p.position, middle), 1 / scale);
    out.target[0] = fminf(1, fmaxf(-1, offset.x));
    out.target[1] = fminf(1, fmaxf(-1, offset.y));
    out.target[2] = fminf(1, fmaxf(-1, offset.z));
    if (m.normals)
        for (unsigned j = 0; j < e.wedges; ++j) {
            auto baseline = norm3(m.normals[e.target[j]]);
            auto n = !supervise_normals ? baseline
                     : index == 20      ? p.normals[j]
                     : normal           ? norm3(add3(baseline, norm3(m.normals[e.source[j]])))
                                        : baseline;
            if (len3(n) < .5)
                n = baseline;
            out.placement.normals[j] = n;
            auto d = sub3(n, baseline);
            out.target[3 + j * 3] = d.x;
            out.target[4 + j * 3] = d.y;
            out.target[5 + j * 3] = d.z;
            if (supervise_normals)
                out.normal_mask |= uint8_t(1u << j);
        }
    proposals[output] = out;
}
struct DeviceScope {
    int previous;
    explicit DeviceScope(int id) {
        check(cudaGetDevice(&previous));
        check(cudaSetDevice(id));
    }
    ~DeviceScope() {
        cudaSetDevice(previous);
    }
};
} // namespace
struct GpuActionState::Impl {
    Device device;
    int id;
    DeviceMesh mesh;
    Bounds bounds;
    uint32_t face_capacity, action_capacity, hash_capacity, source_faces, inference_batch;
    bool placement, packed_uv, preserve_uv;
    std::unique_ptr<DeviceMesh> original;
    Buffer<State> state;
    Buffer<uint32_t> geometry, canonical, slots, seams;
    Topology geometric, attribute;
    Buffer<float> vertex_x, conditions_buffer, features_buffer;
    Buffer<Edit> possible, edits;
    Buffer<uint8_t> valid;
    Buffer<uint32_t> selected, teacher_ranked, remap, keep, offsets, trial_indices,
        original_indices, order, sorted_order;
    Buffer<uint16_t> trial_materials, original_materials;
    Buffer<float> prediction, scores, sorted_scores;
    Buffer<uint8_t> used;
    Buffer<Placement> proposals;
    Buffer<GpuActionState::Proposal> teacher;
    Buffer<uint32_t> trial_bits, precision_used, original_bits;
    Buffer<Vec3> trial_positions, trial_normals;
    Buffer<Vec2> trial_uv;
    Buffer<ColorRGBA8> trial_colors;
    Buffer<Vec4> trial_tangents;
    struct TrialBuffers {
        Buffer<uint32_t> bits;
        Buffer<Vec3> positions, normals;
        Buffer<Vec2> uv;
        Buffer<ColorRGBA8> colors;
        Buffer<Vec4> tangents;
        Buffer<DeviceTrialStatus> status;
        TrialBuffers(Device& d, DeviceMeshView m)
            : bits(d, m.exact_position_bits ? (m.vertices + 31) / 32 : 0), positions(d, m.vertices),
              normals(d, m.normals ? m.vertices : 0), uv(d, m.uv ? m.vertices : 0),
              colors(d, m.colors ? m.vertices : 0), tangents(d, m.tangents ? m.vertices : 0),
              status(d, 1) {}
    };
    std::array<std::unique_ptr<TrialBuffers>, 8> trial_slots;
    Buffer<uint32_t> original_offsets, original_faces;
    std::unique_ptr<Buffer<std::byte>> select_temp, scan_temp, sort_temp;
    size_t select_bytes{}, scan_bytes{}, sort_bytes{};
    State host{};
    uint64_t trial_revision{};
    bool teacher_policy{}, selected_is_action{};
    Action selected_action{};
    static uint32_t hash_size(size_t vertices) {
        if (vertices > uint32_t(INT_MAX) / 2)
            throw std::length_error("GPU vertex hash exceeds u32");
        uint32_t n = 2;
        while (n < vertices * 2)
            n *= 2;
        return n;
    }
    Impl(MeshView m, const NeuralOptions& options, bool free, const VertexBounds* quantization,
         MeshView origin, uint32_t batch)
        : device(options, true), id(options.device),
          mesh(device, m, free && options.draw_storage() == NeuralVertexStorage::Packed,
               quantization),
          bounds(blitz::bounds(origin.positions.count ? origin : m)),
          face_capacity(uint32_t(m.triangles())), action_capacity(face_capacity * 6),
          hash_capacity(hash_size(m.positions.count)),
          source_faces(uint32_t(origin.positions.count ? origin.triangles() : m.triangles())),
          inference_batch(batch), placement(free),
          packed_uv(options.draw_storage() == NeuralVertexStorage::Packed),
          preserve_uv(options.preserve_uv), state(device, 1), geometry(device, m.positions.count),
          canonical(device, m.positions.count), slots(device, hash_capacity),
          seams(device, m.positions.count),
          geometric(device, uint32_t(m.positions.count), face_capacity),
          attribute(device, uint32_t(m.positions.count), face_capacity),
          vertex_x(device, m.positions.count * 24), conditions_buffer(device, conditions),
          possible(device, action_capacity), edits(device, action_capacity),
          valid(device, action_capacity), selected(device, 64), teacher_ranked(device, 48),
          remap(device, m.positions.count), keep(device, size_t(face_capacity) + 1),
          offsets(device, size_t(face_capacity) + 1), trial_indices(device, m.indices.size()),
          original_indices(device, m.indices.size()), order(device, action_capacity),
          sorted_order(device, action_capacity), trial_materials(device, m.materials.size()),
          original_materials(device, m.materials.size()), scores(device, action_capacity),
          sorted_scores(device, action_capacity), used(device, m.positions.count),
          proposals(device, free ? action_capacity : 0), teacher(device, free ? 21 : 0),
          trial_bits(device, m.exact_position_bits.size()),
          precision_used(device, m.exact_position_bits.size()),
          original_bits(device, m.exact_position_bits.size()),
          trial_positions(device, free ? m.positions.count : 0),
          trial_normals(device, free ? m.normals.count : 0),
          trial_uv(device, free ? m.uv.count : 0), trial_colors(device, free ? m.colors.count : 0),
          trial_tangents(device, free ? m.tangents.count : 0),
          original_offsets(device, free ? m.positions.count + 1 : 0),
          original_faces(device, free ? m.indices.size() : 0) {
        if (options.draw_storage() != NeuralVertexStorage::Float32) {
            auto q = quantization ? *quantization : vertex_bounds(m);
            mesh.view.quant_low = q.low;
            mesh.view.quant_extent = q.extent;
            mesh.view.fixed_quantization = true;
        }
        if (options.exact_position_bps > 10000)
            throw std::invalid_argument("exact position cap exceeds 100%");
        mesh.view.exact_position_bps = options.exact_position_bps;
        original_bits.upload(m.exact_position_bits);
        host.faces = face_capacity;
        state.upload({&host, 1});
        check(gpu::copy(original_indices.p, mesh.indices.p, original_indices.n * sizeof(uint32_t),
                        cudaMemcpyDeviceToDevice));
        if (original_materials.n)
            check(gpu::copy(original_materials.p, mesh.materials.p,
                            original_materials.n * sizeof(uint16_t), cudaMemcpyDeviceToDevice));
        for (bool attrs : {false, true}) {
            check(gpu::memset(slots.p, 255, slots.n * sizeof(uint32_t)));
            insert_vertices<<<blocks(mesh.view.vertices), 256, 0, gpu::stream()>>>(
                mesh.view, slots.p, hash_capacity, attrs, preserve_uv);
            lookup_vertices<<<blocks(mesh.view.vertices), 256, 0, gpu::stream()>>>(
                mesh.view, slots.p, hash_capacity, attrs ? canonical.p : geometry.p, attrs,
                preserve_uv);
        }
        if (!preserve_uv) {
            weld_uv_indices<<<blocks(mesh.indices.n), 256, 0, gpu::stream()>>>(
                mesh.indices.p, canonical.p, uint32_t(mesh.indices.n));
            check(gpu::copy(original_indices.p, mesh.indices.p, mesh.indices.n * sizeof(uint32_t),
                            cudaMemcpyDeviceToDevice));
        }
        if (mesh.precision.n) {
            // Validate the effective initial representation too, including a
            // resumed episode that never proposes an edit. UV welding above
            // may change its referenced-ID denominator. Use scratch bits so
            // validation does not modify the admitted mesh or reset snapshot.
            precision_used.zero();
            check(gpu::copy(trial_bits.p, mesh.precision.p, trial_bits.n * sizeof(uint32_t),
                            cudaMemcpyDeviceToDevice));
            precision_references<<<blocks(mesh.indices.n), 256, 0, gpu::stream()>>>(
                mesh.indices.p, &state.p->faces, face_capacity, precision_used.p);
            precision_cap<<<1, 256, 0, gpu::stream()>>>(
                trial_bits.p, precision_used.p, uint32_t(trial_bits.n),
                mesh.view.exact_position_bps, &state.p->invalid_placement);
            read();
            if (host.invalid_placement)
                throw std::invalid_argument("initial mesh exceeds exact position cap");
        }
        seams.zero();
        mark_seams<<<blocks(mesh.view.vertices), 256, 0, gpu::stream()>>>(mesh.view, geometry.p,
                                                                          seams.p, preserve_uv);
        check(cub::DeviceSelect::Flagged(nullptr, select_bytes, possible.p, valid.p, edits.p,
                                         &state.p->actions, int(action_capacity), gpu::stream()));
        check(cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, keep.p, offsets.p,
                                            int(face_capacity + 1), gpu::stream()));
        check(cub::DeviceRadixSort::SortPairsDescending(
            nullptr, sort_bytes, scores.p, sorted_scores.p, order.p, sorted_order.p,
            int(action_capacity), 0, 32, gpu::stream()));
        select_temp = std::make_unique<Buffer<std::byte>>(device, select_bytes);
        scan_temp = std::make_unique<Buffer<std::byte>>(device, scan_bytes);
        sort_temp = std::make_unique<Buffer<std::byte>>(device, sort_bytes);
        rebuild();
        if (placement) {
            original = std::make_unique<DeviceMesh>(device, mesh);
            check(gpu::copy(original_offsets.p, attribute.face_offsets.p,
                            original_offsets.n * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
            check(gpu::copy(original_faces.p, attribute.faces.p,
                            original_faces.n * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
        }
        check(cudaGetLastError());
        check(cudaSetDevice(device.previous));
    }
    ~Impl() {
        cudaSetDevice(id);
    }
    void topology(Topology& t, const uint32_t* ids) {
        t.offsets.zero();
        t.face_offsets.zero();
        t.invalid.zero();
        edge_keys<<<blocks(t.entries), 256, 0, gpu::stream()>>>(
            mesh.view, state.p, ids, t.raw.p, t.face_offsets.p, t.invalid.p, t.entries);
        check(cub::DeviceRadixSort::SortKeys(t.temp->p, t.sort_bytes, t.raw.p, t.sorted.p,
                                             int(t.entries), 0, 64, gpu::stream()));
        check(cub::DeviceRunLengthEncode::Encode(t.temp->p, t.encode_bytes, t.sorted.p, t.edges.p,
                                                 t.counts.p, t.length.p, int(t.entries),
                                                 gpu::stream()));
        edge_degrees<<<blocks(t.entries), 256, 0, gpu::stream()>>>(
            t.edges.p, t.counts.p, t.length.p, t.offsets.p, t.invalid.p, t.entries);
        check(cub::DeviceScan::InclusiveSum(t.temp->p, t.scan_bytes, t.offsets.p, t.offsets.p,
                                            int(t.vertices + 1), gpu::stream()));
        check(cub::DeviceScan::InclusiveSum(t.temp->p, t.scan_bytes, t.face_offsets.p,
                                            t.face_offsets.p, int(t.vertices + 1), gpu::stream()));
        check(gpu::copy(t.next.p, t.offsets.p, t.next.n * sizeof(uint32_t),
                        cudaMemcpyDeviceToDevice));
        scatter_edges<<<blocks(t.entries), 256, 0, gpu::stream()>>>(t.view(), t.next.p,
                                                                    t.neighbors.p, t.entries);
        check(gpu::copy(t.next.p, t.face_offsets.p, t.next.n * sizeof(uint32_t),
                        cudaMemcpyDeviceToDevice));
        scatter_faces<<<blocks(t.entries), 256, 0, gpu::stream()>>>(mesh.view, state.p, ids,
                                                                    t.next.p, t.faces.p, t.entries);
        sort_lists<<<blocks(t.vertices), 256, 0, gpu::stream()>>>(t.offsets.p, t.neighbors.p,
                                                                  t.vertices);
        sort_lists<<<blocks(t.vertices), 256, 0, gpu::stream()>>>(t.face_offsets.p, t.faces.p,
                                                                  t.vertices);
    }
    void rebuild() {
        teacher_policy = false;
        selected_is_action = false;
        topology(geometric, geometry.p);
        topology(attribute, canonical.p);
        connected_fans<<<blocks(mesh.view.vertices), 256, 0, gpu::stream()>>>(
            mesh.view, geometric.view(), geometry.p, geometric.invalid.p, geometric.visited.p);
        vertex_features<<<blocks(mesh.view.vertices), 256, 0, gpu::stream()>>>(
            mesh.view, attribute.view(), canonical.p, geometry.p, seams.p, bounds, vertex_x.p);
        enumerate<<<blocks(action_capacity), 256, 0, gpu::stream()>>>(
            mesh.view, state.p, geometric.view(), geometry.p, possible.p, valid.p, action_capacity,
            placement, preserve_uv);
        check(cub::DeviceSelect::Flagged(select_temp->p, select_bytes, possible.p, valid.p, edits.p,
                                         &state.p->actions, int(action_capacity), gpu::stream()));
        check(cudaGetLastError());
    }
    void ensure(Buffer<float>& buffer, size_t size) {
        if (buffer.n < size) {
            buffer = {};
            buffer = Buffer<float>(device, size);
        }
    }
    void features(const std::array<float, conditions>& c, uint32_t width = action_features,
                  const uint32_t* selection = nullptr, uint32_t count = 0,
                  uint32_t architecture = action_schema) {
        for (float f : c)
            if (!std::isfinite(f))
                throw std::invalid_argument("nonfinite action condition");
        conditions_buffer.upload(c);
        uint32_t rows = selection ? count : action_capacity;
        ensure(features_buffer, size_t(rows) * width);
        fill_action_features<<<blocks(rows), 256, 0, gpu::stream()>>>(
            mesh.view, state.p, geometric.view(), edits.p, vertex_x.p, conditions_buffer.p, bounds,
            source_faces, features_buffer.p, rows, width, selection, 0, architecture, preserve_uv);
        check(cudaGetLastError());
    }
    void read() {
        host = state.download()[0];
        if (host.error) {
            check(gpu::memset(&state.p->error, 0, sizeof(uint32_t)));
            if (host.error == 2)
                throw std::invalid_argument("nonfinite GPU policy prediction");
            throw std::invalid_argument("illegal or stale GPU action");
        }
        mesh.view.faces = host.faces;
    }
    void select(Action a) {
        if (selected_is_action && a == selected_action)
            return;
        selected_is_action = false;
        single_edit<<<1, 256, 0, gpu::stream()>>>(state.p, edits.p, a.from, a.to, a.revision,
                                                  selected.p);
        read();
        selected_action = a;
        selected_is_action = true;
    }
    void trial(Action a) {
        select(a);
        if (placement)
            default_placements<<<blocks(action_capacity), 256, 0, gpu::stream()>>>(
                mesh.view, state.p, edits.p, proposals.p);
        build_trial();
    }
    void build_trial_topology() {
        check(gpu::memset(remap.p, 255, remap.n * sizeof(uint32_t)));
        apply_map<<<1, 64, 0, gpu::stream()>>>(edits.p, selected.p, state.p, remap.p);
        keep_faces<<<blocks(size_t(face_capacity) + 1), 256, 0, gpu::stream()>>>(
            mesh.view, state.p, geometry.p, remap.p, keep.p, face_capacity);
        check(cub::DeviceScan::ExclusiveSum(scan_temp->p, scan_bytes, keep.p, offsets.p,
                                            int(face_capacity + 1), gpu::stream()));
        copy_trial<<<blocks(face_capacity), 256, 0, gpu::stream()>>>(
            mesh.view, state.p, remap.p, keep.p, offsets.p, trial_indices.p, trial_materials.p,
            face_capacity);
    }
    void build_precision(Buffer<uint32_t>& bits, uint32_t* invalid) {
        if (!bits.n)
            return;
        precision_used.zero();
        trial_precision<<<blocks(mesh.view.vertices), 256, 0, gpu::stream()>>>(
            mesh.view, state.p, edits.p, selected.p, geometry.p, bits.p, placement);
        precision_references<<<blocks(size_t(face_capacity) * 3), 256, 0, gpu::stream()>>>(
            trial_indices.p, offsets.p + face_capacity, face_capacity, precision_used.p);
        precision_cap<<<1, 256, 0, gpu::stream()>>>(bits.p, precision_used.p, uint32_t(bits.n),
                                                    mesh.view.exact_position_bps, invalid);
    }
    void build_trial() {
        build_trial_topology();
        check(gpu::memset(&state.p->invalid_placement, 0, sizeof(uint32_t)));
        if (placement) {
            place_vertices<<<blocks(mesh.view.vertices), 256, 0, gpu::stream()>>>(
                mesh.view, original->view, state.p, edits.p, proposals.p, selected.p, geometry.p,
                canonical.p, original_offsets.p, original_faces.p, trial_positions.p,
                trial_normals.p, trial_uv.p, trial_colors.p, trial_tangents.p, packed_uv,
                &state.p->invalid_placement);
            check(gpu::memset(slots.p, 255, slots.n * sizeof(uint32_t)));
            validate_placement<<<blocks(face_capacity), 256, 0, gpu::stream()>>>(
                mesh.view, state.p, remap.p, keep.p, trial_positions.p, trial_uv.p,
                &state.p->invalid_placement, preserve_uv, geometry.p, slots.p, hash_capacity);
        }
        build_precision(trial_bits, &state.p->invalid_placement);
        check(cudaGetLastError());
    }
    Lod download_trial() {
        uint32_t n = 0;
        check(gpu::copy(&n, offsets.p + face_capacity, sizeof(n), cudaMemcpyDeviceToHost));
        if (!n)
            throw std::invalid_argument("GPU action removed the entire mesh");
        Lod lod;
        lod.data.indices.resize(size_t(n) * 3);
        check(gpu::copy(lod.data.indices.data(), trial_indices.p,
                        lod.data.indices.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        if (trial_materials.n) {
            lod.data.materials.resize(n);
            check(gpu::copy(lod.data.materials.data(), trial_materials.p, n * sizeof(uint16_t),
                            cudaMemcpyDeviceToHost));
        }
        return lod;
    }
    template <class T> void copy(Buffer<T>& to, const Buffer<T>& from) {
        if (to.n)
            check(gpu::copy(to.p, from.p, to.n * sizeof(T), cudaMemcpyDeviceToDevice));
    }
    void commit() {
        if (placement || mesh.precision.n) {
            read();
            if (host.invalid_placement)
                throw std::invalid_argument("invalid placement or exact position cap");
        }
        if (placement) {
            copy(mesh.positions, trial_positions);
            copy(mesh.normals, trial_normals);
            copy(mesh.uv, trial_uv);
            copy(mesh.colors, trial_colors);
            copy(mesh.tangents, trial_tangents);
        }
        copy(mesh.precision, trial_bits);
        copy_commit<<<blocks(face_capacity), 256, 0, gpu::stream()>>>(
            mesh.indices.p, mesh.materials.p, trial_indices.p, trial_materials.p, offsets.p,
            face_capacity);
        commit_faces<<<1, 1, 0, gpu::stream()>>>(state.p, offsets.p, face_capacity);
        read();
        mesh.view.revision = ++trial_revision;
        rebuild();
    }
    void rank(const std::array<float, conditions>& c, ActionCuda* network, NeuralRanking ranking,
              uint32_t seed) {
        selected_is_action = false;
        auto architecture = network ? network->architecture() : action_schema;
        auto width = policy_inputs(architecture), outputs = policy_outputs(architecture);
        for (float f : c)
            if (!std::isfinite(f))
                throw std::invalid_argument("nonfinite action condition");
        read();
        if (ranking == NeuralRanking::Learned || ranking == NeuralRanking::Shuffled ||
            (placement && is_placement_schema(architecture))) {
            if (!network)
                throw std::invalid_argument("missing GPU action network");
            ensure(prediction, size_t(host.actions) * outputs);
            // Features are transient policy inputs. Keep only one bounded tile,
            // while scores and placements retain their stable global action IDs.
            conditions_buffer.upload(c);
            ensure(features_buffer, size_t(std::min(host.actions, inference_batch)) * width);
            for (uint32_t first = 0; first < host.actions; first += inference_batch) {
                auto count = std::min(inference_batch, host.actions - first);
                fill_action_features<<<blocks(count), 256, 0, gpu::stream()>>>(
                    mesh.view, state.p, geometric.view(), edits.p, vertex_x.p, conditions_buffer.p,
                    bounds, source_faces, features_buffer.p, count, width, nullptr, first,
                    architecture, preserve_uv);
                network->predict_device(features_buffer.p, prediction.p + size_t(first) * outputs,
                                        count);
            }
        }
        if (placement) {
            if (is_placement_schema(architecture))
                decode_placements<<<blocks(action_capacity), 256, 0, gpu::stream()>>>(
                    mesh.view, state.p, edits.p, prediction.p, proposals.p, nullptr, UINT32_MAX,
                    c[3] != 0 || c[4] != 0 || c[5] != 0);
            else
                default_placements<<<blocks(action_capacity), 256, 0, gpu::stream()>>>(
                    mesh.view, state.p, edits.p, proposals.p);
        }
        rank_actions<<<blocks(action_capacity), 256, 0, gpu::stream()>>>(
            mesh.view, state.p, geometric.view(), edits.p, prediction.p, scores.p, order.p, ranking,
            bounds.diameter(), action_capacity, outputs);
        if (ranking == NeuralRanking::Shuffled)
            shuffle_scores<<<1, 1, 0, gpu::stream()>>>(state.p, scores.p, seed);
        check(cub::DeviceRadixSort::SortPairsDescending(
            sort_temp->p, sort_bytes, scores.p, sorted_scores.p, order.p, sorted_order.p,
            int(action_capacity), 0, 32, gpu::stream()));
        record_rank<<<1, 1, 0, gpu::stream()>>>(state.p);
        read();
    }
    DeviceMeshView trial_view() {
        uint32_t count;
        check(gpu::copy(&count, offsets.p + face_capacity, sizeof(count), cudaMemcpyDeviceToHost));
        auto v = mesh.view;
        v.indices = trial_indices.p;
        v.materials = trial_materials.p;
        v.faces = count;
        v.revision = ++trial_revision;
        if (placement) {
            v.positions = trial_positions.p;
            v.normals = trial_normals.p;
            v.uv = trial_uv.p;
            v.colors = trial_colors.p;
            v.tangents = trial_tangents.p;
        }
        v.exact_position_bits = trial_bits.p;
        v.raster_parent = &mesh.view;
        v.parent_keep = keep.p;
        v.parent_offsets = offsets.p;
        return v;
    }
    Lod download() {
        read();
        Lod lod;
        lod.data.indices.resize(size_t(host.faces) * 3);
        check(gpu::copy(lod.data.indices.data(), mesh.indices.p,
                        lod.data.indices.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        if (mesh.materials.n) {
            lod.data.materials.resize(host.faces);
            check(gpu::copy(lod.data.materials.data(), mesh.materials.p,
                            host.faces * sizeof(uint16_t), cudaMemcpyDeviceToHost));
        }
        if (placement) {
            lod.shared_vertices = false;
            lod.data.positions = mesh.positions.download();
            lod.data.normals = mesh.normals.download();
            lod.data.uv = mesh.uv.download();
            lod.data.colors = mesh.colors.download();
            lod.data.tangents = mesh.tangents.download();
            lod.data.double_sided = mesh.sided.download();
            lod.data.exact_position_bits = mesh.precision.download();
            compact(lod.data);
        }
        return lod;
    }
};
GpuActionState::GpuActionState(MeshView m, const NeuralOptions& options, bool placement,
                               const VertexBounds* quantization, MeshView origin,
                               uint32_t inference_batch) {
    if (!inference_batch || inference_batch > 65536)
        throw std::invalid_argument("inference batch outside 1..65536");
    if (auto e = validate(m); !e.empty())
        throw std::invalid_argument(e);
    if (m.triangles() > uint32_t(INT_MAX) / 6)
        throw std::length_error("GPU action capacity exceeds signed sort range");
    if (origin.positions.count && (!origin.triangles() || origin.triangles() > UINT32_MAX))
        throw std::invalid_argument("episode origin triangle domain");
    VertexBounds domain;
    if (options.draw_storage() != NeuralVertexStorage::Float32) {
        domain = quantization ? *quantization : vertex_bounds(m);
        if (!valid_vertex_bounds(domain))
            throw std::invalid_argument("invalid working mesh bounds");
        quantization = &domain;
    }
    impl_ = std::make_unique<Impl>(m, options, placement, quantization, origin, inference_batch);
}
GpuActionState::~GpuActionState() {
    if (impl_) {
        int previous = 0;
        cudaGetDevice(&previous);
        impl_.reset();
        cudaSetDevice(previous);
    }
}
std::vector<ActionRecord> GpuActionState::actions(const std::array<float, conditions>& c) {
    auto& p = *impl_;
    DeviceScope scope(p.id);
    p.features(c);
    p.read();
    std::vector<Edit> edits(p.host.actions);
    std::vector<float> x(size_t(p.host.actions) * action_features);
    std::vector<ActionRecord> out(p.host.actions);
    if (!out.empty()) {
        check(gpu::copy(edits.data(), p.edits.p, edits.size() * sizeof(Edit),
                        cudaMemcpyDeviceToHost));
        check(gpu::copy(x.data(), p.features_buffer.p, x.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
    }
    for (size_t i = 0; i < out.size(); ++i) {
        out[i].action = {edits[i].from, edits[i].to, p.host.revision};
        std::copy_n(x.data() + i * action_features, action_features, out[i].x.data());
    }
    return out;
}
Lod GpuActionState::trial(Action a) {
    auto& p = *impl_;
    DeviceScope scope(p.id);
    if (p.placement)
        throw std::logic_error("explicit placement required");
    p.trial(a);
    return p.download_trial();
}
bool GpuActionState::trial(Action a, DeviceMeshView& candidate) {
    auto& p = *impl_;
    DeviceScope scope(p.id);
    if (p.placement)
        throw std::logic_error("endpoint trial requires a reuse state");
    p.trial(a);
    p.read();
    if (p.host.invalid_placement)
        return false;
    candidate = p.trial_view();
    return true;
}
void GpuActionState::commit(Action a) {
    auto& p = *impl_;
    DeviceScope scope(p.id);
    if (p.placement)
        throw std::logic_error("explicit placement required");
    p.trial(a);
    p.commit();
}
DeviceMeshView GpuActionState::view() const {
    return impl_->mesh.view;
}
void GpuActionState::reset() {
    auto& p = *impl_;
    DeviceScope scope(p.id);
    p.host = {};
    p.host.faces = p.face_capacity;
    p.state.upload({&p.host, 1});
    p.mesh.view.faces = p.host.faces;
    p.mesh.view.revision = ++p.trial_revision;
    p.copy(p.mesh.precision, p.original_bits);
    if (p.placement) {
        p.copy(p.mesh.positions, p.original->positions);
        p.copy(p.mesh.normals, p.original->normals);
        p.copy(p.mesh.uv, p.original->uv);
        p.copy(p.mesh.colors, p.original->colors);
        p.copy(p.mesh.tangents, p.original->tangents);
    }
    check(gpu::copy(p.mesh.indices.p, p.original_indices.p, p.original_indices.n * sizeof(uint32_t),
                    cudaMemcpyDeviceToDevice));
    if (p.original_materials.n)
        check(gpu::copy(p.mesh.materials.p, p.original_materials.p,
                        p.original_materials.n * sizeof(uint16_t), cudaMemcpyDeviceToDevice));
    p.rebuild();
}
Lod GpuActionState::execute(const std::array<float, conditions>& c, size_t target, uint32_t budget,
                            ActionCuda* network, NeuralRanking ranking, uint32_t seed,
                            uint8_t batch, const std::function<bool(DeviceMeshView)>& gate,
                            ActionStats* statistics, const std::function<bool()>& cancelled,
                            const std::function<void(GpuActionState&)>& observe,
                            const ActionTrialObserver& trace, const RankingSupport* support) {
    if (!batch || batch > 64 || !gate || ranking > NeuralRanking::CurrentPlane)
        throw std::invalid_argument("invalid GPU action executor configuration");
    auto& p = *impl_;
    if (support && (p.placement || batch != 1 || ranking != NeuralRanking::Learned || !network ||
                    network->architecture() != conditioned_placement_schema ||
                    support->collector.architecture() != conditioned_placement_schema))
        throw std::invalid_argument("ranking support requires sequential v4 learned endpoints");
    if (p.placement && network && network->model_use() == ModelUse::EndpointReuseOnly)
        throw std::invalid_argument("endpoint scorer cannot predict free placements");
    DeviceScope scope(p.id);
    Buffer<Action> trace_actions(p.device, trace ? batch : 0);
    Buffer<Action> support_actions(p.device, support ? 16 : 0);
    Buffer<SupportedRanking> support_result(p.device, support ? 1 : 0);
    uint32_t support_iteration = 0;
    target = std::max<size_t>(1, target);
    p.read();
    bool stopped = false;
    auto stop = [&] { return stopped = stopped || (cancelled && cancelled()); };
    uint64_t inference_ns = 0;
    uint32_t accepted_batches = 0;
    while (p.host.faces > target && p.host.trials < budget && !stop()) {
        auto query = [&](const auto& callback) {
            // Entry read() synchronizes both copies of State. Teacher queries may
            // alter counters, selection and trial scratch, but must never commit.
            // The unconditional rank below rebuilds all consumed ranking scratch.
            p.read();
            const State saved = p.host;
            const auto revision = p.mesh.view.revision;
            const auto faces = p.mesh.view.faces;
            const bool teacher_policy = p.teacher_policy;
            auto restore = [&] {
                p.host = saved;
                p.state.upload(std::span{&p.host, 1});
                p.mesh.view.faces = faces;
                p.teacher_policy = teacher_policy;
                p.selected_is_action = false;
                // trial_revision stays monotonic: old teacher candidates must
                // never alias subsequent runtime candidates in an audit cache.
            };
            try {
                callback(*this);
                if (p.mesh.view.revision != revision || p.mesh.view.faces != faces)
                    throw std::logic_error("action observer committed or reset its state");
            } catch (...) {
                restore();
                throw;
            }
            restore();
        };
        if (observe)
            query(observe);
        auto start = std::chrono::steady_clock::now();
        uint32_t ranked_count = 0;
        const auto support_seed = support ? support->seed + support_iteration++ * 0x9e3779b9u : 0;
        std::array<Action, 16> supported_actions{};
        if (support) {
            query([&](GpuActionState& state) {
                const auto rows = state.teacher_actions(c, 16, support_seed, &support->collector,
                                                        TeacherSelection::PolicyMixed);
                ranked_count = uint32_t(rows.size());
                for (size_t i = 0; i < rows.size(); ++i)
                    supported_actions[i] = rows[i].action;
            });
            support_actions.upload(supported_actions);
        }
        p.rank(c, network, ranking, seed);
        if (!support)
            ranked_count = p.host.actions;
        if (support && ranked_count) {
            restrict_ranking<<<1, 1, 0, gpu::stream()>>>(
                p.state.p, p.edits.p, p.sorted_order.p, p.sorted_scores.p, support_actions.p,
                ranked_count, support_seed, support_result.p);
            p.read();
            if (support->observe) {
                SupportedRanking result;
                check(gpu::copy(&result, support_result.p, sizeof(result), cudaMemcpyDeviceToHost));
                support->observe(result);
            }
        }
        inference_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::steady_clock::now() - start)
                                     .count());
        if (!p.host.actions)
            break;
        bool accepted = false;
        while (p.host.position < ranked_count && p.host.trials < budget && !accepted && !stop()) {
            select_batch<<<1, 1, 0, gpu::stream()>>>(
                p.state.p, p.geometric.view(), p.edits.p, p.sorted_order.p, p.selected.p, p.used.p,
                p.mesh.view.vertices, uint32_t(target), batch, ranked_count);
            p.read();
            if (p.host.exhausted)
                break;
            while (p.host.selected && p.host.trials < budget && !stop()) {
                p.build_trial();
                if (p.placement || p.mesh.precision.n)
                    p.read();
                DeviceMeshView candidate;
                if (!p.host.invalid_placement || trace)
                    candidate = p.trial_view();
                accepted = !p.host.invalid_placement && gate(candidate);
                if (trace) {
                    teacher_records<<<1, 64, 0, gpu::stream()>>>(p.state.p, p.edits.p, p.selected.p,
                                                                 trace_actions.p, p.host.selected);
                    std::array<Action, 64> actions;
                    check(gpu::copy(actions.data(), trace_actions.p,
                                    p.host.selected * sizeof(Action), cudaMemcpyDeviceToHost));
                    trace({std::span(actions).first(p.host.selected), p.host.trials,
                           accepted_batches, p.host.faces, candidate.faces, p.host.first,
                           p.host.position, !p.host.invalid_placement, accepted});
                }
                verdict<<<1, 1, 0, gpu::stream()>>>(p.state.p, accepted);
                if (accepted) {
                    p.commit();
                    ++accepted_batches;
                    break;
                }
                p.read();
            }
        }
        if (!accepted)
            break;
    }
    p.read();
    if (statistics) {
        *statistics = {p.host.ranked,   p.host.trials, p.host.accepted,
                       p.host.rejected, inference_ns,  accepted_batches};
        statistics->stop_reason = stopped                   ? NeuralActionStop::Cancelled
                                  : p.host.faces <= target  ? NeuralActionStop::TargetReached
                                  : p.host.trials >= budget ? NeuralActionStop::TrialBudget
                                  : !p.host.actions         ? NeuralActionStop::NoLegalActions
                                                            : NeuralActionStop::NoAcceptedAction;
    }
    return p.download();
}
std::vector<PlacementRecord> GpuActionState::placements(const std::array<float, conditions>& c) {
    auto& p = *impl_;
    DeviceScope scope(p.id);
    if (!p.placement)
        throw std::logic_error("free placement state required");
    p.features(c, placement_features);
    p.read();
    std::vector<Edit> edits(p.host.actions);
    std::vector<float> x(size_t(p.host.actions) * placement_features);
    std::vector<PlacementRecord> out(p.host.actions);
    if (!out.empty()) {
        check(gpu::copy(edits.data(), p.edits.p, edits.size() * sizeof(Edit),
                        cudaMemcpyDeviceToHost));
        check(gpu::copy(x.data(), p.features_buffer.p, x.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
    }
    for (size_t i = 0; i < out.size(); ++i) {
        out[i].action = {edits[i].from, edits[i].to, p.host.revision};
        std::copy_n(x.data() + i * placement_features, placement_features, out[i].x.data());
    }
    return out;
}
std::vector<GpuActionState::Proposal> GpuActionState::teacher_proposals(Action a,
                                                                        bool supervise_normals) {
    auto& p = *impl_;
    DeviceScope scope(p.id);
    if (!p.placement)
        throw std::logic_error("free placement state required");
    p.select(a);
    teacher_placements<<<1, 32, 0, gpu::stream()>>>(
        p.mesh.view, p.geometric.view(), p.edits.p, p.selected.p, p.teacher.p,
        p.teacher_policy ? p.proposals.p : nullptr, supervise_normals);
    std::vector<Proposal> out((supervise_normals ? 20 : 10) + (p.teacher_policy ? 1 : 0));
    check(
        gpu::copy(out.data(), p.teacher.p, out.size() * sizeof(Proposal), cudaMemcpyDeviceToHost));
    return out;
}
std::vector<PlacementRecord>
GpuActionState::teacher_actions(const std::array<float, conditions>& c, uint32_t count,
                                uint32_t seed, ActionCuda* policy, TeacherSelection selection,
                                std::vector<Placement>* decoded_policy) {
    auto& p = *impl_;
    if (p.placement && policy && policy->model_use() == ModelUse::EndpointReuseOnly)
        throw std::invalid_argument("endpoint scorer cannot teach free placements");
    DeviceScope scope(p.id);
    if (!count || count > 16 || (policy && !is_placement_schema(policy->architecture())) ||
        selection > TeacherSelection::PolicyMixed ||
        (selection == TeacherSelection::PolicyMixed && !policy) ||
        (decoded_policy && (!policy || !p.placement)))
        throw std::invalid_argument("invalid teacher pool/policy");
    if (decoded_policy)
        decoded_policy->clear();
    p.selected_is_action = false;
    p.read();
    count = std::min(count, p.host.actions);
    if (!count)
        return {};
    p.teacher_policy = policy;
    // The geometric pool needs no policy pass. Mixed selection ranks legal
    // actions in bounded tiles, then decodes only the at-most-16 queried rows.
    auto geometric_rank = [&](NeuralRanking ranking) {
        rank_actions<<<blocks(p.action_capacity), 256, 0, gpu::stream()>>>(
            p.mesh.view, p.state.p, p.geometric.view(), p.edits.p, nullptr, p.scores.p, p.order.p,
            ranking, std::max(1e-20, p.bounds.diameter()), p.action_capacity, placement_outputs);
        check(cub::DeviceRadixSort::SortPairsDescending(
            p.sort_temp->p, p.sort_bytes, p.scores.p, p.sorted_scores.p, p.order.p,
            p.sorted_order.p, int(p.action_capacity), 0, 32, gpu::stream()));
    };
    if (selection == TeacherSelection::PolicyMixed) {
        p.rank(c, policy, NeuralRanking::Learned, seed);
        check(gpu::copy(p.teacher_ranked.p, p.sorted_order.p, count * sizeof(uint32_t),
                        cudaMemcpyDeviceToDevice));
        geometric_rank(NeuralRanking::CurrentPlane);
        check(gpu::copy(p.teacher_ranked.p + 16, p.sorted_order.p, count * sizeof(uint32_t),
                        cudaMemcpyDeviceToDevice));
        geometric_rank(NeuralRanking::ShortestEdge);
        check(gpu::copy(p.teacher_ranked.p + 32, p.sorted_order.p, count * sizeof(uint32_t),
                        cudaMemcpyDeviceToDevice));
        teacher_mixed_selection<<<1, 1, 0, gpu::stream()>>>(p.state.p, p.teacher_ranked.p,
                                                            p.selected.p, count, seed);
    } else {
        geometric_rank(NeuralRanking::CurrentPlane);
        teacher_selection<<<1, 1, 0, gpu::stream()>>>(p.state.p, p.sorted_order.p, p.selected.p,
                                                      count, seed);
    }
    p.features(c, placement_features, p.selected.p, count,
               policy ? policy->architecture() : placement_schema);
    if (policy && p.placement) {
        p.ensure(p.prediction, size_t(count) * placement_outputs);
        policy->predict_device(p.features_buffer.p, p.prediction.p, count);
        decode_placements<<<1, 32, 0, gpu::stream()>>>(p.mesh.view, p.state.p, p.edits.p,
                                                       p.prediction.p, p.proposals.p, p.selected.p,
                                                       count, c[3] != 0 || c[4] != 0 || c[5] != 0);
        p.read();
        if (decoded_policy) {
            capture_policy_placements<<<1, 32, 0, gpu::stream()>>>(p.selected.p, p.proposals.p,
                                                                   p.teacher.p, count);
            std::vector<Proposal> copied(count);
            check(gpu::copy(copied.data(), p.teacher.p, count * sizeof(Proposal),
                            cudaMemcpyDeviceToHost));
            for (const auto& proposal : copied)
                decoded_policy->push_back(proposal.placement);
        }
    }
    Buffer<Action> actions(p.device, count);
    teacher_records<<<1, 32, 0, gpu::stream()>>>(p.state.p, p.edits.p, p.selected.p, actions.p,
                                                 count);
    auto ids = actions.download();
    std::vector<float> x(size_t(count) * placement_features);
    check(
        gpu::copy(x.data(), p.features_buffer.p, x.size() * sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<PlacementRecord> out(count);
    for (uint32_t i = 0; i < count; ++i) {
        out[i].action = ids[i];
        std::copy_n(x.data() + size_t(i) * placement_features, placement_features, out[i].x.data());
    }
    return out;
}

bool GpuActionState::trial(Action a, const Placement& placement, DeviceMeshView& view) {
    auto& p = *impl_;
    DeviceScope scope(p.id);
    if (!p.placement)
        throw std::logic_error("free placement state required");
    p.select(a);
    assign_placement<<<1, 1, 0, gpu::stream()>>>(p.selected.p, p.proposals.p, placement);
    p.build_trial();
    p.read();
    if (p.host.invalid_placement)
        return false;
    view = p.trial_view();
    return true;
}
std::vector<DeviceMeshView> GpuActionState::trial_batch(Action action,
                                                        std::span<const Proposal> proposals) {
    auto& p = *impl_;
    DeviceScope scope(p.id);
    if (!p.placement || proposals.empty() || proposals.size() > p.trial_slots.size())
        throw std::invalid_argument("independent trial batch outside 1..4");
    for (size_t i = proposals.size(); i < p.trial_slots.size(); ++i)
        p.trial_slots[i].reset();
    // Allocate before queuing writes, so allocation failure can safely retry a
    // smaller batch. No geometry is committed by this operation.
    for (size_t i = 0; i < proposals.size(); ++i)
        if (!p.trial_slots[i])
            p.trial_slots[i] = std::make_unique<Impl::TrialBuffers>(p.device, p.mesh.view);
    p.select(action);
    p.build_trial_topology();
    std::vector<DeviceMeshView> views;
    views.reserve(proposals.size());
    for (size_t i = 0; i < proposals.size(); ++i) {
        auto& slot = *p.trial_slots[i];
        assign_trial<<<1, 1, 0, gpu::stream()>>>(p.selected.p, p.proposals.p,
                                                 proposals[i].placement, slot.status.p,
                                                 p.offsets.p + p.face_capacity);
        place_vertices<<<blocks(p.mesh.view.vertices), 256, 0, gpu::stream()>>>(
            p.mesh.view, p.original->view, p.state.p, p.edits.p, p.proposals.p, p.selected.p,
            p.geometry.p, p.canonical.p, p.original_offsets.p, p.original_faces.p, slot.positions.p,
            slot.normals.p, slot.uv.p, slot.colors.p, slot.tangents.p, p.packed_uv,
            &slot.status.p->invalid);
        check(gpu::memset(p.slots.p, 255, p.slots.n * sizeof(uint32_t)));
        validate_placement<<<blocks(p.face_capacity), 256, 0, gpu::stream()>>>(
            p.mesh.view, p.state.p, p.remap.p, p.keep.p, slot.positions.p, slot.uv.p,
            &slot.status.p->invalid, p.preserve_uv, p.geometry.p, p.slots.p, p.hash_capacity);
        p.build_precision(slot.bits, &slot.status.p->invalid);
        auto view = p.mesh.view;
        view.exact_position_bits = slot.bits.p;
        view.positions = slot.positions.p;
        view.normals = slot.normals.p;
        view.uv = slot.uv.p;
        view.colors = slot.colors.p;
        view.tangents = slot.tangents.p;
        view.indices = p.trial_indices.p;
        view.materials = p.trial_materials.p;
        view.faces = p.face_capacity;
        view.trial_status = slot.status.p;
        view.revision = ++p.trial_revision;
        views.push_back(view);
    }
    check(cudaGetLastError());
    return views;
}
void GpuActionState::commit(Action a, const Placement& placement) {
    DeviceMeshView view;
    if (!trial(a, placement, view))
        throw std::invalid_argument("invalid free placement");
    auto& p = *impl_;
    DeviceScope scope(p.id);
    p.commit();
}
Lod GpuActionState::snapshot() {
    auto& p = *impl_;
    DeviceScope scope(p.id);
    return p.download();
}
} // namespace blitz::neural
