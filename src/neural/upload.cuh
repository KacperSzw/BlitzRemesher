#pragma once
#include "neural/cuda.cuh"
#include "neural/vertex_storage.hpp"
namespace blitz::neural {
// Cold working-mesh upload. The immutable visual reference is uploaded through
// the separate FP32 audit path. Decode once into mutable GPU master streams.
__global__ void decode_working_mesh(const uint16_t* p, const uint32_t* normal, const uint16_t* uv,
                                    const uint32_t* tangent, uint32_t n, VertexBounds q,
                                    Vec3* positions, Vec3* normals, Vec2* texcoords,
                                    Vec4* tangents) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    positions[i] = {unpack_unorm16(p[i * 3], q.low.x, q.extent.x),
                    unpack_unorm16(p[i * 3 + 1], q.low.y, q.extent.y),
                    unpack_unorm16(p[i * 3 + 2], q.low.z, q.extent.z)};
    if (normals)
        normals[i] = unpack_direction(normal[i]);
    if (texcoords)
        texcoords[i] = {unpack_unorm16(uv[i * 2], -8, 16), unpack_unorm16(uv[i * 2 + 1], -8, 16)};
    if (tangents) {
        auto t = unpack_direction(tangent[i]);
        tangents[i] = {t.x, t.y, t.z, (tangent[i] >> 30) == 3 ? -1.f : 1.f};
    }
}
__global__ void restore_exact_positions(const uint32_t* ids, const Vec3* values, uint32_t count,
                                        Vec3* positions) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        positions[ids[i]] = values[i];
}
inline void upload_working_mesh(gpu::Device& device, MeshView m, VertexBounds q, Vec3* positions,
                                Vec3* normals, Vec2* texcoords, Vec4* tangents) {
    std::vector<uint16_t> p(m.positions.count * 3), uv(m.uv.count * 2);
    std::vector<uint32_t> normal(m.normals.count), tangent(m.tangents.count), ids;
    std::vector<Vec3> exact;
    for (size_t i = 0; i < m.positions.count; ++i) {
        auto v = m.positions[i];
        p[i * 3] = pack_unorm16(v.x, q.low.x, q.extent.x);
        p[i * 3 + 1] = pack_unorm16(v.y, q.low.y, q.extent.y);
        p[i * 3 + 2] = pack_unorm16(v.z, q.low.z, q.extent.z);
        if (m.exact_position(i)) {
            ids.push_back(uint32_t(i));
            exact.push_back(v);
        }
    }
    for (size_t i = 0; i < normal.size(); ++i)
        normal[i] = pack_direction(m.normals[i]);
    for (size_t i = 0; i < m.uv.count; ++i) {
        auto v = m.uv[i];
        if (v.x < -8 || v.x > 8 || v.y < -8 || v.y > 8)
            throw std::invalid_argument("working mesh UV outside [-8,8]");
        uv[i * 2] = pack_unorm16(v.x, -8, 16);
        uv[i * 2 + 1] = pack_unorm16(v.y, -8, 16);
    }
    for (size_t i = 0; i < tangent.size(); ++i) {
        auto v = m.tangents[i];
        tangent[i] = pack_direction({v.x, v.y, v.z}, v.w < 0 ? -1 : 1);
    }
    gpu::Buffer<uint16_t> dp(device, p.size()), du(device, uv.size());
    gpu::Buffer<uint32_t> dn(device, normal.size()), dt(device, tangent.size()),
        di(device, ids.size());
    gpu::Buffer<Vec3> de(device, exact.size());
    dp.upload(p);
    dn.upload(normal);
    du.upload(uv);
    dt.upload(tangent);
    di.upload(ids);
    de.upload(exact);
    decode_working_mesh<<<gpu::blocks(m.positions.count), 256, 0, gpu::stream()>>>(
        dp.p, dn.p, du.p, dt.p, uint32_t(m.positions.count), q, positions, normals, texcoords,
        tangents);
    if (!ids.empty())
        restore_exact_positions<<<gpu::blocks(ids.size()), 256, 0, gpu::stream()>>>(
            di.p, de.p, uint32_t(ids.size()), positions);
    gpu::check(cudaGetLastError());
}
} // namespace blitz::neural
