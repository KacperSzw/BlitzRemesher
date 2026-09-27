#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <vector>
namespace blitz {
struct Vec2 { float x{}, y{}; };
struct Vec3 { float x{}, y{}, z{}; };
struct Vec4 { float x{}, y{}, z{}, w{1}; };
inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
inline Vec3 operator*(Vec3 a, double s) { return {float(a.x*s),float(a.y*s),float(a.z*s)}; }
inline double dot(Vec3 a, Vec3 b) { return double(a.x)*b.x+double(a.y)*b.y+double(a.z)*b.z; }
inline Vec3 cross(Vec3 a, Vec3 b) { return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x}; }
inline double length(Vec3 a) { return std::sqrt(dot(a,a)); }
inline Vec3 normalized(Vec3 a) { double l=length(a); return l>0?a*(1/l):Vec3{}; }
inline bool finite(Vec3 a) { return std::isfinite(a.x)&&std::isfinite(a.y)&&std::isfinite(a.z); }
template<class T> struct Stream {
    const std::byte* data{}; size_t count{}; size_t stride{sizeof(T)};
    Stream()=default;
    Stream(const T* p,size_t n):data(reinterpret_cast<const std::byte*>(p)),count(n){}
    Stream(const std::vector<T>& v):Stream(v.data(),v.size()){}
    T operator[](size_t i) const { T v; std::memcpy(&v,data+i*stride,sizeof(T)); return v; }
    explicit operator bool() const { return data && count; }
};
struct MeshView {
    Stream<Vec3> positions,normals;
    Stream<Vec2> uv;
    Stream<Vec4> colors,tangents;
    std::span<const uint32_t> indices;
    std::span<const uint16_t> materials;
    std::span<const uint8_t> double_sided;
    size_t triangles() const { return indices.size()/3; }
    uint16_t material(size_t face) const { return materials.empty()?0:materials[face]; }
    bool two_sided(size_t face) const { auto m=material(face); return m<double_sided.size() && double_sided[m]; }
};
struct Mesh {
    std::vector<Vec3> positions,normals;
    std::vector<Vec2> uv;
    std::vector<Vec4> colors,tangents;
    std::vector<uint32_t> indices;
    std::vector<uint16_t> materials;
    std::vector<uint8_t> double_sided;
    MeshView view() const { return {positions,normals,uv,colors,tangents,indices,materials,double_sided}; }
};
struct Bounds { Vec3 center{}; double radius{}; double diameter() const {return radius*2;} };
Bounds bounds(MeshView);
std::string validate(MeshView);
Mesh copy_mesh(MeshView);
void compact(Mesh&);
struct Distortion {
    double mean_uv_density{},max_uv_density{},max_uv_anisotropy{};
    uint64_t negative_uv_faces{},degenerate_uv_faces{};
};
Distortion uv_distortion(MeshView);
}
