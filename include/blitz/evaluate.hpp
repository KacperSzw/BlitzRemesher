#pragma once
#include "mesh.hpp"
#include <functional>
#include <limits>
namespace blitz {
enum class Profile:uint8_t { Coverage,Normals,Attributes };
struct Weights { double normal{180.0/(10*3.14159265358979323846)},color{4},material{4}; };
struct ViewSet { uint16_t orthographic{642},perspective{64}; uint32_t rotation_seed{0xB1172026}; };
struct PerformanceStats {
    uint64_t reduction_ns{},raster_ns{},distance_ns{};
    uint64_t solve_attempts{},singular_solves{},nonfinite_solves{},position_fallbacks{},nonfinite_costs{};
};
struct EvalSettings {
    Profile profile{Profile::Normals}; Weights weights{}; ViewSet views{};
    uint8_t supersample{8},max_supersample{32};
    double screen_size{512},limit{2};
    bool force_two_sided{false},force_scalar{false};
    std::function<bool()> cancelled;
    PerformanceStats* performance{}; // Borrowed optional accumulator; evaluate() adds to it.
};
struct Measurement {
    double error{},coverage{},coverage_upper{},changed_area{},normal_degrees{};
    uint32_t worst_view{},views_evaluated{}; uint8_t supersample{};
    bool complete{true},passed{true},resource_limited{};
};
struct Pixel {
    Vec3 normal{}; Vec4 color{};
    float depth{std::numeric_limits<float>::infinity()};
    uint16_t material{}; uint8_t covered{},visible{};
};
struct Raster { uint32_t width{},height{}; std::vector<Pixel> pixels; bool clipped{}; };
struct Camera { Vec3 right,up,forward; double distance{},focal{},scale{}; bool perspective{}; };
std::vector<Camera> cameras(const Bounds&,double,ViewSet);
Raster rasterize(MeshView,const Bounds&,const Camera&,double,uint8_t,bool);
double coverage_distance(const Raster&,const Raster&,uint8_t,bool force_scalar=false);
double attributed_distance(const Raster&,const Raster&,const EvalSettings&,double);
Measurement evaluate(MeshView,MeshView,const Bounds&,const EvalSettings&);
const char* evaluator_backend(bool force_scalar=false);
bool packed_coverage_enabled();
}
