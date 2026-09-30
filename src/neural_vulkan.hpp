#pragma once
#include "neural_device_view.hpp"
#include "blitz/neural.hpp"
#include <memory>
#include <span>
namespace blitz::neural {
struct AuditPixel;
struct MemoryBudget;
struct RasterDebugPixel {uint32_t face;float depth;};
struct RasterSurfaces {uint64_t mask{},attributes{},colors{};uint32_t size{};};
struct RasterOutput {AuditPixel* pixels{};Vec3* colors{};RasterDebugPixel* debug{};RasterSurfaces surfaces{};bool clipped{};};
struct VulkanTiming {double packing_seconds{},render_seconds{},unpack_seconds{};uint64_t draws{},packed_meshes{};};
// One bounded, reusable hardware target. Image caches own compact linear CUDA
// rasters, independently of Vulkan attachment lifetime and renderer ownership.
class VulkanRaster {
    struct Impl;std::unique_ptr<Impl> impl_;
public:
    explicit VulkanRaster(const NeuralOptions&,bool timings=false,MemoryBudget* budget=nullptr);
    ~VulkanRaster();
    VulkanRaster(const VulkanRaster&)=delete;
    // All pointers are device storage on options.device. Work joins CUDA's
    // default stream via external semaphores. Output must outlive queued work.
    bool render(DeviceMeshView,const Bounds&,const Camera&,double screen,uint8_t sampling,bool two_sided,
        NeuralVertexStorage,AuditPixel* pixels,Vec3* colors,RasterDebugPixel* debug=nullptr);
    void render_batch(DeviceMeshView,const Bounds&,std::span<const Camera>,double screen,uint8_t sampling,bool two_sided,NeuralVertexStorage,std::span<RasterOutput>);
    size_t bytes()const;
    RasterSurfaces surfaces()const;
    void trim_targets(uint32_t keep);
    VulkanTiming timing()const;
};
}
