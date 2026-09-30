#pragma once
#include "neural_device_view.hpp"
#include "blitz/neural.hpp"
#include <memory>
namespace blitz::neural {
struct AuditPixel;
struct MemoryBudget;
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
        NeuralVertexStorage,AuditPixel* pixels,Vec3* colors);
    size_t bytes()const;
    VulkanTiming timing()const;
};
}
