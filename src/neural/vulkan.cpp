#include "neural/vulkan.hpp"
#include "neural/cuda.cuh"
#include "neural/memory.hpp"
#include "neural/vulkan_cuda.hpp"
#include "neural_coverage_frag.inc"
#include "neural_coverage_vert.inc"
#include "neural_raster_frag.inc"
#include "neural_raster_vert.inc"
#include <algorithm>
#include <array>
#include <cstring>
#include <cuda_runtime.h>
#include <limits>
#include <unistd.h>
#include <vulkan/vulkan.h>
namespace blitz::neural {
namespace {
void vkcheck(VkResult r) {
    if (r == VK_ERROR_OUT_OF_DEVICE_MEMORY)
        throw gpu::ResourceError(NeuralResourceLimit::DeviceMemory, 0, 0,
                                 "Vulkan device memory exhausted");
    if (r != VK_SUCCESS)
        throw std::runtime_error("Vulkan error " + std::to_string(r));
}
void cucheck(cudaError_t r) {
    gpu::check(r);
}
template <class T> T info(VkStructureType type) {
    T x{};
    x.sType = type;
    return x;
}
struct Push {
    float right[4], up[4], forward[4], center[4], depth[4];
    uint32_t options[4];
};
} // namespace
struct VulkanRaster::Impl {
    VkInstance instance{};
    VkPhysicalDevice physical{};
    VkDevice device{};
    VkQueue queue{};
    uint32_t family{};
    VkCommandPool commands{};
    VkCommandBuffer command{};
    VkDescriptorPool descriptors{};
    VkDescriptorSetLayout set_layout{};
    VkPipelineLayout pipeline_layout{};
    VkSemaphore semaphore{};
    cudaExternalSemaphore_t cuda_semaphore{};
    uint64_t serial{};
    VkFence completion{};
    bool submitted{};
    VkPhysicalDeviceMemoryProperties memory{};
    VkPhysicalDeviceProperties properties{};
    PFN_vkGetMemoryFdKHR get_memory_fd{};
    PFN_vkGetSemaphoreFdKHR get_semaphore_fd{};
    std::array<VkPipeline, 48> pipelines{}; // storage × normal/color presence × pass
    size_t allocated{}, limit;
    MemoryBudget* budget{};
    bool timings;
    VulkanTiming measured;
    VkQueryPool queries{};
    std::array<cudaEvent_t, 4> events{};
    struct Geometry {
        VkBuffer buffer{};
        VkDeviceMemory memory{};
        cudaExternalMemory_t external{};
        void* cuda{};
        size_t bytes{}, capacity{};
        DeviceMeshView mesh{};
        NeuralVertexStorage storage{};
        DrawLayout layout{};
        VkDescriptorSet set{};
        bool released{}, coverage_only{};
    };
    std::array<Geometry, 8> geometries{};
    unsigned next_geometry{};
    struct Target {
        VkImage image{};
        VkDeviceMemory memory{};
        VkImageView view{};
        cudaExternalMemory_t external{};
        cudaMipmappedArray_t mip{};
        cudaSurfaceObject_t surface{};
        size_t bytes{};
        bool released{};
    };
    struct Attachments {
        Target mask, attributes, colors, depth, witness;
        uint32_t extent{};
        bool colors_enabled{}, debug_enabled{}, coverage_only{};
    };
    std::array<Attachments, 8> targets;
    explicit Impl(const NeuralOptions& o, bool timing, MemoryBudget* shared)
        : limit(size_t(o.memory_mib) << 20), budget(shared), timings(timing) {
        if (o.exact_position_bps > 10000 || o.vertex_storage > NeuralVertexStorage::Automatic ||
            o.memory_mib < 128 || o.memory_mib > 65536)
            throw std::invalid_argument("invalid Vulkan draw options");
        if (!budget && current_memory_budget && current_memory_budget->device == o.device)
            budget = current_memory_budget;
        try {
            initialize(o);
        } catch (...) {
            destroy();
            throw;
        }
    }
    ~Impl() {
        destroy();
    }
    void account(size_t bytes) {
        if (bytes > limit - allocated || (budget && !budget->reserve(bytes)))
            throw gpu::ResourceError(NeuralResourceLimit::WorkspaceMemory,
                                     (budget ? budget->live.load() : allocated) + bytes,
                                     budget ? budget->limit : limit,
                                     "Vulkan workspace exceeds memory cap");
        allocated += bytes;
    }
    void unaccount(size_t bytes) {
        allocated -= bytes;
        if (budget)
            budget->release(bytes);
    }
    uint32_t memory_type(uint32_t bits) {
        for (uint32_t i = 0; i < memory.memoryTypeCount; ++i)
            if ((bits & (1u << i)) &&
                (memory.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
                return i;
        throw NeuralUnavailable("no Vulkan device-local memory");
    }
    int memory_fd(VkDeviceMemory value) {
        auto i = info<VkMemoryGetFdInfoKHR>(VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR);
        i.memory = value;
        i.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        int fd = -1;
        vkcheck(get_memory_fd(device, &i, &fd));
        return fd;
    }
    void initialize(const NeuralOptions& o) {
        auto application = info<VkApplicationInfo>(VK_STRUCTURE_TYPE_APPLICATION_INFO);
        application.pApplicationName = "BlitzRemesher audit";
        application.apiVersion = VK_API_VERSION_1_3;
        auto create = info<VkInstanceCreateInfo>(VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO);
        create.pApplicationInfo = &application;
        vkcheck(vkCreateInstance(&create, nullptr, &instance));
        uint32_t count = 0;
        vkcheck(vkEnumeratePhysicalDevices(instance, &count, nullptr));
        std::vector<VkPhysicalDevice> devices(count);
        vkcheck(vkEnumeratePhysicalDevices(instance, &count, devices.data()));
        cudaDeviceProp cuda_properties{};
        cucheck(cudaGetDeviceProperties(&cuda_properties, o.device));
        for (auto candidate : devices) {
            auto id =
                info<VkPhysicalDeviceIDProperties>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES);
            auto p =
                info<VkPhysicalDeviceProperties2>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2);
            p.pNext = &id;
            vkGetPhysicalDeviceProperties2(candidate, &p);
            if (!std::memcmp(id.deviceUUID, cuda_properties.uuid.bytes, VK_UUID_SIZE)) {
                physical = candidate;
                properties = p.properties;
                break;
            }
        }
        if (!physical)
            throw NeuralUnavailable("CUDA GPU has no matching Vulkan UUID");
        std::vector<const char*> extensions = {VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
                                               VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
                                               VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME};
        vkcheck(vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, nullptr));
        std::vector<VkExtensionProperties> supported(count);
        vkcheck(vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, supported.data()));
        for (auto name : extensions)
            if (std::none_of(supported.begin(), supported.end(),
                             [&](auto& e) { return std::strcmp(e.extensionName, name) == 0; }))
                throw NeuralUnavailable(std::string("missing Vulkan extension: ") + name);
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
        std::vector<VkQueueFamilyProperties> queues(count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, queues.data());
        family = UINT32_MAX;
        for (uint32_t i = 0; i < count; ++i)
            if (queues[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                family = i;
                break;
            }
        if (family == UINT32_MAX)
            throw NeuralUnavailable("no Vulkan graphics queue");
        auto f13 = info<VkPhysicalDeviceVulkan13Features>(
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES);
        auto f12 = info<VkPhysicalDeviceVulkan12Features>(
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES);
        f12.pNext = &f13;
        auto features =
            info<VkPhysicalDeviceFeatures2>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2);
        features.pNext = &f12;
        vkGetPhysicalDeviceFeatures2(physical, &features);
        if (!features.features.geometryShader || !f12.timelineSemaphore || !f13.dynamicRendering ||
            !f13.synchronization2 || !f13.shaderDemoteToHelperInvocation)
            throw NeuralUnavailable(
                "Vulkan audit requires geometry primitive IDs, timeline semaphores, dynamic "
                "rendering, synchronization2 and helper invocation demotion");
        f12 = info<VkPhysicalDeviceVulkan12Features>(
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES);
        f12.timelineSemaphore = VK_TRUE;
        f12.pNext = &f13;
        f13 = info<VkPhysicalDeviceVulkan13Features>(
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES);
        f13.dynamicRendering = VK_TRUE;
        f13.synchronization2 = VK_TRUE;
        f13.shaderDemoteToHelperInvocation = VK_TRUE;
        features.features = {};
        features.features.geometryShader = VK_TRUE;
        float priority = 1;
        auto qc = info<VkDeviceQueueCreateInfo>(VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO);
        qc.queueFamilyIndex = family;
        qc.queueCount = 1;
        qc.pQueuePriorities = &priority;
        auto dc = info<VkDeviceCreateInfo>(VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO);
        dc.pNext = &features;
        dc.queueCreateInfoCount = 1;
        dc.pQueueCreateInfos = &qc;
        dc.enabledExtensionCount = uint32_t(extensions.size());
        dc.ppEnabledExtensionNames = extensions.data();
        vkcheck(vkCreateDevice(physical, &dc, nullptr, &device));
        vkGetDeviceQueue(device, family, 0, &queue);
        vkGetPhysicalDeviceMemoryProperties(physical, &memory);
        get_memory_fd =
            reinterpret_cast<PFN_vkGetMemoryFdKHR>(vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR"));
        get_semaphore_fd = reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(
            vkGetDeviceProcAddr(device, "vkGetSemaphoreFdKHR"));
        auto st = info<VkSemaphoreTypeCreateInfo>(VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO);
        st.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        auto se = info<VkExportSemaphoreCreateInfo>(VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO);
        se.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
        se.pNext = &st;
        auto sc = info<VkSemaphoreCreateInfo>(VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO);
        sc.pNext = &se;
        vkcheck(vkCreateSemaphore(device, &sc, nullptr, &semaphore));
        auto sf = info<VkSemaphoreGetFdInfoKHR>(VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR);
        sf.semaphore = semaphore;
        sf.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
        int fd = -1;
        vkcheck(get_semaphore_fd(device, &sf, &fd));
        cudaExternalSemaphoreHandleDesc sem_desc{};
        sem_desc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
        sem_desc.handle.fd = fd;
        auto status = cudaImportExternalSemaphore(&cuda_semaphore, &sem_desc);
        if (status != cudaSuccess)
            close(fd);
        cucheck(status);
        auto fc = info<VkFenceCreateInfo>(VK_STRUCTURE_TYPE_FENCE_CREATE_INFO);
        vkcheck(vkCreateFence(device, &fc, nullptr, &completion));
        auto cp = info<VkCommandPoolCreateInfo>(VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO);
        cp.queueFamilyIndex = family;
        cp.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        vkcheck(vkCreateCommandPool(device, &cp, nullptr, &commands));
        auto ca = info<VkCommandBufferAllocateInfo>(VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO);
        ca.commandPool = commands;
        ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ca.commandBufferCount = 1;
        vkcheck(vkAllocateCommandBuffers(device, &ca, &command));
        std::array<VkDescriptorSetLayoutBinding, 3> bindings = {
            VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                         VK_SHADER_STAGE_VERTEX_BIT, nullptr},
            VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                         VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            VkDescriptorSetLayoutBinding{2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                         VK_SHADER_STAGE_VERTEX_BIT, nullptr}};
        auto sl = info<VkDescriptorSetLayoutCreateInfo>(
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO);
        sl.bindingCount = bindings.size();
        sl.pBindings = bindings.data();
        vkcheck(vkCreateDescriptorSetLayout(device, &sl, nullptr, &set_layout));
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 24};
        auto dp = info<VkDescriptorPoolCreateInfo>(VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO);
        dp.maxSets = 8;
        dp.poolSizeCount = 1;
        dp.pPoolSizes = &ps;
        vkcheck(vkCreateDescriptorPool(device, &dp, nullptr, &descriptors));
        for (auto& g : geometries) {
            auto ds =
                info<VkDescriptorSetAllocateInfo>(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO);
            ds.descriptorPool = descriptors;
            ds.descriptorSetCount = 1;
            ds.pSetLayouts = &set_layout;
            vkcheck(vkAllocateDescriptorSets(device, &ds, &g.set));
        }
        VkPushConstantRange range{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                                  sizeof(Push)};
        auto pl = info<VkPipelineLayoutCreateInfo>(VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO);
        pl.setLayoutCount = 1;
        pl.pSetLayouts = &set_layout;
        pl.pushConstantRangeCount = 1;
        pl.pPushConstantRanges = &range;
        vkcheck(vkCreatePipelineLayout(device, &pl, nullptr, &pipeline_layout));
        for (auto format : {VK_FORMAT_R32G32B32_SFLOAT, VK_FORMAT_R16G16B16_UNORM,
                            VK_FORMAT_A2B10G10R10_SNORM_PACK32, VK_FORMAT_R8G8B8A8_UNORM}) {
            VkFormatProperties p;
            vkGetPhysicalDeviceFormatProperties(physical, format, &p);
            if (!(p.bufferFeatures & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT))
                throw NeuralUnavailable("unsupported packed Vulkan vertex format");
        }
        if (timings) {
            auto q = info<VkQueryPoolCreateInfo>(VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO);
            q.queryType = VK_QUERY_TYPE_TIMESTAMP;
            q.queryCount = 2;
            vkcheck(vkCreateQueryPool(device, &q, nullptr, &queries));
            for (auto& event : events)
                cucheck(cudaEventCreate(&event));
        }
    }
    VkShaderModule shader(const uint32_t* code, size_t bytes) {
        auto s = info<VkShaderModuleCreateInfo>(VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO);
        s.codeSize = bytes;
        s.pCode = code;
        VkShaderModule out{};
        vkcheck(vkCreateShaderModule(device, &s, nullptr, &out));
        return out;
    }
    VkPipeline pipeline(NeuralVertexStorage storage, bool normals, bool has_colors, bool coverage,
                        bool debug) {
        size_t index = (size_t(storage) * 8 + size_t(normals) * 4 + size_t(has_colors) * 2 +
                        size_t(coverage)) *
                           2 +
                       debug;
        if (pipelines[index])
            return pipelines[index];
        VkShaderModule vs = coverage ? shader(neural_coverage_vert, sizeof(neural_coverage_vert))
                                     : shader(neural_raster_vert, sizeof(neural_raster_vert)),
                       fs = coverage ? shader(neural_coverage_frag, sizeof(neural_coverage_frag))
                                     : shader(neural_raster_frag, sizeof(neural_raster_frag));
        try {
            std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
            for (auto& s : stages)
                s = info<VkPipelineShaderStageCreateInfo>(
                    VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO);
            stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = vs;
            stages[0].pName = "main";
            stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = fs;
            stages[1].pName = "main";
            bool p16 = storage != NeuralVertexStorage::Float32,
                 packed = storage == NeuralVertexStorage::Packed;
            std::array<VkVertexInputBindingDescription, 3> bindings = {
                VkVertexInputBindingDescription{0, p16 ? 6u : 12u, VK_VERTEX_INPUT_RATE_VERTEX},
                {1, normals ? (packed ? 4u : 12u) : 0u, VK_VERTEX_INPUT_RATE_VERTEX},
                {2, has_colors ? 4u : 0u, VK_VERTEX_INPUT_RATE_VERTEX}};
            std::array<VkVertexInputAttributeDescription, 3> attrs = {
                VkVertexInputAttributeDescription{
                    0, 0, p16 ? VK_FORMAT_R16G16B16_UNORM : VK_FORMAT_R32G32B32_SFLOAT, 0},
                {1, 1, packed ? VK_FORMAT_A2B10G10R10_SNORM_PACK32 : VK_FORMAT_R32G32B32_SFLOAT, 0},
                {2, 2, VK_FORMAT_R8G8B8A8_UNORM, 0}};
            auto vi = info<VkPipelineVertexInputStateCreateInfo>(
                VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO);
            vi.vertexBindingDescriptionCount = coverage ? 1 : bindings.size();
            vi.pVertexBindingDescriptions = bindings.data();
            vi.vertexAttributeDescriptionCount = coverage ? 1 : attrs.size();
            vi.pVertexAttributeDescriptions = attrs.data();
            auto ia = info<VkPipelineInputAssemblyStateCreateInfo>(
                VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO);
            ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            auto vp = info<VkPipelineViewportStateCreateInfo>(
                VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO);
            vp.viewportCount = 1;
            vp.scissorCount = 1;
            auto cr = info<VkPipelineRasterizationConservativeStateCreateInfoEXT>(
                VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_CONSERVATIVE_STATE_CREATE_INFO_EXT);
            cr.conservativeRasterizationMode = VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT;
            auto rs = info<VkPipelineRasterizationStateCreateInfo>(
                VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO);
            rs.pNext = coverage ? &cr : nullptr;
            rs.polygonMode = VK_POLYGON_MODE_FILL;
            rs.cullMode = VK_CULL_MODE_NONE;
            rs.frontFace = VK_FRONT_FACE_CLOCKWISE;
            rs.lineWidth = 1;
            auto ms = info<VkPipelineMultisampleStateCreateInfo>(
                VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO);
            ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
            auto ds = info<VkPipelineDepthStencilStateCreateInfo>(
                VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO);
            ds.depthTestEnable = !coverage;
            ds.depthWriteEnable = !coverage;
            ds.depthCompareOp = VK_COMPARE_OP_LESS;
            std::array<VkPipelineColorBlendAttachmentState, 3> blending{};
            for (auto& b : blending)
                b.colorWriteMask = 15;
            auto blend = info<VkPipelineColorBlendStateCreateInfo>(
                VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO);
            blend.attachmentCount = coverage ? 1 : debug ? 3 : 2;
            blend.pAttachments = blending.data();
            VkDynamicState dynamic[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
            auto dyn = info<VkPipelineDynamicStateCreateInfo>(
                VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO);
            dyn.dynamicStateCount = 2;
            dyn.pDynamicStates = dynamic;
            std::array<VkFormat, 3> formats = {
                coverage ? VK_FORMAT_R8_UINT : VK_FORMAT_R32G32B32A32_SFLOAT,
                has_colors ? VK_FORMAT_R32G32B32A32_SFLOAT : VK_FORMAT_UNDEFINED,
                debug ? VK_FORMAT_R32G32_UINT : VK_FORMAT_UNDEFINED};
            auto rendering = info<VkPipelineRenderingCreateInfo>(
                VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO);
            rendering.colorAttachmentCount = coverage ? 1 : debug ? 3 : 2;
            rendering.pColorAttachmentFormats = formats.data();
            rendering.depthAttachmentFormat = coverage ? VK_FORMAT_UNDEFINED : VK_FORMAT_D32_SFLOAT;
            auto pc =
                info<VkGraphicsPipelineCreateInfo>(VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO);
            pc.pNext = &rendering;
            pc.stageCount = 2;
            pc.pStages = stages.data();
            pc.pVertexInputState = &vi;
            pc.pInputAssemblyState = &ia;
            pc.pViewportState = &vp;
            pc.pRasterizationState = &rs;
            pc.pMultisampleState = &ms;
            pc.pDepthStencilState = &ds;
            pc.pColorBlendState = &blend;
            pc.pDynamicState = &dyn;
            pc.layout = pipeline_layout;
            vkcheck(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pc, nullptr,
                                              &pipelines[index]));
        } catch (...) {
            vkDestroyShaderModule(device, vs, nullptr);
            vkDestroyShaderModule(device, fs, nullptr);
            throw;
        }
        vkDestroyShaderModule(device, vs, nullptr);
        vkDestroyShaderModule(device, fs, nullptr);
        return pipelines[index];
    }
    void drop(Geometry& g) {
        if (g.cuda)
            cudaFree(g.cuda);
        if (g.external)
            cudaDestroyExternalMemory(g.external);
        if (g.buffer)
            vkDestroyBuffer(device, g.buffer, nullptr);
        if (g.memory)
            vkFreeMemory(device, g.memory, nullptr);
        unaccount(g.bytes);
        auto set = g.set;
        g = {};
        g.set = set;
    }
    void drop(Target& t) {
        if (t.surface)
            cudaDestroySurfaceObject(t.surface);
        if (t.mip)
            cudaFreeMipmappedArray(t.mip);
        if (t.external)
            cudaDestroyExternalMemory(t.external);
        if (t.view)
            vkDestroyImageView(device, t.view, nullptr);
        if (t.image)
            vkDestroyImage(device, t.image, nullptr);
        if (t.memory)
            vkFreeMemory(device, t.memory, nullptr);
        unaccount(t.bytes);
        t = {};
    }
    void geometry_buffer(Geometry& g, size_t bytes) {
        if (g.capacity >= bytes)
            return;
        drop(g);
        auto ext = info<VkExternalMemoryBufferCreateInfo>(
            VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO);
        ext.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        auto bc = info<VkBufferCreateInfo>(VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO);
        bc.pNext = &ext;
        bc.size = bytes;
        bc.usage = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                   VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        vkcheck(vkCreateBuffer(device, &bc, nullptr, &g.buffer));
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device, g.buffer, &req);
        account(req.size);
        g.bytes = req.size;
        auto ex = info<VkExportMemoryAllocateInfo>(VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO);
        ex.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        auto dedicated =
            info<VkMemoryDedicatedAllocateInfo>(VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO);
        dedicated.buffer = g.buffer;
        ex.pNext = &dedicated;
        auto ai = info<VkMemoryAllocateInfo>(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);
        ai.pNext = &ex;
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = memory_type(req.memoryTypeBits);
        vkcheck(vkAllocateMemory(device, &ai, nullptr, &g.memory));
        vkcheck(vkBindBufferMemory(device, g.buffer, g.memory, 0));
        cudaExternalMemoryHandleDesc desc{};
        desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
        desc.handle.fd = memory_fd(g.memory);
        desc.size = req.size;
        desc.flags = cudaExternalMemoryDedicated;
        auto status = cudaImportExternalMemory(&g.external, &desc);
        if (status != cudaSuccess)
            close(desc.handle.fd);
        cucheck(status);
        cudaExternalMemoryBufferDesc map{};
        map.size = bytes;
        cucheck(cudaExternalMemoryGetMappedBuffer(&g.cuda, g.external, &map));
        g.capacity = bytes;
    }
    void target(Target& t, VkFormat format, uint32_t size, bool exported) {
        VkImageUsageFlags usage = exported ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                                           : VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        auto ex = info<VkExternalMemoryImageCreateInfo>(
            VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO);
        ex.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        if (exported) {
            auto ei = info<VkPhysicalDeviceExternalImageFormatInfo>(
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO);
            ei.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
            auto fi = info<VkPhysicalDeviceImageFormatInfo2>(
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2);
            fi.pNext = &ei;
            fi.format = format;
            fi.type = VK_IMAGE_TYPE_2D;
            fi.tiling = VK_IMAGE_TILING_OPTIMAL;
            fi.usage = usage;
            auto ep = info<VkExternalImageFormatProperties>(
                VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES);
            auto fp = info<VkImageFormatProperties2>(VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2);
            fp.pNext = &ep;
            vkcheck(vkGetPhysicalDeviceImageFormatProperties2(physical, &fi, &fp));
            if (!(ep.externalMemoryProperties.externalMemoryFeatures &
                  VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT))
                throw NeuralUnavailable("Vulkan attachment cannot be exported to CUDA");
        }
        auto ic = info<VkImageCreateInfo>(VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO);
        ic.pNext = exported ? &ex : nullptr;
        ic.imageType = VK_IMAGE_TYPE_2D;
        ic.format = format;
        ic.extent = {size, size, 1};
        ic.mipLevels = 1;
        ic.arrayLayers = 1;
        ic.samples = VK_SAMPLE_COUNT_1_BIT;
        ic.tiling = VK_IMAGE_TILING_OPTIMAL;
        ic.usage = usage;
        vkcheck(vkCreateImage(device, &ic, nullptr, &t.image));
        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(device, t.image, &req);
        account(req.size);
        t.bytes = req.size;
        auto de =
            info<VkMemoryDedicatedAllocateInfo>(VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO);
        de.image = t.image;
        auto ea = info<VkExportMemoryAllocateInfo>(VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO);
        ea.pNext = &de;
        ea.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        auto ai = info<VkMemoryAllocateInfo>(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);
        ai.pNext = exported ? static_cast<void*>(&ea) : &de;
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = memory_type(req.memoryTypeBits);
        vkcheck(vkAllocateMemory(device, &ai, nullptr, &t.memory));
        vkcheck(vkBindImageMemory(device, t.image, t.memory, 0));
        auto view = info<VkImageViewCreateInfo>(VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO);
        view.image = t.image;
        view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view.format = format;
        view.subresourceRange = {exported ? VK_IMAGE_ASPECT_COLOR_BIT : VK_IMAGE_ASPECT_DEPTH_BIT,
                                 0, 1, 0, 1};
        vkcheck(vkCreateImageView(device, &view, nullptr, &t.view));
        if (exported) {
            cudaExternalMemoryHandleDesc d{};
            d.type = cudaExternalMemoryHandleTypeOpaqueFd;
            d.handle.fd = memory_fd(t.memory);
            d.size = req.size;
            d.flags = cudaExternalMemoryDedicated;
            auto status = cudaImportExternalMemory(&t.external, &d);
            if (status != cudaSuccess)
                close(d.handle.fd);
            cucheck(status);
            cudaExternalMemoryMipmappedArrayDesc m{};
            m.formatDesc = format == VK_FORMAT_R8_UINT
                               ? cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned)
                           : format == VK_FORMAT_R32G32_UINT
                               ? cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindUnsigned)
                               : cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
            m.extent = make_cudaExtent(size, size, 0);
            m.flags = cudaArrayColorAttachment | cudaArraySurfaceLoadStore;
            m.numLevels = 1;
            cucheck(cudaExternalMemoryGetMappedMipmappedArray(&t.mip, t.external, &m));
            cudaArray_t level{};
            cucheck(cudaGetMipmappedArrayLevel(&level, t.mip, 0));
            cudaResourceDesc r{};
            r.resType = cudaResourceTypeArray;
            r.res.array.array = level;
            cucheck(cudaCreateSurfaceObject(&t.surface, &r));
        }
    }
    void image_barrier(Target& t, bool acquire, bool is_depth = false) {
        auto b = info<VkImageMemoryBarrier2>(VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2);
        b.image = t.image;
        b.subresourceRange = {is_depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT, 0,
                              1, 0, 1};
        auto attachment = is_depth ? VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL
                                   : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        if (acquire) {
            b.oldLayout = t.released ? (is_depth ? attachment : VK_IMAGE_LAYOUT_GENERAL)
                                     : VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout = attachment;
            b.dstStageMask = is_depth ? VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                                            VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT
                                      : VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            b.dstAccessMask = is_depth ? VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
                                       : VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            if (t.released && !is_depth) {
                b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
                b.dstQueueFamilyIndex = family;
            }
            if (t.released && is_depth) {
                b.srcStageMask = b.dstStageMask;
                b.srcAccessMask = b.dstAccessMask;
            }
        } else {
            b.oldLayout = attachment;
            b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            b.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            b.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            b.srcQueueFamilyIndex = family;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
        }
        auto d = info<VkDependencyInfo>(VK_STRUCTURE_TYPE_DEPENDENCY_INFO);
        d.imageMemoryBarrierCount = 1;
        d.pImageMemoryBarriers = &b;
        vkCmdPipelineBarrier2(command, &d);
        t.released = true;
    }
    void buffer_barrier(Geometry& g, bool acquire) {
        auto b = info<VkBufferMemoryBarrier2>(VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2);
        b.buffer = g.buffer;
        b.size = VK_WHOLE_SIZE;
        b.srcQueueFamilyIndex = acquire ? VK_QUEUE_FAMILY_EXTERNAL : family;
        b.dstQueueFamilyIndex = acquire ? family : VK_QUEUE_FAMILY_EXTERNAL;
        if (acquire) {
            b.dstStageMask =
                VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT |
                VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            b.dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT |
                              VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_2_INDEX_READ_BIT |
                              VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        } else {
            b.srcStageMask =
                VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
            b.srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT;
        }
        auto d = info<VkDependencyInfo>(VK_STRUCTURE_TYPE_DEPENDENCY_INFO);
        d.bufferMemoryBarrierCount = 1;
        d.pBufferMemoryBarriers = &b;
        vkCmdPipelineBarrier2(command, &d);
        g.released = true;
    }
    void release_targets(Attachments& t) {
        drop(t.mask);
        drop(t.attributes);
        drop(t.colors);
        drop(t.depth);
        drop(t.witness);
        t.extent = 0;
    }
    void render_batch(std::span<const DeviceMeshView> meshes, const Bounds& bounds,
                      std::span<const Camera> cameras, double screen, uint8_t ss, bool two,
                      NeuralVertexStorage storage, std::span<RasterOutput> out,
                      bool coverage_only) {
        if (cameras.empty() || cameras.size() > targets.size() || cameras.size() != out.size() ||
            meshes.size() != cameras.size())
            throw std::invalid_argument("Vulkan camera batch outside 1..8");
        if (storage == NeuralVertexStorage::Automatic)
            storage = NeuralVertexStorage::Packed;
        if (storage > NeuralVertexStorage::Packed || !std::isfinite(screen) || screen <= 0 || !ss)
            throw std::invalid_argument("invalid Vulkan draw");
        uint32_t size = uint32_t(std::ceil(screen + 8)) * ss;
        if (!size || size > properties.limits.maxImageDimension2D)
            throw NeuralUnavailable("Vulkan target extent exceeds device limit");
        if (submitted) {
            vkcheck(vkWaitForFences(device, 1, &completion, VK_TRUE, UINT64_MAX));
            vkcheck(vkResetFences(device, 1, &completion));
            submitted = false;
        }
        std::array<Geometry*, 8> geometry{};
        uint32_t repacked = 0;
        DrawStatusSources status{};
        status.count = uint32_t(meshes.size());
        if (timings)
            cucheck(cudaEventRecord(events[0], gpu::stream()));
        for (size_t slot = 0; slot < meshes.size(); ++slot) {
            auto mesh = meshes[slot];
            if (!mesh.identity)
                throw std::invalid_argument("Vulkan candidate identity");
            Geometry* found = nullptr;
            for (auto& g : geometries)
                if (g.buffer && g.mesh.identity == mesh.identity &&
                    g.mesh.revision == mesh.revision && g.mesh.positions == mesh.positions &&
                    g.mesh.exact_position_bits == mesh.exact_position_bits &&
                    g.mesh.exact_position_bps == mesh.exact_position_bps &&
                    g.mesh.normals == mesh.normals && g.mesh.uv == mesh.uv &&
                    g.mesh.colors == mesh.colors && g.mesh.tangents == mesh.tangents &&
                    g.mesh.materials == mesh.materials &&
                    g.mesh.double_sided == mesh.double_sided &&
                    g.mesh.sided_count == mesh.sided_count && g.mesh.indices == mesh.indices &&
                    g.mesh.trial_status == mesh.trial_status && g.mesh.faces == mesh.faces &&
                    g.mesh.vertices == mesh.vertices && g.storage == storage &&
                    g.coverage_only == coverage_only &&
                    g.mesh.fixed_quantization == mesh.fixed_quantization &&
                    (!mesh.fixed_quantization ||
                     (!std::memcmp(&g.mesh.quant_low, &mesh.quant_low, sizeof(Vec3)) &&
                      !std::memcmp(&g.mesh.quant_extent, &mesh.quant_extent, sizeof(Vec3))))) {
                    found = &g;
                    break;
                }

            bool repack = !found;
            if (!found) {
                do {
                    found = &geometries[next_geometry++ % geometries.size()];
                } while (std::find(geometry.begin(), geometry.begin() + slot, found) !=
                         geometry.begin() + slot);
            }
            auto& g = *found;
            geometry[slot] = found;
            repacked += repack;
            if (repack) {
                g.layout = draw_layout(mesh, storage, coverage_only);
                auto l = g.layout;
                geometry_buffer(g, l.bytes);
                g.layout = l;
                g.mesh = mesh;
                g.storage = storage;
                g.coverage_only = coverage_only;
                pack_draw(mesh, storage, l, g.cuda, coverage_only);
                std::array<VkDescriptorBufferInfo, 3> buffers = {
                    VkDescriptorBufferInfo{g.buffer, l.metadata, sizeof(DrawMetadata)},
                    VkDescriptorBufferInfo{g.buffer, l.faces, size_t(mesh.faces) * 4},
                    VkDescriptorBufferInfo{g.buffer, 0, l.bytes}};
                std::array<VkWriteDescriptorSet, 3> writes{};
                for (uint32_t i = 0; i < 3; ++i) {
                    writes[i] = info<VkWriteDescriptorSet>(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET);
                    writes[i].dstSet = g.set;
                    writes[i].dstBinding = i;
                    writes[i].descriptorCount = 1;
                    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    writes[i].pBufferInfo = &buffers[i];
                }
                vkUpdateDescriptorSets(device, writes.size(), writes.data(), 0, nullptr);
            }

            status.metadata[slot] = reinterpret_cast<const DrawMetadata*>(
                static_cast<char*>(g.cuda) + g.layout.metadata);
            status.trial[slot] = mesh.trial_status;
            status.faces[slot] = mesh.faces;
            status.layers[slot] = uint32_t(slot);
        }
        if (timings)
            cucheck(cudaEventRecord(events[1], gpu::stream()));
        bool rgb = out[0].colors != nullptr, debug = out[0].debug != nullptr;
        if (coverage_only && (rgb || debug))
            throw std::invalid_argument("coverage-only draw has shading attachments");
        for (size_t i = 0; i < cameras.size(); ++i) {
            if (bool(out[i].colors) != rgb || bool(out[i].debug) != debug)
                throw std::invalid_argument("mixed Vulkan attachment batch");
            auto& t = targets[i];
            if (size > t.extent || uint64_t(size) * size * 4 < uint64_t(t.extent) * t.extent ||
                rgb != t.colors_enabled || debug != t.debug_enabled ||
                coverage_only != t.coverage_only) {
                cucheck(cudaStreamSynchronize(gpu::stream()));
                vkcheck(vkDeviceWaitIdle(device));
                release_targets(t);
                try {
                    target(t.mask, VK_FORMAT_R8_UINT, size, true);
                    if (!coverage_only)
                        target(t.attributes, VK_FORMAT_R32G32B32A32_SFLOAT, size, true);
                    if (rgb)
                        target(t.colors, VK_FORMAT_R32G32B32A32_SFLOAT, size, true);
                    if (!coverage_only)
                        target(t.depth, VK_FORMAT_D32_SFLOAT, size, false);
                    if (debug)
                        target(t.witness, VK_FORMAT_R32G32_UINT, size, true);
                    t.extent = size;
                    t.colors_enabled = rgb;
                    t.debug_enabled = debug;
                    t.coverage_only = coverage_only;
                } catch (...) {
                    release_targets(t);
                    throw;
                }
            }
            auto& g = *geometry[i];
            check_draw_clip(meshes[i], storage, g.layout, g.cuda, bounds, cameras[i], size, ss,
                            uint32_t(i));
        }
        cudaExternalSemaphoreSignalParams signal{};
        signal.params.fence.value = ++serial;
        cucheck(cudaSignalExternalSemaphoresAsync(&cuda_semaphore, &signal, 1, gpu::stream()));
        uint64_t wait_value = serial;
        vkcheck(vkResetCommandBuffer(command, 0));
        auto begin = info<VkCommandBufferBeginInfo>(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkcheck(vkBeginCommandBuffer(command, &begin));
        for (size_t i = 0; i < meshes.size(); ++i)
            if (std::find(geometry.begin(), geometry.begin() + i, geometry[i]) ==
                geometry.begin() + i)
                buffer_barrier(*geometry[i], true);
        if (timings) {
            vkCmdResetQueryPool(command, queries, 0, 2);
            vkCmdWriteTimestamp2(command, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, queries, 0);
        }
        for (size_t i = 0; i < cameras.size(); ++i) {
            auto mesh = meshes[i];
            auto& g = *geometry[i];
            auto coverage_pipeline = pipeline(storage, mesh.normals != nullptr, rgb, true, debug),
                 visibility_pipeline =
                     coverage_only ? VK_NULL_HANDLE
                                   : pipeline(storage, mesh.normals != nullptr, rgb, false, debug);
            std::array<VkBuffer, 3> buffers = {g.buffer, g.buffer, g.buffer};
            std::array<VkDeviceSize, 3> offsets = {g.layout.position, g.layout.normal,
                                                   g.layout.color};
            vkCmdBindVertexBuffers(command, 0, 3, buffers.data(), offsets.data());
            vkCmdBindIndexBuffer(command, g.buffer, g.layout.indices, VK_INDEX_TYPE_UINT32);
            vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1,
                                    &g.set, 0, nullptr);
            auto& t = targets[i];
            const auto& camera = cameras[i];
            image_barrier(t.mask, true);
            if (!coverage_only)
                image_barrier(t.attributes, true);
            if (rgb)
                image_barrier(t.colors, true);
            if (!coverage_only)
                image_barrier(t.depth, true, true);
            if (debug)
                image_barrier(t.witness, true);
            double near = std::max(bounds.radius * .01, camera.distance - bounds.radius * 2),
                   far = camera.distance + bounds.radius * 2;
            Push p{};
            auto vec = [](float* out, Vec3 v, float w) {
                out[0] = v.x;
                out[1] = v.y;
                out[2] = v.z;
                out[3] = w;
            };
            float scale = float((camera.perspective ? camera.focal : camera.scale) * ss * 2 / size);
            vec(p.right, camera.right, scale);
            vec(p.up, camera.up, scale);
            vec(p.forward, camera.forward, float(camera.distance));
            vec(p.center, bounds.center, float(near));
            p.depth[0] = float(far / (far - near));
            p.depth[1] = float(-far * near / (far - near));
            p.depth[2] = float(1 / (far - near));
            p.depth[3] = float(1 / bounds.diameter());
            p.options[0] = storage != NeuralVertexStorage::Float32;
            p.options[1] = camera.perspective;
            p.options[2] = two;
            vkCmdPushConstants(command, pipeline_layout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                               sizeof(p), &p);
            VkViewport viewport{0, 0, float(size), float(size), 0, 1};
            VkRect2D rect{{0, 0}, {size, size}};
            vkCmdSetViewport(command, 0, 1, &viewport);
            vkCmdSetScissor(command, 0, 1, &rect);
            for (bool coverage : {true, false}) {
                if (!coverage && !coverage_only)
                    ++measured.visibility_passes;
                if (coverage)
                    ++measured.coverage_passes;
                if (!coverage && coverage_only)
                    break;
                std::array<VkRenderingAttachmentInfo, 3> attachments{};
                for (auto& a : attachments) {
                    a = info<VkRenderingAttachmentInfo>(
                        VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO);
                    a.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                    a.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                    a.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                }
                attachments[0].imageView = coverage ? t.mask.view : t.attributes.view;
                attachments[1].imageView = t.colors.view;
                attachments[2].imageView = t.witness.view;
                auto da =
                    info<VkRenderingAttachmentInfo>(VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO);
                da.imageView = t.depth.view;
                da.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
                da.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                da.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                da.clearValue.depthStencil.depth = 1;
                auto rendering = info<VkRenderingInfo>(VK_STRUCTURE_TYPE_RENDERING_INFO);
                rendering.renderArea = rect;
                rendering.layerCount = 1;
                rendering.colorAttachmentCount = coverage ? 1 : debug ? 3 : 2;
                rendering.pColorAttachments = attachments.data();
                rendering.pDepthAttachment = coverage ? nullptr : &da;
                vkCmdBeginRendering(command, &rendering);
                vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                  coverage ? coverage_pipeline : visibility_pipeline);
                vkCmdDrawIndexedIndirect(command, g.buffer, g.layout.indirect, 1,
                                         5 * sizeof(uint32_t));
                vkCmdEndRendering(command);
            }
            image_barrier(t.mask, false);
            if (!coverage_only)
                image_barrier(t.attributes, false);
            if (rgb)
                image_barrier(t.colors, false);
            if (debug)
                image_barrier(t.witness, false);
        }
        if (timings)
            vkCmdWriteTimestamp2(command, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, queries, 1);
        for (size_t i = 0; i < meshes.size(); ++i)
            if (std::find(geometry.begin(), geometry.begin() + i, geometry[i]) ==
                geometry.begin() + i)
                buffer_barrier(*geometry[i], false);
        vkcheck(vkEndCommandBuffer(command));
        auto wait = info<VkSemaphoreSubmitInfo>(VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO);
        wait.semaphore = semaphore;
        wait.value = wait_value;
        wait.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        auto done = wait;
        done.value = ++serial;
        auto command_info =
            info<VkCommandBufferSubmitInfo>(VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO);
        command_info.commandBuffer = command;
        auto submit = info<VkSubmitInfo2>(VK_STRUCTURE_TYPE_SUBMIT_INFO_2);
        submit.waitSemaphoreInfoCount = 1;
        submit.pWaitSemaphoreInfos = &wait;
        submit.commandBufferInfoCount = 1;
        submit.pCommandBufferInfos = &command_info;
        submit.signalSemaphoreInfoCount = 1;
        submit.pSignalSemaphoreInfos = &done;
        vkcheck(vkQueueSubmit2(queue, 1, &submit, completion));
        submitted = true;
        cudaExternalSemaphoreWaitParams cuda_wait{};
        cuda_wait.params.fence.value = serial;
        cucheck(cudaWaitExternalSemaphoresAsync(&cuda_semaphore, &cuda_wait, 1, gpu::stream()));
        if (timings)
            cucheck(cudaEventRecord(events[2], gpu::stream()));
        for (size_t i = 0; i < cameras.size(); ++i) {
            auto& t = targets[i];
            out[i].surfaces = {t.mask.surface, t.attributes.surface, t.colors.surface, size};
            if (out[i].pixels)
                unpack_hardware(t.mask.surface, t.attributes.surface, t.colors.surface,
                                out[i].pixels, out[i].colors, size, t.witness.surface,
                                out[i].debug);
            if (out[i].mask_bits)
                pack_coverage_mask(t.mask.surface, out[i].mask_bits, size);
        }
        if (timings)
            cucheck(cudaEventRecord(events[3], gpu::stream()));
        auto* collected = reinterpret_cast<uint32_t*>(static_cast<char*>(geometry[0]->cuda) +
                                                      geometry[0]->layout.status);
        collect_draw_status(status, collected);
        uint32_t flags[10]{};
        cucheck(gpu::copy(flags, collected, (cameras.size() + 2) * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
        if (flags[0]) {
            std::string reason = "invalid draw";
            for (size_t i = 0; i < cameras.size(); ++i)
                if (flags[0] & (1u << i)) {
                    DrawMetadata metadata;
                    cucheck(gpu::copy(&metadata, status.metadata[i], sizeof(metadata),
                                      cudaMemcpyDeviceToHost));
                    reason += "; lane " + std::to_string(i) + ":";
                    if (metadata.invalid & DrawDomain)
                        reason += " invalid quantization domain";
                    if (metadata.invalid & DrawPrecisionCap)
                        reason += " exact-position cap exceeded";
                    if (metadata.invalid & DrawPosition)
                        reason += " position outside fixed bounds";
                    if (metadata.invalid & DrawUv)
                        reason += " UV outside [-8,8]";
                    if (metadata.invalid & DrawNonfinite)
                        reason += " nonfinite position";
                }
            throw std::invalid_argument(reason);
        }
        for (size_t i = 0; i < cameras.size(); ++i) {
            out[i].clipped = (flags[1] & (1u << i)) != 0;
            out[i].faces = flags[2 + i];
        }

        if (timings) {
            float ms = 0;
            if (repacked) {
                cucheck(cudaEventElapsedTime(&ms, events[0], events[1]));
                measured.packing_seconds += ms * .001;
                measured.packed_meshes += repacked;
            }
            cucheck(cudaEventElapsedTime(&ms, events[2], events[3]));
            measured.unpack_seconds += ms * .001;
            uint64_t ticks[2]{};
            vkcheck(vkGetQueryPoolResults(device, queries, 0, 2, sizeof(ticks), ticks,
                                          sizeof(uint64_t),
                                          VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
            measured.render_seconds +=
                (ticks[1] - ticks[0]) * double(properties.limits.timestampPeriod) * 1e-9;
            measured.draws += cameras.size();
        }
    }
    void destroy() noexcept {
        if (device) {
            cudaStreamSynchronize(gpu::stream());
            vkDeviceWaitIdle(device);
            for (auto event : events)
                if (event)
                    cudaEventDestroy(event);
            if (queries)
                vkDestroyQueryPool(device, queries, nullptr);
            for (auto& g : geometries)
                drop(g);
            for (auto& t : targets)
                release_targets(t);
            for (auto p : pipelines)
                if (p)
                    vkDestroyPipeline(device, p, nullptr);
            if (pipeline_layout)
                vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
            if (descriptors)
                vkDestroyDescriptorPool(device, descriptors, nullptr);
            if (set_layout)
                vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
            if (commands)
                vkDestroyCommandPool(device, commands, nullptr);
            if (cuda_semaphore)
                cudaDestroyExternalSemaphore(cuda_semaphore);
            if (semaphore)
                vkDestroySemaphore(device, semaphore, nullptr);
            if (completion)
                vkDestroyFence(device, completion, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        if (instance)
            vkDestroyInstance(instance, nullptr);
    }
};
VulkanRaster::VulkanRaster(const NeuralOptions& o, bool timings, MemoryBudget* budget)
    : impl_(std::make_unique<Impl>(o, timings, budget)) {}
VulkanRaster::~VulkanRaster() = default;
void VulkanRaster::trim_targets(uint32_t keep) {
    cucheck(cudaStreamSynchronize(gpu::stream()));
    vkcheck(vkDeviceWaitIdle(impl_->device));
    for (size_t i = keep; i < impl_->targets.size(); ++i)
        impl_->release_targets(impl_->targets[i]);
}
RasterSurfaces VulkanRaster::surfaces() const {
    auto& t = impl_->targets[0];
    return {t.mask.surface, t.attributes.surface, t.colors.surface, t.extent};
}
bool VulkanRaster::render(DeviceMeshView m, const Bounds& b, const Camera& c, double s, uint8_t ss,
                          bool two, NeuralVertexStorage storage, AuditPixel* p, Vec3* rgb,
                          RasterDebugPixel* debug, bool coverage_only, uint32_t* mask_bits) {
    RasterOutput out{p, rgb, debug};
    out.mask_bits = mask_bits;
    impl_->render_batch({&m, 1}, b, {&c, 1}, s, ss, two, storage, {&out, 1}, coverage_only);
    return out.clipped;
}
void VulkanRaster::render_batch(DeviceMeshView m, const Bounds& b, std::span<const Camera> c,
                                double s, uint8_t ss, bool two, NeuralVertexStorage storage,
                                std::span<RasterOutput> out, bool coverage_only) {
    std::array<DeviceMeshView, 4> meshes;
    meshes.fill(m);
    if (c.size() > meshes.size())
        throw std::invalid_argument("Vulkan view batch size");
    impl_->render_batch({meshes.data(), c.size()}, b, c, s, ss, two, storage, out, coverage_only);
}
void VulkanRaster::render_candidates(std::span<const DeviceMeshView> meshes, const Bounds& b,
                                     const Camera& c, double s, uint8_t ss, bool two,
                                     NeuralVertexStorage storage, std::span<RasterOutput> out,
                                     bool coverage_only) {
    std::array<Camera, 8> cameras;
    cameras.fill(c);
    if (meshes.size() > cameras.size())
        throw std::invalid_argument("Vulkan candidate batch size");
    impl_->render_batch(meshes, b, {cameras.data(), meshes.size()}, s, ss, two, storage, out,
                        coverage_only);
}
size_t VulkanRaster::bytes() const {
    return impl_->allocated;
}
VulkanTiming VulkanRaster::timing() const {
    return impl_->measured;
}
} // namespace blitz::neural
