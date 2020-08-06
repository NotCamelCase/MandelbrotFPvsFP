#pragma once

#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan\vulkan.h>

#include "Common.h"

namespace gpu
{
#define DISPATCH_LOCALWORKGROUP_SIZE 16

#ifndef NDEBUG
#define VK_CHECK(x) do { VkResult res = x; assert(res == VK_SUCCESS); } while(false)
#else
#define VK_CHECK(x) x
#endif

    // Find the appropriate memory allocation type based on flags passed by app
    uint32_t FindMemoryType(const uint32_t deviceReq, VkMemoryPropertyFlags memoryFlags, const VkPhysicalDeviceMemoryProperties& memoryProps)
    {
        for (unsigned i = 0; i < memoryProps.memoryTypeCount; i++)
        {
            if (deviceReq & (1u << i))
            {
                if ((memoryProps.memoryTypes[i].propertyFlags & memoryFlags) == memoryFlags)
                {
                    return i;
                }
            }
        }

        assert(false);
        return ~0u;
    }

    struct ComputeContext
    {
        VkInstance                  m_Instance;
        VkDebugUtilsMessengerEXT    m_DebugUtils;

        VkPhysicalDevice            m_PhysicalDevice;
        VkDevice                    m_Device;
        VkQueue                     m_ComputeQueue;
        uint32_t                    m_QueueFamilyIdx;

        VkCommandPool               m_CmdPool;
        VkCommandBuffer             m_CmdBuffer;

        VkPipeline                  m_Pipeline;
        VkPipelineLayout            m_Layout;
        VkDescriptorPool            m_DescriptorPool;
        VkDescriptorSetLayout       m_DescriptorLayout;
        VkDescriptorSet             m_DescriptorSet;
        VkShaderModule              m_ShaderModule;

        VkImage                     m_StorageImage;
        VkDeviceMemory              m_StorageImageMemory;
        VkImageView                 m_StorageImageView;

        VkBuffer                    m_ResultBuffer;
        VkDeviceMemory              m_ResultBufferMemory;
    };

    VKAPI_ATTR VkBool32 VKAPI_CALL debug_messenger_callback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData)
    {
        switch (messageSeverity)
        {
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
            printf("%s\n", pCallbackData->pMessage);
            break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
            printf("%s\n", pCallbackData->pMessage);
            assert(false);
            break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
            break;
        default:
            break;
        }

        // Don't bail out, but keep going.
        return false;
    }

    bool InitVulkanInstance(ComputeContext* pContext)
    {
        VkApplicationInfo appInfo = {};
        appInfo.pNext = nullptr;
        appInfo.pEngineName = "GI_Basic";
        appInfo.pApplicationName = "GI_Basic";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_2;

#ifdef _DEBUG
        VkDebugUtilsMessengerCreateInfoEXT debugInfo = { VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
        debugInfo.pNext = nullptr;
        debugInfo.flags = 0;
        debugInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debugInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
        debugInfo.pfnUserCallback = debug_messenger_callback;
        debugInfo.pUserData = pContext;
#endif

        const char* enabledExts = { VK_EXT_DEBUG_UTILS_EXTENSION_NAME };
        const char* enabledLayers = { "VK_LAYER_KHRONOS_validation" };

        VkInstanceCreateInfo instanceInfo = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
#ifdef _DEBUG
        instanceInfo.pNext = &debugInfo;
        instanceInfo.enabledExtensionCount = 1;
        instanceInfo.ppEnabledExtensionNames = &enabledExts;
        instanceInfo.enabledLayerCount = 1;
        instanceInfo.ppEnabledLayerNames = &enabledLayers;
#else
        instanceInfo.enabledExtensionCount = 0;
        instanceInfo.ppEnabledExtensionNames = nullptr;
        instanceInfo.enabledLayerCount = 0;
        instanceInfo.ppEnabledLayerNames = nullptr;
#endif
        instanceInfo.flags = VkInstanceCreateFlags(0);
        instanceInfo.pApplicationInfo = &appInfo;

        VK_CHECK(vkCreateInstance(&instanceInfo, nullptr, &pContext->m_Instance));
        assert(pContext->m_Instance != nullptr);

#ifdef _DEBUG
        PFN_vkCreateDebugUtilsMessengerEXT CreateDebugUtilsMessengerEXT =
            reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(pContext->m_Instance, "vkCreateDebugUtilsMessengerEXT"));
        assert(CreateDebugUtilsMessengerEXT != nullptr);

        VK_CHECK(CreateDebugUtilsMessengerEXT(pContext->m_Instance, &debugInfo, nullptr, &pContext->m_DebugUtils));
#endif

        return pContext->m_Instance != nullptr;
    }

    bool InitVulkanDevice(ComputeContext* pContext, bool useDiscreteGPU)
    {
        uint32_t numPhysicalDevices = 0u;
        VK_CHECK(vkEnumeratePhysicalDevices(pContext->m_Instance, &numPhysicalDevices, nullptr));

        assert(numPhysicalDevices > 0);

        std::vector<VkPhysicalDevice> physicalDevices(numPhysicalDevices);
        VK_CHECK(vkEnumeratePhysicalDevices(pContext->m_Instance, &numPhysicalDevices, physicalDevices.data()));

        VkPhysicalDevice physicalDevice = nullptr;

        for (size_t i = 0; i < numPhysicalDevices; i++)
        {
            VkPhysicalDevice physicalDeviceHandle = physicalDevices[i];
            assert(physicalDeviceHandle != nullptr);

            VkPhysicalDeviceProperties props = {};
            vkGetPhysicalDeviceProperties(physicalDeviceHandle, &props);

            if ((useDiscreteGPU && (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)) ||
                (!useDiscreteGPU && (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)))
            {
                physicalDevice = physicalDeviceHandle;
                break;
            }
        }

        pContext->m_PhysicalDevice = physicalDevice;

        uint32_t numQueueFamilyProps = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &numQueueFamilyProps, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilyProps(numQueueFamilyProps);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &numQueueFamilyProps, queueFamilyProps.data());

        for (uint32_t i = 0; i < numQueueFamilyProps; i++)
        {
            auto props = queueFamilyProps[i];
            if ((props.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0)
            {
                // Use the first queue capable of compute work
                pContext->m_QueueFamilyIdx = i;
            }
        }

        float queuePriority = { 1.f };

        VkDeviceQueueCreateInfo queueInfo = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
        queueInfo.pNext = nullptr;
        queueInfo.queueCount = 1;
        queueInfo.queueFamilyIndex = pContext->m_QueueFamilyIdx;
        queueInfo.pQueuePriorities = &queuePriority;
        queueInfo.flags = VkDeviceQueueCreateFlags(0);

        // Check support for shaderInt64
        VkPhysicalDeviceFeatures2 deviceFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
        vkGetPhysicalDeviceFeatures2(physicalDevice, &deviceFeatures);

        if (deviceFeatures.features.shaderInt64 != VK_TRUE)
        {
            // Require shaderInt64 support
            throw;
        }

        VkDeviceCreateInfo deviceInfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
        deviceInfo.pNext = nullptr;
        deviceInfo.enabledExtensionCount = 0;
        deviceInfo.enabledLayerCount = 0;
        deviceInfo.pEnabledFeatures = &deviceFeatures.features;
        deviceInfo.flags = VkDeviceCreateFlags(0);
        deviceInfo.ppEnabledExtensionNames = nullptr;
        deviceInfo.ppEnabledLayerNames = nullptr;
        deviceInfo.queueCreateInfoCount = 1; // Use single queue capable of compute work
        deviceInfo.pQueueCreateInfos = &queueInfo;
        VK_CHECK(vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &pContext->m_Device));

        vkGetDeviceQueue(pContext->m_Device, pContext->m_QueueFamilyIdx, 0, &pContext->m_ComputeQueue);
        assert(pContext->m_ComputeQueue != nullptr);

        return pContext->m_Device != nullptr;
    }

    void InitCommandBuffer(ComputeContext* pContext)
    {
        VkCommandPoolCreateInfo commandPoolInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        commandPoolInfo.pNext = nullptr;
        commandPoolInfo.queueFamilyIndex = pContext->m_QueueFamilyIdx;

        VK_CHECK(vkCreateCommandPool(pContext->m_Device, &commandPoolInfo, nullptr, &pContext->m_CmdPool));
        assert(pContext->m_CmdPool != nullptr);

        VkCommandBufferAllocateInfo commandBufferInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        commandBufferInfo.pNext = nullptr;
        commandBufferInfo.commandPool = pContext->m_CmdPool;
        commandBufferInfo.commandBufferCount = 1;
        commandBufferInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        VK_CHECK(vkAllocateCommandBuffers(pContext->m_Device, &commandBufferInfo, &pContext->m_CmdBuffer));
        assert(pContext->m_CmdBuffer != nullptr);
    }

    void CreateStorageImage(ComputeContext* pContext, const Params& params)
    {
        VkImageCreateInfo imageInfo = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        imageInfo.pNext = nullptr;
        imageInfo.flags = 0;
        imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT; // Used in CS to store calculated values
        imageInfo.format = VK_FORMAT_R32_UINT; // Corresponds to rgba32ui in CS
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.arrayLayers = 1;
        imageInfo.extent.width = params.m_Width;
        imageInfo.extent.height = params.m_Height;
        imageInfo.extent.depth = 1;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.mipLevels = 1;
        imageInfo.pQueueFamilyIndices = &pContext->m_QueueFamilyIdx;
        imageInfo.queueFamilyIndexCount = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // Used in single queue
        VK_CHECK(vkCreateImage(pContext->m_Device, &imageInfo, nullptr, &pContext->m_StorageImage));

        VkMemoryRequirements memreq = {};
        vkGetImageMemoryRequirements(pContext->m_Device, pContext->m_StorageImage, &memreq);

        VkPhysicalDeviceMemoryProperties memprops = {};
        vkGetPhysicalDeviceMemoryProperties(pContext->m_PhysicalDevice, &memprops);

        uint32_t memType = FindMemoryType(memreq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memprops);

        VkMemoryAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        allocInfo.pNext = nullptr;
        allocInfo.allocationSize = memreq.size;
        allocInfo.memoryTypeIndex = memType;
        VK_CHECK(vkAllocateMemory(pContext->m_Device, &allocInfo, nullptr, &pContext->m_StorageImageMemory));

        VK_CHECK(vkBindImageMemory(pContext->m_Device, pContext->m_StorageImage, pContext->m_StorageImageMemory, 0));

        VkImageSubresourceRange range = {};
        range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        range.baseArrayLayer = 0;
        range.baseMipLevel = 0;
        range.layerCount = 1;
        range.levelCount = 1;

        VkImageViewCreateInfo imageviewInfo = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        imageviewInfo.pNext = nullptr;
        imageviewInfo.flags = 0;
        imageviewInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        imageviewInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        imageviewInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        imageviewInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        imageviewInfo.format = VK_FORMAT_R32_UINT;
        imageviewInfo.image = pContext->m_StorageImage;
        imageviewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        imageviewInfo.subresourceRange = range;
        VK_CHECK(vkCreateImageView(pContext->m_Device, &imageviewInfo, nullptr, &pContext->m_StorageImageView));
    }

    void InitDescriptorSet(ComputeContext* pContext)
    {
        VkDescriptorSetLayoutBinding binding = {};
        binding.binding = 0;
        binding.descriptorCount = 1;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        binding.pImmutableSamplers = nullptr;
        binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo setLayoutInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        setLayoutInfo.pNext = nullptr;
        setLayoutInfo.flags = 0;
        setLayoutInfo.bindingCount = 1;
        setLayoutInfo.pBindings = &binding;
        VK_CHECK(vkCreateDescriptorSetLayout(pContext->m_Device, &setLayoutInfo, nullptr, &pContext->m_DescriptorLayout));
        assert(pContext->m_DescriptorLayout != nullptr);

        VkDescriptorPoolSize poolSize = {
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            1
        };

        VkDescriptorPoolCreateInfo poolInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.pNext = nullptr;
        poolInfo.flags = 0;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        poolInfo.maxSets = 1;
        VK_CHECK(vkCreateDescriptorPool(pContext->m_Device, &poolInfo, nullptr, &pContext->m_DescriptorPool));
        assert(pContext->m_DescriptorPool != nullptr);

        VkPipelineLayoutCreateInfo layoutInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        layoutInfo.pNext = nullptr;
        layoutInfo.flags = 0;
        layoutInfo.pPushConstantRanges = nullptr;
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pushConstantRangeCount = 0;
        layoutInfo.pSetLayouts = &pContext->m_DescriptorLayout;
        VK_CHECK(vkCreatePipelineLayout(pContext->m_Device, &layoutInfo, nullptr, &pContext->m_Layout));

        VkDescriptorSetAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        allocInfo.pNext = nullptr;
        allocInfo.pSetLayouts = &pContext->m_DescriptorLayout;
        allocInfo.descriptorPool = pContext->m_DescriptorPool;
        allocInfo.descriptorSetCount = 1;
        VK_CHECK(vkAllocateDescriptorSets(pContext->m_Device, &allocInfo, &pContext->m_DescriptorSet));

        VkDescriptorImageInfo imageInfo = {};
        imageInfo.sampler = nullptr;
        imageInfo.imageView = pContext->m_StorageImageView;
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet writeDescSet = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writeDescSet.pNext = nullptr;
        writeDescSet.pBufferInfo = nullptr;
        writeDescSet.pImageInfo = &imageInfo;
        writeDescSet.descriptorCount = 1;
        writeDescSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writeDescSet.dstSet = pContext->m_DescriptorSet;
        writeDescSet.dstBinding = 0;
        vkUpdateDescriptorSets(pContext->m_Device, 1, &writeDescSet, 0, nullptr);
    }

    void CreatePipeline(ComputeContext* pContext, const char* shaderFileName)
    {
        std::vector<char> source;
        {
            // Open a file at its end
            std::ifstream file(shaderFileName, std::ifstream::binary | std::ifstream::ate | std::ifstream::in);
            assert(file.is_open());

            // Determine content's length
            size_t length = (size_t)file.tellg();
            source.resize(length);

            // Go to the beginning and read whole file content
            file.seekg(0, file.beg);
            file.read(source.data(), length);
            file.close();
        }

        VkShaderModuleCreateInfo moduleInfo = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        moduleInfo.pNext = nullptr;
        moduleInfo.flags = 0;
        moduleInfo.codeSize = source.size();
        moduleInfo.pCode = reinterpret_cast<const uint32_t*> (source.data());
        VK_CHECK(vkCreateShaderModule(pContext->m_Device, &moduleInfo, nullptr, &pContext->m_ShaderModule));

        VkPipelineShaderStageCreateInfo shaderInfo = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
        shaderInfo.pNext = nullptr;
        shaderInfo.flags = 0;
        shaderInfo.pSpecializationInfo = nullptr;
        shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        shaderInfo.pName = "main";
        shaderInfo.module = pContext->m_ShaderModule;

        VkComputePipelineCreateInfo pipeInfo = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        pipeInfo.pNext = nullptr;
        pipeInfo.flags = 0;
        pipeInfo.stage = shaderInfo;
        pipeInfo.basePipelineIndex = ~0ul;
        pipeInfo.basePipelineHandle = VK_NULL_HANDLE;
        pipeInfo.layout = pContext->m_Layout;
        VK_CHECK(vkCreateComputePipelines(pContext->m_Device, nullptr, 1, &pipeInfo, nullptr, &pContext->m_Pipeline));
        assert(pContext->m_Pipeline != nullptr);
    }

    void ConvertImageLayoutToGeneral(ComputeContext* pContext)
    {
        VkImageMemoryBarrier imageBarrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        imageBarrier.pNext = nullptr;
        imageBarrier.srcQueueFamilyIndex = imageBarrier.dstQueueFamilyIndex = pContext->m_QueueFamilyIdx;
        imageBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        imageBarrier.image = pContext->m_StorageImage;
        imageBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        imageBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        imageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageBarrier.subresourceRange.baseArrayLayer = 0;
        imageBarrier.subresourceRange.baseMipLevel = 0;
        imageBarrier.subresourceRange.layerCount = imageBarrier.subresourceRange.levelCount = 1;

        vkCmdPipelineBarrier(
            pContext->m_CmdBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0,
            nullptr,
            0,
            nullptr,
            1,
            &imageBarrier);
    }

    void DispatchGPGPU(ComputeContext* pContext, const Params& params)
    {
        // Dispatch GPGPU command

        vkCmdBindPipeline(pContext->m_CmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pContext->m_Pipeline);
        vkCmdBindDescriptorSets(pContext->m_CmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pContext->m_Layout, 0, 1, &pContext->m_DescriptorSet, 0, nullptr);

        vkCmdDispatch(pContext->m_CmdBuffer, params.m_Width / DISPATCH_LOCALWORKGROUP_SIZE, params.m_Height / DISPATCH_LOCALWORKGROUP_SIZE, 1);
    }

    void FetchStorageImageContents(ComputeContext* pContext, const Params& params)
    {
        // Prepare buffer to copy the storage image to

        VkBufferCreateInfo bufferInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bufferInfo.pNext = nullptr;
        bufferInfo.flags = 0;
        bufferInfo.queueFamilyIndexCount = 1;
        bufferInfo.pQueueFamilyIndices = &pContext->m_QueueFamilyIdx;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufferInfo.size = params.m_Width * params.m_Height * sizeof(uint32_t);
        VK_CHECK(vkCreateBuffer(pContext->m_Device, &bufferInfo, nullptr, &pContext->m_ResultBuffer));

        VkMemoryRequirements memreq = {};
        vkGetBufferMemoryRequirements(pContext->m_Device, pContext->m_ResultBuffer, &memreq);

        VkPhysicalDeviceMemoryProperties memprops = {};
        vkGetPhysicalDeviceMemoryProperties(pContext->m_PhysicalDevice, &memprops);

        uint32_t memType = FindMemoryType(memreq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memprops);

        VkMemoryAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        allocInfo.pNext = nullptr;
        allocInfo.allocationSize = memreq.size;
        allocInfo.memoryTypeIndex = memType;
        VK_CHECK(vkAllocateMemory(pContext->m_Device, &allocInfo, nullptr, &pContext->m_ResultBufferMemory));

        VK_CHECK(vkBindBufferMemory(pContext->m_Device, pContext->m_ResultBuffer, pContext->m_ResultBufferMemory, 0));

        // Stall the copy until compute is done
        VkImageMemoryBarrier imageMemoryBarrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        imageMemoryBarrier.image = pContext->m_StorageImage;
        imageMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        imageMemoryBarrier.srcQueueFamilyIndex = imageMemoryBarrier.dstQueueFamilyIndex = pContext->m_QueueFamilyIdx;
        imageMemoryBarrier.oldLayout = imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        imageMemoryBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageMemoryBarrier.subresourceRange.baseArrayLayer = 0;
        imageMemoryBarrier.subresourceRange.baseMipLevel = 0;
        imageMemoryBarrier.subresourceRange.layerCount = 1;
        imageMemoryBarrier.subresourceRange.levelCount = 1;
        vkCmdPipelineBarrier(pContext->m_CmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageMemoryBarrier);

        // Copy storage image contents from the device-local memory to host-visible buffer

        VkBufferImageCopy copyRegion = {};
        copyRegion.imageExtent.width = params.m_Width;
        copyRegion.imageExtent.height = params.m_Height;
        copyRegion.imageExtent.depth = 1;
        copyRegion.imageOffset = {};
        copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegion.imageSubresource.baseArrayLayer = 0;
        copyRegion.imageSubresource.layerCount = 1;
        copyRegion.imageSubresource.mipLevel = 0;
        copyRegion.bufferOffset = 0;
        copyRegion.bufferRowLength = 0;
        copyRegion.bufferImageHeight = 0;
        vkCmdCopyImageToBuffer(pContext->m_CmdBuffer, pContext->m_StorageImage, VK_IMAGE_LAYOUT_GENERAL, pContext->m_ResultBuffer, 1, &copyRegion);

        // Barrier to ensure writes to the buffer are made visible on the host
        VkBufferMemoryBarrier bufferMemoryBarrier = { VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER };
        bufferMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        bufferMemoryBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        bufferMemoryBarrier.dstQueueFamilyIndex = bufferMemoryBarrier.srcQueueFamilyIndex = pContext->m_QueueFamilyIdx;
        bufferMemoryBarrier.buffer = pContext->m_ResultBuffer;
        bufferMemoryBarrier.size = bufferInfo.size;
        vkCmdPipelineBarrier(pContext->m_CmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &bufferMemoryBarrier, 0, nullptr);
    }

    void CleanUp(ComputeContext* pContext)
    {
        vkDeviceWaitIdle(pContext->m_Device);

        vkDestroyDescriptorSetLayout(pContext->m_Device, pContext->m_DescriptorLayout, nullptr);
        vkDestroyPipelineLayout(pContext->m_Device, pContext->m_Layout, nullptr);
        vkDestroyShaderModule(pContext->m_Device, pContext->m_ShaderModule, nullptr);
        vkDestroyPipeline(pContext->m_Device, pContext->m_Pipeline, nullptr);

        vkDestroyDescriptorPool(pContext->m_Device, pContext->m_DescriptorPool, nullptr);

        vkFreeCommandBuffers(pContext->m_Device, pContext->m_CmdPool, 1, &pContext->m_CmdBuffer);
        vkDestroyCommandPool(pContext->m_Device, pContext->m_CmdPool, nullptr);

        vkDestroyImage(pContext->m_Device, pContext->m_StorageImage, nullptr);
        vkDestroyImageView(pContext->m_Device, pContext->m_StorageImageView, nullptr);
        vkFreeMemory(pContext->m_Device, pContext->m_StorageImageMemory, nullptr);

        vkDestroyBuffer(pContext->m_Device, pContext->m_ResultBuffer, nullptr);
        vkFreeMemory(pContext->m_Device, pContext->m_ResultBufferMemory, nullptr);

        vkDestroyDevice(pContext->m_Device, nullptr);

        if (pContext->m_DebugUtils != nullptr)
        {
            PFN_vkDestroyDebugUtilsMessengerEXT DestroyDebugUtilsMessengerEXT =
                reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(pContext->m_Instance, "vkDestroyDebugUtilsMessengerEXT"));
            assert(DestroyDebugUtilsMessengerEXT != nullptr);

            DestroyDebugUtilsMessengerEXT(pContext->m_Instance, pContext->m_DebugUtils, nullptr);
        }

        vkDestroyInstance(pContext->m_Instance, nullptr);
    }

    // Calculate Mandelbrot set for each pixel using GPU native floating-point
    void RenderMandelbrot(const Params& params, const char* shaderFileName, const char* outputFileName, bool renderResults)
    {
        ComputeContext computeContext = {};

        if (!InitVulkanInstance(&computeContext))
        {
            assert(false && "Failed to initialize Vulkan instance!");
        }

        if (!InitVulkanDevice(&computeContext, params.m_UseDiscreteGPU))
        {
            assert(false && "Failed to initialize Vulkan device!");
        }

        InitCommandBuffer(&computeContext);

        VkCommandBufferBeginInfo cmdBeginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        cmdBeginInfo.pNext = nullptr;
        cmdBeginInfo.flags = 0;
        cmdBeginInfo.pInheritanceInfo = nullptr;
        VK_CHECK(vkBeginCommandBuffer(computeContext.m_CmdBuffer, &cmdBeginInfo));

        CreateStorageImage(&computeContext, params);
        InitDescriptorSet(&computeContext);
        CreatePipeline(&computeContext, shaderFileName);
        ConvertImageLayoutToGeneral(&computeContext);
        DispatchGPGPU(&computeContext, params);
        FetchStorageImageContents(&computeContext, params);

        VK_CHECK(vkEndCommandBuffer(computeContext.m_CmdBuffer));

        VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        submitInfo.pNext = nullptr;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &computeContext.m_CmdBuffer;
        submitInfo.waitSemaphoreCount = 0;
        submitInfo.pWaitSemaphores = nullptr;
        submitInfo.signalSemaphoreCount = 0;
        submitInfo.pSignalSemaphores = nullptr;
        submitInfo.pWaitDstStageMask = nullptr;

        VkFence doneFence = nullptr;
        VkFenceCreateInfo fenceInfo = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        VK_CHECK(vkCreateFence(computeContext.m_Device, &fenceInfo, nullptr, &doneFence));

        // Start timing execution
        auto begin = std::chrono::high_resolution_clock::now();
        VK_CHECK(vkQueueSubmit(computeContext.m_ComputeQueue, 1, &submitInfo, doneFence));
        vkWaitForFences(computeContext.m_Device, 1, &doneFence, true, UINT64_MAX);
        // End timing
        auto end = std::chrono::high_resolution_clock::now();

        vkDestroyFence(computeContext.m_Device, doneFence, nullptr);

        std::chrono::duration<double, std::milli> diff = end - begin;
        std::cout << "\tRuntime: " << diff.count() << " ms" << std::endl;

        if (renderResults)
        {
            // Operation completed, render results if needed
            void* mappedResultsBufferAddress = nullptr;
            VK_CHECK(vkMapMemory(computeContext.m_Device, computeContext.m_ResultBufferMemory, 0, VK_WHOLE_SIZE, 0, &mappedResultsBufferAddress));
            assert(mappedResultsBufferAddress != nullptr);
            OutputFrame(reinterpret_cast<uint32_t*>(mappedResultsBufferAddress), outputFileName, params);
            vkUnmapMemory(computeContext.m_Device, computeContext.m_ResultBufferMemory);
        }

        CleanUp(&computeContext);
    }
}