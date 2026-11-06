#include "graphics_internal.hpp"

#include <iostream>
#include <vector>

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <VkBootstrap.h>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4100 4189 4324)
#endif
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <backends/imgui_impl_vulkan.h>

namespace graphics::internal {

namespace {

VkInstance vk_instance;
uint32_t vk_api_version;
VkSurfaceKHR vk_surface;

VkSwapchainKHR vk_swapchain;
std::vector<VkImage> vk_swapchain_images;
std::vector<VkImageView> vk_swapchain_image_views;
uint32_t vk_swapchain_current_image;

uint32_t vk_swapchain_resize_width;
uint32_t vk_swapchain_resize_height;
bool vk_swapchain_resize_require;

VkFormat vk_depth_buffer_format = VK_FORMAT_UNDEFINED;
VkImage vk_image_depth_buffer;
VmaAllocation vma_allocation_depth_buffer;
VkImageView vk_image_view_depth_buffer;

std::vector<VkFramebuffer> vk_framebuffers;

VkSemaphore vk_semaphore_image_available;
std::vector<VkSemaphore> vk_semaphores_image_finished;
VkFence vk_fence_frame_in_flight;

VkCommandPool vk_command_pool;
VkCommandBuffer vk_command_buffer;

VkDescriptorPool vk_imgui_descriptor_pool;

VkFormat selectDepthFormat(VkPhysicalDevice physical_device) {
    const VkFormat candidates[] = {
        VK_FORMAT_D32_SFLOAT_S8_UINT,
        VK_FORMAT_D24_UNORM_S8_UINT,
        VK_FORMAT_D32_SFLOAT
    };

    for (VkFormat format : candidates) {
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(physical_device, format, &properties);
        if (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
            return format;
        }
    }

    return VK_FORMAT_UNDEFINED;
}

bool initializeImGUI() {
    const VkDescriptorPoolSize descriptor_pool_sizes[] = {
        { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 }
    };

    const VkDescriptorPoolCreateInfo descriptor_pool = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets = 1000,
        .poolSizeCount = static_cast<uint32_t>(sizeof(descriptor_pool_sizes) / sizeof(descriptor_pool_sizes[0])),
        .pPoolSizes = descriptor_pool_sizes,
    };

    if (vkCreateDescriptorPool(context.device, &descriptor_pool, nullptr,
                               &vk_imgui_descriptor_pool) != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan descriptor pool for ImGUI\n";
        return false;
    }

    ImGui_ImplVulkan_InitInfo init = {
        .ApiVersion = vk_api_version,
        .Instance = vk_instance,
        .PhysicalDevice = context.physical_device,
        .Device = context.device,
        .QueueFamily = context.graphics_queue_index,
        .Queue = context.graphics_queue,
        .DescriptorPool = vk_imgui_descriptor_pool,
        .MinImageCount = static_cast<uint32_t>(vk_swapchain_images.size()),
        .ImageCount = static_cast<uint32_t>(vk_swapchain_images.size()),
        .PipelineInfoMain = {
            .RenderPass = context.render_pass,
        },
    };

    return ImGui_ImplVulkan_Init(&init);
}

bool rebuildSwapchain(uint32_t width, uint32_t height) {
    vkQueueWaitIdle(context.graphics_queue);

    vkb::SwapchainBuilder sb(context.physical_device, context.device, vk_surface,
                             context.graphics_queue_index, context.graphics_queue_index);

    auto sb_result = sb.set_desired_extent(width, height)
                       .use_default_format_selection()
                       .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
                       .use_default_image_usage_flags()
                       .set_old_swapchain(vk_swapchain)
                       .build();
    if (!sb_result) return false;

    auto vkb_swapchain = sb_result.value();

    for (size_t i = 0, n = vk_swapchain_image_views.size(); i < n; ++i) {
        vkDestroyFramebuffer(context.device, vk_framebuffers[i], nullptr);
        vkDestroyImageView(context.device, vk_swapchain_image_views[i], nullptr);
    }

    vkDestroySwapchainKHR(context.device, vk_swapchain, nullptr);

    vk_swapchain = vkb_swapchain.swapchain;
    context.swapchain_format = vkb_swapchain.image_format;
    context.swapchain_extent = vkb_swapchain.extent;

    vk_swapchain_images = vkb_swapchain.get_images().value();
    vk_swapchain_image_views = vkb_swapchain.get_image_views().value();

    vkDestroyImageView(context.device, vk_image_view_depth_buffer, nullptr);
    vmaDestroyImage(context.allocator, vk_image_depth_buffer, vma_allocation_depth_buffer);

    const VkImageCreateInfo depth_buffer = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = vk_depth_buffer_format,
        .extent = { context.swapchain_extent.width, context.swapchain_extent.height, 1 },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };

    const VmaAllocationCreateInfo depth_buffer_allocation = { .usage = VMA_MEMORY_USAGE_AUTO };

    if (vmaCreateImage(context.allocator, &depth_buffer, &depth_buffer_allocation,
                       &vk_image_depth_buffer, &vma_allocation_depth_buffer, nullptr) != VK_SUCCESS) {
        return false;
    }

    VkImageAspectFlags aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    if (vk_depth_buffer_format == VK_FORMAT_D32_SFLOAT_S8_UINT || vk_depth_buffer_format == VK_FORMAT_D24_UNORM_S8_UINT) {
        aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }

    const VkImageViewCreateInfo depth_buffer_view = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = vk_image_depth_buffer,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = vk_depth_buffer_format,
        .subresourceRange = {
            .aspectMask = aspectMask,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };

    if (vkCreateImageView(context.device, &depth_buffer_view, nullptr, &vk_image_view_depth_buffer) != VK_SUCCESS) {
        return false;
    }

    const uint32_t swapchain_images_count = uint32_t(vk_swapchain_images.size());
    VkImageView framebuffer_attachments[] = { VK_NULL_HANDLE, vk_image_view_depth_buffer };

    const VkFramebufferCreateInfo framebuffer = {
        .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .renderPass = context.render_pass,
        .attachmentCount = 2,
        .pAttachments = framebuffer_attachments,
        .width = context.swapchain_extent.width,
        .height = context.swapchain_extent.height,
        .layers = 1,
    };

    vk_framebuffers.resize(swapchain_images_count);
    for (uint32_t i = 0; i < swapchain_images_count; ++i) {
        framebuffer_attachments[0] = vk_swapchain_image_views[i];
        if (vkCreateFramebuffer(context.device, &framebuffer, nullptr, &vk_framebuffers[i]) != VK_SUCCESS) {
            return false;
        }
    }

    vk_swapchain_resize_require = false;
    return true;
}

} // namespace

Context context;

bool initialize(GLFWwindow* const window) {
    vkb::InstanceBuilder ib;
    auto ibr = ib.require_api_version(VK_MAKE_VERSION(1, 1, 0))
                 .request_validation_layers()
                 .build();

    if (!ibr) return false;
    auto vkb_instance = ibr.value();
    vk_instance = vkb_instance.instance;
    vk_api_version = vkb_instance.api_version;

    if (glfwCreateWindowSurface(vk_instance, window, nullptr, &vk_surface) != VK_SUCCESS) {
        return false;
    }

    vkb::PhysicalDeviceSelector pds(vkb_instance, vk_surface);
    auto pds_result = pds.prefer_gpu_device_type(vkb::PreferredDeviceType::discrete)
                         .require_present()
                         .select();
    if (!pds_result) return false;
    auto vkb_physical_device = pds_result.value();

    vk_depth_buffer_format = selectDepthFormat(vkb_physical_device.physical_device);

    vkb::DeviceBuilder db(vkb_physical_device);
    auto db_result = db.build();
    if (!db_result) return false;
    auto vkb_device = db_result.value();

    context.physical_device = vkb_device.physical_device;
    context.device = vkb_device.device;
    context.graphics_queue = vkb_device.get_queue(vkb::QueueType::graphics).value();
    context.graphics_queue_index = vkb_device.get_queue_index(vkb::QueueType::graphics).value();

    const VmaAllocatorCreateInfo allocator = {
        .physicalDevice = context.physical_device,
        .device = context.device,
        .instance = vk_instance,
        .vulkanApiVersion = vk_api_version,
    };
    if (vmaCreateAllocator(&allocator, &context.allocator) != VK_SUCCESS) return false;

    vkb::SwapchainBuilder sb(vkb_device);
    auto sb_result = sb.use_default_format_selection()
                       .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
                       .use_default_image_usage_flags()
                       .build();
    if (!sb_result) return false;

    auto vkb_swapchain = sb_result.value();
    vk_swapchain = vkb_swapchain.swapchain;
    context.swapchain_format = vkb_swapchain.image_format;
    context.swapchain_extent = vkb_swapchain.extent;
    vk_swapchain_images = vkb_swapchain.get_images().value();
    vk_swapchain_image_views = vkb_swapchain.get_image_views().value();
    vk_swapchain_current_image = UINT32_MAX;

    const uint32_t swapchain_images_count = uint32_t(vk_swapchain_images.size());

    const VkImageCreateInfo depth_buffer = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = vk_depth_buffer_format,
        .extent = { context.swapchain_extent.width, context.swapchain_extent.height, 1 },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };

    const VmaAllocationCreateInfo depth_buffer_allocation = { .usage = VMA_MEMORY_USAGE_AUTO };
    if (vmaCreateImage(context.allocator, &depth_buffer, &depth_buffer_allocation,
                       &vk_image_depth_buffer, &vma_allocation_depth_buffer, nullptr) != VK_SUCCESS) {
        return false;
    }

    VkImageAspectFlags aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    if (vk_depth_buffer_format == VK_FORMAT_D32_SFLOAT_S8_UINT || vk_depth_buffer_format == VK_FORMAT_D24_UNORM_S8_UINT) {
        aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }

    const VkImageViewCreateInfo depth_buffer_view = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = vk_image_depth_buffer,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = vk_depth_buffer_format,
        .subresourceRange = {
            .aspectMask = aspectMask,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };

    if (vkCreateImageView(context.device, &depth_buffer_view, nullptr, &vk_image_view_depth_buffer) != VK_SUCCESS) {
        return false;
    }

    const VkAttachmentDescription render_pass_attachments[] = {
        {
            .format = context.swapchain_format,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        },
        {
            .format = vk_depth_buffer_format,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        },
    };

    const VkAttachmentReference render_pass_color_attachment = {
        .attachment = 0,
        .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };

    const VkAttachmentReference render_pass_depth_attachment = {
        .attachment = 1,
        .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };

    const VkSubpassDescription render_pass_subpass = {
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = 1,
        .pColorAttachments = &render_pass_color_attachment,
        .pDepthStencilAttachment = &render_pass_depth_attachment,
    };

    const VkSubpassDependency dependency = {
        .srcSubpass = VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
    };

    const VkRenderPassCreateInfo render_pass = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = 2,
        .pAttachments = render_pass_attachments,
        .subpassCount = 1,
        .pSubpasses = &render_pass_subpass,
        .dependencyCount = 1,
        .pDependencies = &dependency,
    };

    if (vkCreateRenderPass(context.device, &render_pass, nullptr, &context.render_pass) != VK_SUCCESS) {
        return false;
    }

    VkImageView framebuffer_attachments[] = { VK_NULL_HANDLE, vk_image_view_depth_buffer };

    const VkFramebufferCreateInfo framebuffer = {
        .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .renderPass = context.render_pass,
        .attachmentCount = 2,
        .pAttachments = framebuffer_attachments,
        .width = context.swapchain_extent.width,
        .height = context.swapchain_extent.height,
        .layers = 1,
    };

    vk_framebuffers.resize(swapchain_images_count);
    for (uint32_t i = 0; i < swapchain_images_count; ++i) {
        framebuffer_attachments[0] = vk_swapchain_image_views[i];
        if (vkCreateFramebuffer(context.device, &framebuffer, nullptr, &vk_framebuffers[i]) != VK_SUCCESS) {
            return false;
        }
    }

    const VkSemaphoreCreateInfo semaphore = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    const VkFenceCreateInfo fence = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };

    vk_semaphores_image_finished.resize(swapchain_images_count);
    for (uint32_t i = 0; i < swapchain_images_count; ++i) {
        if (vkCreateSemaphore(context.device, &semaphore, nullptr, &vk_semaphores_image_finished[i]) != VK_SUCCESS) {
            return false;
        }
    }

    if (vkCreateSemaphore(context.device, &semaphore, nullptr, &vk_semaphore_image_available) != VK_SUCCESS) return false;
    if (vkCreateFence(context.device, &fence, nullptr, &vk_fence_frame_in_flight) != VK_SUCCESS) return false;

    const VkCommandPoolCreateInfo command_pool = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = context.graphics_queue_index,
    };
    if (vkCreateCommandPool(context.device, &command_pool, nullptr, &vk_command_pool) != VK_SUCCESS) return false;

    const VkCommandBufferAllocateInfo command_buffers = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = vk_command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    if (vkAllocateCommandBuffers(context.device, &command_buffers, &vk_command_buffer) != VK_SUCCESS) return false;

    if (!initializeImGUI()) return false;

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    vk_swapchain_resize_width = width;
    vk_swapchain_resize_height = height;

    return true;
}

void shutdown() {
    vkQueueWaitIdle(context.graphics_queue);

    ImGui_ImplVulkan_Shutdown();
    vkDestroyDescriptorPool(context.device, vk_imgui_descriptor_pool, nullptr);

    vkDestroyCommandPool(context.device, vk_command_pool, nullptr);
    vkDestroyFence(context.device, vk_fence_frame_in_flight, nullptr);

    for (size_t i = 0, n = vk_swapchain_images.size(); i < n; ++i) {
        vkDestroySemaphore(context.device, vk_semaphores_image_finished[i], nullptr);
    }
    vkDestroySemaphore(context.device, vk_semaphore_image_available, nullptr);

    for (size_t i = 0, n = vk_swapchain_images.size(); i < n; ++i) {
        vkDestroyFramebuffer(context.device, vk_framebuffers[i], nullptr);
    }
    vkDestroyRenderPass(context.device, context.render_pass, nullptr);

    vkDestroyImageView(context.device, vk_image_view_depth_buffer, nullptr);
    vmaDestroyImage(context.allocator, vk_image_depth_buffer, vma_allocation_depth_buffer);

    for (size_t i = 0, n = vk_swapchain_images.size(); i < n; ++i) {
        vkDestroyImageView(context.device, vk_swapchain_image_views[i], nullptr);
    }
    vkDestroySwapchainKHR(context.device, vk_swapchain, nullptr);

    vmaDestroyAllocator(context.allocator);
    vkDestroyDevice(context.device, nullptr);
    vkDestroySurfaceKHR(vk_instance, vk_surface, nullptr);
    vkDestroyInstance(vk_instance, nullptr);
}

void resize(uint32_t width, uint32_t height) {
    if (width == 0 || height == 0) return;
    vk_swapchain_resize_width = width;
    vk_swapchain_resize_height = height;
    vk_swapchain_resize_require = true;
}

FrameData prepare() {
    vkWaitForFences(context.device, 1, &vk_fence_frame_in_flight, VK_TRUE, UINT64_MAX);

retry_acquire:
    switch (vkAcquireNextImageKHR(context.device, vk_swapchain, UINT64_MAX,
                                  vk_semaphore_image_available, VK_NULL_HANDLE,
                                  &vk_swapchain_current_image)) {
    case VK_SUCCESS:
        break;
    case VK_ERROR_OUT_OF_DATE_KHR:
        rebuildSwapchain(vk_swapchain_resize_width, vk_swapchain_resize_height);
        goto retry_acquire;
    default:
        return {};
    }

    vkResetFences(context.device, 1, &vk_fence_frame_in_flight);

    vkResetCommandBuffer(vk_command_buffer, 0);
    const VkCommandBufferBeginInfo begin_info = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    vkBeginCommandBuffer(vk_command_buffer, &begin_info);

    VkClearValue clear_values[2] = {};
    clear_values[0].color = {{ 0, 0, 0, 1.0f }};
    clear_values[1].depthStencil = { 1.0f, 0 };

    const VkRenderPassBeginInfo rp_begin = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = context.render_pass,
        .framebuffer = vk_framebuffers[vk_swapchain_current_image],
        .renderArea = { .extent = context.swapchain_extent },
        .clearValueCount = 2,
        .pClearValues = clear_values,
    };
    vkCmdBeginRenderPass(vk_command_buffer, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);

    return {
        .framebuffer = vk_framebuffers[vk_swapchain_current_image],
        .command_buffer = vk_command_buffer,
    };
}

void submitAndPresent() {
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), vk_command_buffer);

    vkCmdEndRenderPass(vk_command_buffer);
    vkEndCommandBuffer(vk_command_buffer);

    const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    const VkSubmitInfo submit = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &vk_semaphore_image_available,
        .pWaitDstStageMask = &stage,
        .commandBufferCount = 1,
        .pCommandBuffers = &vk_command_buffer,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &vk_semaphores_image_finished[vk_swapchain_current_image],
    };

    vkQueueSubmit(context.graphics_queue, 1, &submit, vk_fence_frame_in_flight);

    const VkPresentInfoKHR present = {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &vk_semaphores_image_finished[vk_swapchain_current_image],
        .swapchainCount = 1,
        .pSwapchains = &vk_swapchain,
        .pImageIndices = &vk_swapchain_current_image,
    };

    VkResult result = vkQueuePresentKHR(context.graphics_queue, &present);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || vk_swapchain_resize_require) {
        rebuildSwapchain(vk_swapchain_resize_width, vk_swapchain_resize_height);
    }
}

} // namespace graphics::internal