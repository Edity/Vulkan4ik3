#include "application.hpp"
#include "graphics_internal.hpp"

#include <imgui.h>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/type_ptr.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using namespace graphics::internal;

namespace {
struct Vertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 uv;
};

struct alignas(16) SceneUBO {
    glm::mat4 view;
    glm::mat4 proj;
    glm::vec4 cameraPos;
    glm::vec4 ambientColor;
    glm::vec4 directionalDirection;
    glm::vec4 directionalColor;
};

struct alignas(16) ModelUBO {
    glm::mat4 model;
    glm::vec4 tint;
    glm::vec4 materialParams;
};

struct SamplingPushConstants {
    uint32_t flags = 0;
    float time = 0.0f;
    float modulation = 1.0f;
    float reserved = 0.0f;
};
static_assert(sizeof(SamplingPushConstants) == 16);

struct alignas(16) PointLightGPU { glm::vec4 position; glm::vec4 colorIntensity; };
struct alignas(16) SpotLightGPU {
    glm::vec4 position;
    glm::vec4 direction;
    glm::vec4 colorIntensity;
    glm::vec4 coneParams;
};

struct Buffer {
    VkBuffer handle = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
};

struct Texture {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
};

struct Material {
    std::string name;
    size_t albedoTexture = 0;
    size_t specularTexture = 0;
    size_t emissiveTexture = 0;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    bool showInUi = true;
};

struct SceneObject {
    glm::vec3 pos{0.0f};
    glm::vec3 rot{0.0f};
    glm::vec3 scale{1.0f};
    glm::vec4 tint{1.0f};
    float shininess = 48.0f;
    float specularStrength = 1.0f;
    float emissiveStrength = 1.0f;
    bool visible = true;
    bool animatedSampling = false;
    int materialIndex = 0;
};

struct PointLight {
    glm::vec3 position{2.0f, 1.5f, 2.0f};
    glm::vec3 color{1.0f, 0.72f, 0.42f};
    float intensity = 20.0f;
};

struct SpotLight {
    glm::vec3 position{-2.0f, 2.0f, 2.0f};
    glm::vec3 direction{0.5f, -0.5f, -1.0f};
    glm::vec3 color{0.35f, 0.55f, 1.0f};
    float intensity = 35.0f;
    float innerDegrees = 18.0f;
    float outerDegrees = 28.0f;
};

constexpr uint32_t kMaxDraws = 256;
constexpr uint32_t kMaxLightsPerType = 64;

std::vector<Vertex> vertices;
std::vector<uint32_t> indices;
Buffer vertexBuffer, indexBuffer, sceneBuffer, pointBuffer, spotBuffer, lineBuffer;
Buffer dynamicModelBuffer;
void* dynamicModelMapped = nullptr;
VkDeviceSize dynamicModelStride = sizeof(ModelUBO);
size_t lineBufferCapacity = 0;
uint32_t dynamicDrawSlot = 0;

VkDescriptorSetLayout sceneSetLayout = VK_NULL_HANDLE;
VkDescriptorSetLayout materialSetLayout = VK_NULL_HANDLE;
VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
VkPipeline graphicsPipeline = VK_NULL_HANDLE;
VkPipeline linePipeline = VK_NULL_HANDLE;
VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
VkDescriptorSet sceneSet = VK_NULL_HANDLE;

std::vector<Texture> textures;
std::vector<Material> materials;
size_t blackTextureIndex = 0;
size_t whiteTextureIndex = 0;
int markerMaterialIndex = -1;

SceneObject sphere;
SceneObject orbitSphere;
std::vector<PointLight> pointLights;
std::vector<SpotLight> spotLights;

glm::vec3 cameraPosition{0.0f, 1.5f, 6.0f};
float cameraYaw = -90.0f;
float cameraPitch = -12.0f;
float fovDegrees = 45.0f;
glm::vec3 ambientColor{0.12f, 0.12f, 0.14f};
glm::vec3 directionalDirection{-0.4f, -1.0f, -0.3f};
glm::vec3 directionalColor{0.8f, 0.8f, 0.75f};
float directionalIntensity = 1.0f;
bool usePerspective = true;
bool showLightMarkers = true;
bool animateOrbit = true;
bool enableAnimatedSampling = true;
float orbitRadius = 2.1f;
float orbitSpeed = 1.0f;
float animationTime = 0.0f;
bool buffersDirty = true;

VkDeviceSize alignUp(VkDeviceSize value, VkDeviceSize alignment) {
    if (alignment <= 1) return value;
    return (value + alignment - 1) & ~(alignment - 1);
}

bool createHostBuffer(VkDeviceSize size, VkBufferUsageFlags usage, Buffer& out) {
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = std::max<VkDeviceSize>(size, 16);
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    return vmaCreateBuffer(context.allocator, &bi, &ai, &out.handle, &out.allocation, nullptr) == VK_SUCCESS;
}

bool createDeviceBuffer(VkDeviceSize size, VkBufferUsageFlags usage, Buffer& out) {
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = std::max<VkDeviceSize>(size, 16);
    bi.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    return vmaCreateBuffer(context.allocator, &bi, &ai, &out.handle, &out.allocation, nullptr) == VK_SUCCESS;
}

void destroyBuffer(Buffer& buffer) {
    if (buffer.handle != VK_NULL_HANDLE) {
        vmaDestroyBuffer(context.allocator, buffer.handle, buffer.allocation);
    }
    buffer = {};
}

bool uploadBuffer(const Buffer& buffer, const void* source, size_t bytes) {
    if (!buffer.handle || !source || bytes == 0) return false;
    void* mapped = nullptr;
    if (vmaMapMemory(context.allocator, buffer.allocation, &mapped) != VK_SUCCESS) return false;
    std::memcpy(mapped, source, bytes);
    const VkResult flushResult = vmaFlushAllocation(context.allocator, buffer.allocation, 0, bytes);
    vmaUnmapMemory(context.allocator, buffer.allocation);
    return flushResult == VK_SUCCESS;
}

bool immediateSubmit(const std::function<void(VkCommandBuffer)>& record) {
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pci.queueFamilyIndex = context.graphics_queue_index;
    if (vkCreateCommandPool(context.device, &pci, nullptr, &pool) != VK_SUCCESS) return false;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(context.device, &cai, &cmd) != VK_SUCCESS) {
        vkDestroyCommandPool(context.device, pool, nullptr);
        return false;
    }

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    bool ok = vkBeginCommandBuffer(cmd, &bi) == VK_SUCCESS;
    if (ok) {
        record(cmd);
        ok = vkEndCommandBuffer(cmd) == VK_SUCCESS;
    }
    if (ok) {
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cmd;
        ok = vkQueueSubmit(context.graphics_queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS;
        if (ok) ok = vkQueueWaitIdle(context.graphics_queue) == VK_SUCCESS;
    }

    vkDestroyCommandPool(context.device, pool, nullptr);
    return ok;
}

bool uploadDeviceBuffer(const void* source, VkDeviceSize bytes, VkBufferUsageFlags finalUsage, Buffer& destination) {
    Buffer staging;
    if (!createHostBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, staging)) return false;
    if (!uploadBuffer(staging, source, size_t(bytes)) ||
        !createDeviceBuffer(bytes, finalUsage, destination)) {
        destroyBuffer(staging);
        destroyBuffer(destination);
        return false;
    }

    const bool copied = immediateSubmit([&](VkCommandBuffer cmd) {
        VkBufferCopy region{};
        region.size = bytes;
        vkCmdCopyBuffer(cmd, staging.handle, destination.handle, 1, &region);

        VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = (finalUsage & VK_BUFFER_USAGE_INDEX_BUFFER_BIT)
            ? VK_ACCESS_INDEX_READ_BIT : VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = destination.handle;
        barrier.offset = 0;
        barrier.size = bytes;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                             0, 0, nullptr, 1, &barrier, 0, nullptr);
    });
    destroyBuffer(staging);
    if (!copied) destroyBuffer(destination);
    return copied;
}

void destroyTexture(Texture& texture) {
    if (texture.sampler) vkDestroySampler(context.device, texture.sampler, nullptr);
    if (texture.view) vkDestroyImageView(context.device, texture.view, nullptr);
    if (texture.image) vmaDestroyImage(context.allocator, texture.image, texture.allocation);
    texture = {};
}

bool createTextureFromPixels(int width, int height, const unsigned char* pixels, bool srgb, Texture& out) {
    if (width <= 0 || height <= 0 || !pixels) return false;
    const VkDeviceSize imageBytes = VkDeviceSize(width) * VkDeviceSize(height) * 4;

    Buffer staging;
    if (!createHostBuffer(imageBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, staging)) return false;
    if (!uploadBuffer(staging, pixels, size_t(imageBytes))) {
        destroyBuffer(staging);
        return false;
    }

    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
    ici.extent = {uint32_t(width), uint32_t(height), 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo ainfo{};
    ainfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (vmaCreateImage(context.allocator, &ici, &ainfo, &out.image, &out.allocation, nullptr) != VK_SUCCESS) {
        destroyBuffer(staging);
        return false;
    }

    const bool copied = immediateSubmit([&](VkCommandBuffer cmd) {
        VkImageMemoryBarrier toTransfer{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        toTransfer.srcAccessMask = 0;
        toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toTransfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toTransfer.image = out.image;
        toTransfer.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        toTransfer.subresourceRange.baseMipLevel = 0;
        toTransfer.subresourceRange.levelCount = 1;
        toTransfer.subresourceRange.baseArrayLayer = 0;
        toTransfer.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &toTransfer);

        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = {uint32_t(width), uint32_t(height), 1};
        vkCmdCopyBufferToImage(cmd, staging.handle, out.image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        VkImageMemoryBarrier toShader{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        toShader.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toShader.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        toShader.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toShader.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        toShader.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toShader.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toShader.image = out.image;
        toShader.subresourceRange = toTransfer.subresourceRange;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &toShader);
    });
    destroyBuffer(staging);
    if (!copied) {
        destroyTexture(out);
        return false;
    }

    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = out.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = ici.format;
    vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vci.subresourceRange.baseMipLevel = 0;
    vci.subresourceRange.levelCount = 1;
    vci.subresourceRange.baseArrayLayer = 0;
    vci.subresourceRange.layerCount = 1;
    if (vkCreateImageView(context.device, &vci, nullptr, &out.view) != VK_SUCCESS) {
        destroyTexture(out);
        return false;
    }

    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.maxAnisotropy = 1.0f;
    sci.minLod = 0.0f;
    sci.maxLod = 0.0f;
    if (vkCreateSampler(context.device, &sci, nullptr, &out.sampler) != VK_SUCCESS) {
        destroyTexture(out);
        return false;
    }
    return true;
}

bool loadTexture(const std::string& path, bool srgb, size_t& outIndex) {
    int width = 0, height = 0, channels = 0;
    stbi_uc* pixels = stbi_load(path.c_str(), &width, &height, &channels, STBI_rgb_alpha);
    if (!pixels) {
        std::cerr << "Failed to load texture '" << path << "': "
                  << (stbi_failure_reason() ? stbi_failure_reason() : "unknown error") << '\n';
        return false;
    }
    Texture texture;
    const bool ok = createTextureFromPixels(width, height, pixels, srgb, texture);
    stbi_image_free(pixels);
    if (!ok) {
        std::cerr << "Failed to upload texture: " << path << '\n';
        return false;
    }
    outIndex = textures.size();
    textures.push_back(texture);
    return true;
}

bool createSolidTexture(const std::array<unsigned char, 4>& pixel, bool srgb, size_t& outIndex) {
    Texture texture;
    if (!createTextureFromPixels(1, 1, pixel.data(), srgb, texture)) return false;
    outIndex = textures.size();
    textures.push_back(texture);
    return true;
}

bool addMaterial(const std::string& name, const std::string& albedoPath,
                 const std::string& specularPath, const std::string& emissivePath,
                 bool showInUi = true) {
    Material material;
    material.name = name;
    material.showInUi = showInUi;
    if (!loadTexture(albedoPath, true, material.albedoTexture)) return false;
    material.specularTexture = blackTextureIndex;
    material.emissiveTexture = blackTextureIndex;
    if (!specularPath.empty() && !loadTexture(specularPath, false, material.specularTexture)) return false;
    if (!emissivePath.empty() && !loadTexture(emissivePath, true, material.emissiveTexture)) return false;
    materials.push_back(std::move(material));
    return true;
}

std::vector<char> readFile(const std::string& name) {
    std::ifstream file(name, std::ios::binary | std::ios::ate);
    if (!file) {
        file.clear();
        file.open(std::string("build/") + name, std::ios::binary | std::ios::ate);
    }
    if (!file) {
        std::cerr << "Cannot open shader: " << name << ". Build shaders with CMake first.\n";
        return {};
    }
    const size_t size = size_t(file.tellg());
    std::vector<char> data(size);
    file.seekg(0);
    file.read(data.data(), std::streamsize(size));
    return data;
}

VkShaderModule createShaderModule(const std::vector<char>& code) {
    if (code.empty() || code.size() % 4 != 0) return VK_NULL_HANDLE;
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = code.size();
    ci.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(context.device, &ci, nullptr, &module) != VK_SUCCESS) return VK_NULL_HANDLE;
    return module;
}

void generateSphere(float radius, int stacks, int sectors) {
    vertices.clear();
    indices.clear();
    for (int i = 0; i <= stacks; ++i) {
        const float v = float(i) / float(stacks);
        const float phi = v * glm::pi<float>();
        for (int j = 0; j <= sectors; ++j) {
            const float u = float(j) / float(sectors);
            const float theta = u * 2.0f * glm::pi<float>();
            glm::vec3 p(
                radius * std::sin(phi) * std::cos(theta),
                radius * std::cos(phi),
                radius * std::sin(phi) * std::sin(theta)
            );
            vertices.push_back({p, glm::normalize(p), glm::vec2(u, 1.0f - v)});
        }
    }
    for (int i = 0; i < stacks; ++i) {
        for (int j = 0; j < sectors; ++j) {
            const uint32_t a = uint32_t(i * (sectors + 1) + j);
            const uint32_t b = a + uint32_t(sectors + 1);
            indices.insert(indices.end(), {a, b, a + 1, b, b + 1, a + 1});
        }
    }
}

glm::mat4 makeModel(const SceneObject& object, const glm::vec3& position) {
    glm::mat4 model(1.0f);
    model = glm::translate(model, position);
    model = glm::rotate(model, glm::radians(object.rot.x), glm::vec3(1, 0, 0));
    model = glm::rotate(model, glm::radians(object.rot.y), glm::vec3(0, 1, 0));
    model = glm::rotate(model, glm::radians(object.rot.z), glm::vec3(0, 0, 1));
    return glm::scale(model, object.scale);
}

uint32_t writeDynamicModel(const glm::mat4& model, const glm::vec4& tint,
                           float shininess, float specularStrength,
                           bool unlit, float emissiveStrength) {
    if (!dynamicModelMapped || dynamicDrawSlot >= kMaxDraws) return 0;
    const ModelUBO data{
        model,
        tint,
        glm::vec4(shininess, specularStrength, unlit ? 1.0f : 0.0f, emissiveStrength)
    };
    const VkDeviceSize offset = VkDeviceSize(dynamicDrawSlot) * dynamicModelStride;
    std::memcpy(static_cast<unsigned char*>(dynamicModelMapped) + offset, &data, sizeof(data));
    const uint32_t dynamicOffset = uint32_t(offset);
    ++dynamicDrawSlot;
    return dynamicOffset;
}

bool createTextureDescriptorLayout() {
    std::array<VkDescriptorSetLayoutBinding, 4> bindings{};
    bindings[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1,
                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    for (uint32_t i = 1; i < 4; ++i) {
        bindings[i] = {i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                       VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    }
    VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    ci.bindingCount = uint32_t(bindings.size());
    ci.pBindings = bindings.data();
    return vkCreateDescriptorSetLayout(context.device, &ci, nullptr, &materialSetLayout) == VK_SUCCESS;
}

bool createMaterialDescriptorSets() {
    for (Material& material : materials) {
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = descriptorPool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &materialSetLayout;
        if (vkAllocateDescriptorSets(context.device, &ai, &material.descriptorSet) != VK_SUCCESS) return false;

        VkDescriptorBufferInfo bufferInfo{dynamicModelBuffer.handle, 0, sizeof(ModelUBO)};
        const Texture& albedo = textures[material.albedoTexture];
        const Texture& specular = textures[material.specularTexture];
        const Texture& emissive = textures[material.emissiveTexture];
        std::array<VkDescriptorImageInfo, 3> imageInfos{{
            {albedo.sampler, albedo.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
            {specular.sampler, specular.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
            {emissive.sampler, emissive.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}
        }};
        std::array<VkWriteDescriptorSet, 4> writes{};
        writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[0].dstSet = material.descriptorSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        writes[0].pBufferInfo = &bufferInfo;
        for (uint32_t i = 0; i < 3; ++i) {
            writes[i + 1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            writes[i + 1].dstSet = material.descriptorSet;
            writes[i + 1].dstBinding = i + 1;
            writes[i + 1].descriptorCount = 1;
            writes[i + 1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[i + 1].pImageInfo = &imageInfos[i];
        }
        vkUpdateDescriptorSets(context.device, uint32_t(writes.size()), writes.data(), 0, nullptr);
    }
    return true;
}

void updateSceneDescriptorBuffers() {
    std::vector<PointLightGPU> pointGpu;
    pointGpu.reserve(pointLights.size());
    for (const PointLight& light : pointLights) {
        pointGpu.push_back({glm::vec4(light.position, 1.0f), glm::vec4(light.color, light.intensity)});
    }

    std::vector<SpotLightGPU> spotGpu;
    spotGpu.reserve(spotLights.size());
    for (const SpotLight& light : spotLights) {
        const glm::vec3 direction = glm::length(light.direction) > 0.0001f
            ? glm::normalize(light.direction) : glm::vec3(0, -1, 0);
        spotGpu.push_back({
            glm::vec4(light.position, 1.0f),
            glm::vec4(direction, 0.0f),
            glm::vec4(light.color, light.intensity),
            glm::vec4(std::cos(glm::radians(light.innerDegrees)),
                      std::cos(glm::radians(light.outerDegrees)), 0.0f, 0.0f)
        });
    }

    destroyBuffer(pointBuffer);
    destroyBuffer(spotBuffer);
    PointLightGPU dummyPoint{};
    SpotLightGPU dummySpot{};
    const void* pointData = pointGpu.empty() ? static_cast<const void*>(&dummyPoint) : pointGpu.data();
    const void* spotData = spotGpu.empty() ? static_cast<const void*>(&dummySpot) : spotGpu.data();
    const size_t pointBytes = pointGpu.empty() ? sizeof(dummyPoint) : pointGpu.size() * sizeof(PointLightGPU);
    const size_t spotBytes = spotGpu.empty() ? sizeof(dummySpot) : spotGpu.size() * sizeof(SpotLightGPU);

    if (!createHostBuffer(pointBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, pointBuffer) ||
        !createHostBuffer(spotBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, spotBuffer)) {
        std::cerr << "Failed to create light SSBOs\n";
        return;
    }
    uploadBuffer(pointBuffer, pointData, pointBytes);
    uploadBuffer(spotBuffer, spotData, spotBytes);

    std::array<VkDescriptorBufferInfo, 2> infos{{
        {pointBuffer.handle, 0, pointBytes},
        {spotBuffer.handle, 0, spotBytes}
    }};
    std::array<VkWriteDescriptorSet, 2> writes{};
    for (uint32_t i = 0; i < 2; ++i) {
        writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[i].dstSet = sceneSet;
        writes[i].dstBinding = i + 1;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(context.device, uint32_t(writes.size()), writes.data(), 0, nullptr);
    buffersDirty = false;
}

void bindMaterialAndModel(VkCommandBuffer cmd, const Material& material,
                          const glm::mat4& model, const glm::vec4& tint,
                          float shininess, float specularStrength,
                          bool unlit, float emissiveStrength,
                          bool animatedSampling, float time) {
    const uint32_t dynamicOffset = writeDynamicModel(
        model, tint, shininess, specularStrength, unlit, emissiveStrength
    );
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
                            1, 1, &material.descriptorSet, 1, &dynamicOffset);
    SamplingPushConstants push{};
    push.flags = animatedSampling ? 1u : 0u;
    push.time = time;
    push.modulation = 1.0f;
    vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(push), &push);
}

void drawObject(VkCommandBuffer cmd, const SceneObject& object,
                const glm::vec3& position, float time) {
    if (!object.visible || object.materialIndex < 0 ||
        size_t(object.materialIndex) >= materials.size()) return;
    const Material& material = materials[size_t(object.materialIndex)];
    bindMaterialAndModel(cmd, material, makeModel(object, position), object.tint,
                         object.shininess, object.specularStrength, false,
                         object.emissiveStrength,
                         object.animatedSampling && enableAnimatedSampling, time);
    vkCmdDrawIndexed(cmd, uint32_t(indices.size()), 1, 0, 0, 0);
}

bool createPipeline() {
    VkShaderModule vertexModule = createShaderModule(readFile("shaders/vert.spv"));
    VkShaderModule fragmentModule = createShaderModule(readFile("shaders/frag.spv"));
    if (!vertexModule || !fragmentModule) {
        if (vertexModule) vkDestroyShaderModule(context.device, vertexModule, nullptr);
        if (fragmentModule) vkDestroyShaderModule(context.device, fragmentModule, nullptr);
        return false;
    }

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                 VK_SHADER_STAGE_VERTEX_BIT, vertexModule, "main", nullptr};
    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                 VK_SHADER_STAGE_FRAGMENT_BIT, fragmentModule, "main", nullptr};

    VkVertexInputBindingDescription binding{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
    std::array<VkVertexInputAttributeDescription, 3> attributes{{
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, uint32_t(offsetof(Vertex, pos))},
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, uint32_t(offsetof(Vertex, normal))},
        {2, 0, VK_FORMAT_R32G32_SFLOAT, uint32_t(offsetof(Vertex, uv))}
    }};
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = 1;
    vi.pVertexBindingDescriptions = &binding;
    vi.vertexAttributeDescriptionCount = uint32_t(attributes.size());
    vi.pVertexAttributeDescriptions = attributes.data();

    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depth.depthTestEnable = VK_TRUE;
    depth.depthWriteEnable = VK_TRUE;
    depth.depthCompareOp = VK_COMPARE_OP_LESS;
    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                     VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = 1;
    blend.pAttachments = &blendAttachment;
    std::array<VkDynamicState, 2> dynamicStates{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = uint32_t(dynamicStates.size());
    dynamic.pDynamicStates = dynamicStates.data();

    const std::array<VkDescriptorSetLayout, 2> setLayouts{sceneSetLayout, materialSetLayout};
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(SamplingPushConstants);
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = uint32_t(setLayouts.size());
    plci.pSetLayouts = setLayouts.data();
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pushRange;
    if (vkCreatePipelineLayout(context.device, &plci, nullptr, &pipelineLayout) != VK_SUCCESS) {
        vkDestroyShaderModule(context.device, fragmentModule, nullptr);
        vkDestroyShaderModule(context.device, vertexModule, nullptr);
        return false;
    }

    VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gp.stageCount = 2;
    gp.pStages = stages;
    gp.pVertexInputState = &vi;
    gp.pInputAssemblyState = &ia;
    gp.pViewportState = &viewport;
    gp.pRasterizationState = &raster;
    gp.pMultisampleState = &multisample;
    gp.pDepthStencilState = &depth;
    gp.pColorBlendState = &blend;
    gp.pDynamicState = &dynamic;
    gp.layout = pipelineLayout;
    gp.renderPass = context.render_pass;
    gp.subpass = 0;

    VkResult result = vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &gp, nullptr, &graphicsPipeline);
    if (result == VK_SUCCESS) {
        ia.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
        result = vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &gp, nullptr, &linePipeline);
    }
    vkDestroyShaderModule(context.device, fragmentModule, nullptr);
    vkDestroyShaderModule(context.device, vertexModule, nullptr);
    return result == VK_SUCCESS;
}

void buildObjectUI(const char* title, SceneObject& object) {
    if (!ImGui::TreeNode(title)) return;
    ImGui::Checkbox("Visible", &object.visible);
    ImGui::DragFloat3("Position", glm::value_ptr(object.pos), 0.03f);
    ImGui::DragFloat3("Rotation", glm::value_ptr(object.rot), 0.5f);
    ImGui::DragFloat3("Scale", glm::value_ptr(object.scale), 0.01f, 0.05f, 5.0f);
    ImGui::ColorEdit4("Tint", glm::value_ptr(object.tint));
    ImGui::SliderFloat("Shininess", &object.shininess, 1.0f, 256.0f);
    ImGui::SliderFloat("Specular strength", &object.specularStrength, 0.0f, 4.0f);
    ImGui::SliderFloat("Emissive strength", &object.emissiveStrength, 0.0f, 8.0f);

    std::vector<const char*> names;
    for (const Material& material : materials) {
        if (material.showInUi) names.push_back(material.name.c_str());
    }
    if (!names.empty()) {
        object.materialIndex = std::clamp(object.materialIndex, 0, int(names.size()) - 1);
        ImGui::Combo("Material / albedo", &object.materialIndex, names.data(), int(names.size()));
    }
    if (&object == &orbitSphere) {
        ImGui::Checkbox("Animated multi-sampling (push constant)", &object.animatedSampling);
    }
    ImGui::TreePop();
}

void buildUI() {
    ImGui::Begin("Vulkan Texturing Lab");
    if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::DragFloat3("Position", glm::value_ptr(cameraPosition), 0.05f);
        ImGui::SliderFloat("Yaw", &cameraYaw, -180.0f, 180.0f);
        ImGui::SliderFloat("Pitch", &cameraPitch, -89.0f, 89.0f);
        ImGui::SliderFloat("Field of view", &fovDegrees, 20.0f, 100.0f);
        ImGui::Checkbox("Perspective projection", &usePerspective);
    }
    if (ImGui::CollapsingHeader("Lighting", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::ColorEdit3("Ambient color", glm::value_ptr(ambientColor));
        ImGui::DragFloat3("Directional direction", glm::value_ptr(directionalDirection), 0.02f);
        ImGui::ColorEdit3("Directional color", glm::value_ptr(directionalColor));
        ImGui::SliderFloat("Directional intensity", &directionalIntensity, 0.0f, 5.0f);
    }
    if (ImGui::CollapsingHeader("Objects / materials", ImGuiTreeNodeFlags_DefaultOpen)) {
        buildObjectUI("Main sphere", sphere);
        buildObjectUI("Orbit sphere", orbitSphere);
        ImGui::Checkbox("Enable animated sampling globally", &enableAnimatedSampling);
        ImGui::Checkbox("Animate orbit", &animateOrbit);
        ImGui::SliderFloat("Orbit radius", &orbitRadius, 0.5f, 4.0f);
        ImGui::SliderFloat("Orbit speed", &orbitSpeed, 0.0f, 3.0f);
    }
    if (ImGui::CollapsingHeader("Point lights", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (pointLights.size() < kMaxLightsPerType && ImGui::Button("Add point light")) {
            pointLights.emplace_back();
            pointLights.back().position.x = float(pointLights.size()) * 0.75f;
            buffersDirty = true;
        }
        for (size_t i = 0; i < pointLights.size(); ++i) {
            ImGui::PushID(int(i));
            const std::string label = "Point " + std::to_string(i);
            if (ImGui::TreeNode(label.c_str())) {
                PointLight& light = pointLights[i];
                buffersDirty |= ImGui::DragFloat3("Position", glm::value_ptr(light.position), 0.03f);
                buffersDirty |= ImGui::ColorEdit3("Color", glm::value_ptr(light.color));
                buffersDirty |= ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 100.0f);
                if (ImGui::Button("Remove point")) {
                    pointLights.erase(pointLights.begin() + std::ptrdiff_t(i));
                    buffersDirty = true;
                    ImGui::TreePop();
                    ImGui::PopID();
                    break;
                }
                ImGui::TreePop();
            }
            ImGui::PopID();
        }
    }
    if (ImGui::CollapsingHeader("Spotlights", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (spotLights.size() < kMaxLightsPerType && ImGui::Button("Add spotlight")) {
            spotLights.emplace_back();
            spotLights.back().position.x = -float(spotLights.size());
            buffersDirty = true;
        }
        for (size_t i = 0; i < spotLights.size(); ++i) {
            ImGui::PushID(int(i));
            const std::string label = "Spot " + std::to_string(i);
            if (ImGui::TreeNode(label.c_str())) {
                SpotLight& light = spotLights[i];
                buffersDirty |= ImGui::DragFloat3("Position", glm::value_ptr(light.position), 0.03f);
                buffersDirty |= ImGui::DragFloat3("Direction", glm::value_ptr(light.direction), 0.02f);
                buffersDirty |= ImGui::ColorEdit3("Color", glm::value_ptr(light.color));
                buffersDirty |= ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 150.0f);
                buffersDirty |= ImGui::SliderFloat("Inner angle", &light.innerDegrees, 1.0f, 85.0f);
                buffersDirty |= ImGui::SliderFloat("Outer angle", &light.outerDegrees, 1.0f, 89.0f);
                if (light.outerDegrees < light.innerDegrees) light.outerDegrees = light.innerDegrees;
                if (ImGui::Button("Remove spotlight")) {
                    spotLights.erase(spotLights.begin() + std::ptrdiff_t(i));
                    buffersDirty = true;
                    ImGui::TreePop();
                    ImGui::PopID();
                    break;
                }
                ImGui::TreePop();
            }
            ImGui::PopID();
        }
    }
    ImGui::Checkbox("Show light markers and spotlight cones", &showLightMarkers);
    ImGui::Text("Materials: %zu | Point lights: %zu | Spotlights: %zu",
                materials.size() - 1, pointLights.size(), spotLights.size());
    ImGui::End();
}

} // namespace

bool application::initialize() {
    generateSphere(1.0f, 48, 64);
    if (vertices.empty() || indices.empty()) return false;

    if (!uploadDeviceBuffer(vertices.data(), VkDeviceSize(vertices.size() * sizeof(Vertex)),
                            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertexBuffer) ||
        !uploadDeviceBuffer(indices.data(), VkDeviceSize(indices.size() * sizeof(uint32_t)),
                            VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indexBuffer) ||
        !createHostBuffer(sizeof(SceneUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sceneBuffer)) {
        std::cerr << "Failed to create geometry or scene buffers\n";
        return false;
    }

    if (!createSolidTexture({0, 0, 0, 255}, false, blackTextureIndex) ||
        !createSolidTexture({255, 255, 255, 255}, true, whiteTextureIndex)) return false;

    if (!addMaterial("Checker", "textures/checker.png", "", "") ||
        !addMaterial("Stone", "textures/stone.png", "", "") ||
        !addMaterial("Metal", "textures/metal.png", "textures/metal_specular.png", "") ||
        !addMaterial("Neon", "textures/neon.png", "", "textures/neon_emissive.png")) {
        std::cerr << "Could not load one or more material textures. Check the textures/ folder.\n";
        return false;
    }

    Material markerMaterial;
    markerMaterial.name = "Internal white marker";
    markerMaterial.albedoTexture = whiteTextureIndex;
    markerMaterial.specularTexture = blackTextureIndex;
    markerMaterial.emissiveTexture = blackTextureIndex;
    markerMaterial.showInUi = false;
    markerMaterialIndex = int(materials.size());
    materials.push_back(markerMaterial);

    VkDescriptorSetLayoutBinding sceneBindings[3]{};
    sceneBindings[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    sceneBindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    sceneBindings[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo sceneLayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    sceneLayoutInfo.bindingCount = 3;
    sceneLayoutInfo.pBindings = sceneBindings;
    if (vkCreateDescriptorSetLayout(context.device, &sceneLayoutInfo, nullptr, &sceneSetLayout) != VK_SUCCESS ||
        !createTextureDescriptorLayout()) return false;

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(context.physical_device, &properties);
    dynamicModelStride = alignUp(sizeof(ModelUBO), properties.limits.minUniformBufferOffsetAlignment);
    const VkDeviceSize dynamicBufferSize = dynamicModelStride * kMaxDraws;
    if (!createHostBuffer(dynamicBufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, dynamicModelBuffer) ||
        vmaMapMemory(context.allocator, dynamicModelBuffer.allocation, &dynamicModelMapped) != VK_SUCCESS) {
        std::cerr << "Failed to create/map the dynamic model UBO\n";
        return false;
    }

    std::array<VkDescriptorPoolSize, 4> poolSizes{{
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, uint32_t(materials.size())},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, uint32_t(materials.size() * 3)}
    }};
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = uint32_t(materials.size() + 1);
    poolInfo.poolSizeCount = uint32_t(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    if (vkCreateDescriptorPool(context.device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) return false;

    VkDescriptorSetAllocateInfo sceneSetInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    sceneSetInfo.descriptorPool = descriptorPool;
    sceneSetInfo.descriptorSetCount = 1;
    sceneSetInfo.pSetLayouts = &sceneSetLayout;
    if (vkAllocateDescriptorSets(context.device, &sceneSetInfo, &sceneSet) != VK_SUCCESS) return false;

    VkDescriptorBufferInfo sceneBufferInfo{sceneBuffer.handle, 0, sizeof(SceneUBO)};
    VkWriteDescriptorSet sceneWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    sceneWrite.dstSet = sceneSet;
    sceneWrite.dstBinding = 0;
    sceneWrite.descriptorCount = 1;
    sceneWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sceneWrite.pBufferInfo = &sceneBufferInfo;
    vkUpdateDescriptorSets(context.device, 1, &sceneWrite, 0, nullptr);

    if (!createMaterialDescriptorSets()) return false;

    if (!createPipeline()) return false;

    sphere.materialIndex = 0;
    sphere.tint = glm::vec4(1.0f);
    sphere.shininess = 64.0f;
    sphere.specularStrength = 0.35f;
    sphere.emissiveStrength = 0.0f;

    orbitSphere.materialIndex = 3;
    orbitSphere.scale = glm::vec3(0.42f);
    orbitSphere.tint = glm::vec4(1.0f);
    orbitSphere.shininess = 96.0f;
    orbitSphere.specularStrength = 0.25f;
    orbitSphere.emissiveStrength = 2.0f;
    orbitSphere.animatedSampling = true;

    pointLights.emplace_back();
    spotLights.emplace_back();
    updateSceneDescriptorBuffers();
    return true;
}

void application::update(double time) {
    static double previous = 0.0;
    const float dt = previous == 0.0 ? 0.0f : float(time - previous);
    previous = time;
    const float clampedDt = std::clamp(dt, 0.0f, 0.1f);
    animationTime += clampedDt;
    if (animateOrbit) {
        orbitSphere.rot.y += clampedDt * 45.0f * orbitSpeed;
    }

    buildUI();
    if (buffersDirty) updateSceneDescriptorBuffers();

    const float yaw = glm::radians(cameraYaw);
    const float pitch = glm::radians(cameraPitch);
    const glm::vec3 forward = glm::normalize(glm::vec3(
        std::cos(yaw) * std::cos(pitch),
        std::sin(pitch),
        std::sin(yaw) * std::cos(pitch)
    ));
    const float aspect = float(context.swapchain_extent.width) /
                         float(std::max(1u, context.swapchain_extent.height));

    SceneUBO scene{};
    scene.view = glm::lookAt(cameraPosition, cameraPosition + forward, glm::vec3(0, 1, 0));
    scene.proj = usePerspective
        ? glm::perspective(glm::radians(fovDegrees), aspect, 0.1f, 100.0f)
        : glm::ortho(-aspect * 3.0f, aspect * 3.0f, -3.0f, 3.0f, 0.1f, 100.0f);
    scene.proj[1][1] *= -1.0f;
    scene.cameraPos = glm::vec4(cameraPosition, 1.0f);
    scene.ambientColor = glm::vec4(ambientColor, 1.0f);
    const glm::vec3 direction = glm::length(directionalDirection) > 0.0001f
        ? glm::normalize(directionalDirection) : glm::vec3(0, -1, 0);
    scene.directionalDirection = glm::vec4(direction, 0.0f);
    scene.directionalColor = glm::vec4(directionalColor, directionalIntensity);
    uploadBuffer(sceneBuffer, &scene, sizeof(scene));
}

void application::render(const graphics::internal::FrameData& frameData) {
    VkCommandBuffer cmd = frameData.command_buffer;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

    VkViewport viewport{0.0f, 0.0f, float(context.swapchain_extent.width),
                        float(context.swapchain_extent.height), 0.0f, 1.0f};
    VkRect2D scissor{{0, 0}, context.swapchain_extent};
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
                            0, 1, &sceneSet, 0, nullptr);

    const VkDeviceSize vertexOffset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuffer.handle, &vertexOffset);
    vkCmdBindIndexBuffer(cmd, indexBuffer.handle, 0, VK_INDEX_TYPE_UINT32);

    dynamicDrawSlot = 0;
    drawObject(cmd, sphere, sphere.pos, float(animationTime));
    const float orbitAngle = animationTime * orbitSpeed;
    const glm::vec3 orbitPosition = sphere.pos + (animateOrbit
        ? glm::vec3(std::cos(orbitAngle) * orbitRadius, 0.0f, std::sin(orbitAngle) * orbitRadius)
        : glm::vec3(orbitRadius, 0.0f, 0.0f));
    drawObject(cmd, orbitSphere, orbitPosition, float(animationTime));

    if (showLightMarkers && markerMaterialIndex >= 0) {
        SceneObject marker;
        marker.scale = glm::vec3(0.09f);
        marker.materialIndex = markerMaterialIndex;
        marker.shininess = 1.0f;
        marker.specularStrength = 0.0f;
        marker.emissiveStrength = 0.0f;
        marker.tint.a = 1.0f;

        for (const PointLight& light : pointLights) {
            marker.tint = glm::vec4(light.color, 1.0f);
            const uint32_t offset = writeDynamicModel(makeModel(marker, light.position), marker.tint,
                                                       1.0f, 0.0f, true, 0.0f);
            const Material& mat = materials[size_t(markerMaterialIndex)];
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
                                    1, 1, &mat.descriptorSet, 1, &offset);
            const SamplingPushConstants push{};
            vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
            vkCmdDrawIndexed(cmd, uint32_t(indices.size()), 1, 0, 0, 0);
        }

        std::vector<Vertex> lineVertices;
        std::vector<std::pair<uint32_t, uint32_t>> lineRanges;
        constexpr int segments = 24;
        for (const SpotLight& light : spotLights) {
            const glm::vec3 direction = glm::length(light.direction) > 0.0001f
                ? glm::normalize(light.direction) : glm::vec3(0, -1, 0);
            const glm::vec3 helper = std::abs(glm::dot(direction, glm::vec3(0, 1, 0))) > 0.95f
                ? glm::vec3(1, 0, 0) : glm::vec3(0, 1, 0);
            const glm::vec3 right = glm::normalize(glm::cross(direction, helper));
            const glm::vec3 up = glm::normalize(glm::cross(right, direction));
            const float length = 1.25f;
            const float radius = std::tan(glm::radians(light.outerDegrees)) * length;
            const glm::vec3 apex = light.position;
            const glm::vec3 center = apex + direction * length;
            const uint32_t first = uint32_t(lineVertices.size());
            auto addLine = [&](const glm::vec3& a, const glm::vec3& b) {
                lineVertices.push_back({a, {0, 1, 0}, {0, 0}});
                lineVertices.push_back({b, {0, 1, 0}, {0, 0}});
            };
            addLine(apex, center);
            for (int segment = 0; segment < segments; ++segment) {
                const float a0 = 2.0f * glm::pi<float>() * float(segment) / float(segments);
                const float a1 = 2.0f * glm::pi<float>() * float(segment + 1) / float(segments);
                const glm::vec3 p0 = center + radius * (std::cos(a0) * right + std::sin(a0) * up);
                const glm::vec3 p1 = center + radius * (std::cos(a1) * right + std::sin(a1) * up);
                addLine(p0, p1);
                if (segment % 3 == 0) addLine(apex, p0);
            }
            lineRanges.emplace_back(first, uint32_t(lineVertices.size()) - first);
        }

        if (!lineVertices.empty()) {
            const size_t bytes = lineVertices.size() * sizeof(Vertex);
            if (bytes > lineBufferCapacity) {
                destroyBuffer(lineBuffer);
                lineBufferCapacity = std::max<size_t>(bytes, 4096);
                if (!createHostBuffer(lineBufferCapacity, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, lineBuffer)) {
                    lineBufferCapacity = 0;
                }
            }
            if (lineBuffer.handle && uploadBuffer(lineBuffer, lineVertices.data(), bytes)) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, linePipeline);
                const VkDeviceSize lineOffset = 0;
                vkCmdBindVertexBuffers(cmd, 0, 1, &lineBuffer.handle, &lineOffset);
                const Material& mat = materials[size_t(markerMaterialIndex)];
                for (size_t i = 0; i < spotLights.size(); ++i) {
                    const glm::vec4 tint(spotLights[i].color, 1.0f);
                    const uint32_t offset = writeDynamicModel(glm::mat4(1.0f), tint,
                                                               1.0f, 0.0f, true, 0.0f);
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
                                            1, 1, &mat.descriptorSet, 1, &offset);
                    const SamplingPushConstants push{};
                    vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
                                       0, sizeof(push), &push);
                    vkCmdDraw(cmd, lineRanges[i].second, 1, lineRanges[i].first, 0);
                }
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
                vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuffer.handle, &vertexOffset);
                vkCmdBindIndexBuffer(cmd, indexBuffer.handle, 0, VK_INDEX_TYPE_UINT32);
            }
        }
    }

    if (dynamicDrawSlot > 0) {
        const VkDeviceSize usedBytes = VkDeviceSize(dynamicDrawSlot) * dynamicModelStride;
        vmaFlushAllocation(context.allocator, dynamicModelBuffer.allocation, 0, usedBytes);
    }
}

void application::shutdown() {
    if (context.device == VK_NULL_HANDLE) return;
    vkDeviceWaitIdle(context.device);

    if (dynamicModelMapped) {
        vmaUnmapMemory(context.allocator, dynamicModelBuffer.allocation);
        dynamicModelMapped = nullptr;
    }
    destroyBuffer(dynamicModelBuffer);
    destroyBuffer(vertexBuffer);
    destroyBuffer(indexBuffer);
    destroyBuffer(sceneBuffer);
    destroyBuffer(pointBuffer);
    destroyBuffer(spotBuffer);
    destroyBuffer(lineBuffer);

    if (descriptorPool) vkDestroyDescriptorPool(context.device, descriptorPool, nullptr);
    if (graphicsPipeline) vkDestroyPipeline(context.device, graphicsPipeline, nullptr);
    if (linePipeline) vkDestroyPipeline(context.device, linePipeline, nullptr);
    if (pipelineLayout) vkDestroyPipelineLayout(context.device, pipelineLayout, nullptr);
    if (sceneSetLayout) vkDestroyDescriptorSetLayout(context.device, sceneSetLayout, nullptr);
    if (materialSetLayout) vkDestroyDescriptorSetLayout(context.device, materialSetLayout, nullptr);

    for (Texture& texture : textures) destroyTexture(texture);
    textures.clear();
    materials.clear();
}
