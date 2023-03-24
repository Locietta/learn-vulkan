#include "app.h"
#include "geometry.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <fmt/core.h>
#include <fstream>
#include <glm/gtc/matrix_transform.hpp>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <stb_image.h>
#include <stdexcept>

namespace {

/// Private Classes & Functions

struct QueueFamilyIndices {
    uint32_t graphics_family;
    uint32_t present_family;
    uint32_t transfer_family;
};

struct UniformBufferObject {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};

struct QueueFamilyIndicesOptional {
    std::optional<uint32_t> graphics_family;
    std::optional<uint32_t> present_family;
    std::optional<uint32_t> transfer_family;

    [[nodiscard]] bool isComplete() const noexcept {
        return graphics_family.has_value() && present_family.has_value() && transfer_family.has_value();
    }
    [[nodiscard]] QueueFamilyIndices value() const noexcept {
        assert(isComplete());
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        return {graphics_family.value(), present_family.value(), transfer_family.value()};
    }
};

QueueFamilyIndicesOptional findQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface) {
    QueueFamilyIndicesOptional queue_indices;

    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.data());

    auto present_support = VkBool32(false);

    for (int i = 0; const auto &queue_family : queue_families) {
        if ((queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0u) {
            queue_indices.graphics_family = i;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &present_support);
            if (bool(present_support)) queue_indices.present_family = i;

            if (queue_indices.isComplete()) break;
        } else if ((queue_family.queueFlags & VK_QUEUE_TRANSFER_BIT) != 0u) {
            queue_indices.transfer_family = i;
        }
        i++;
    }

    if (!queue_indices.transfer_family.has_value()) {
        // use graphics queue as transfer queue, if there's no separate one
        queue_indices.transfer_family = queue_indices.graphics_family;
    }

    return queue_indices;
}

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> present_modes;
};

SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device, VkSurfaceKHR surface) {
    SwapChainSupportDetails details;

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

    uint32_t format_count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, nullptr);
    if (format_count != 0) {
        details.formats.resize(format_count);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, details.formats.data());
    }

    uint32_t present_mode_count = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &present_mode_count, nullptr);
    if (present_mode_count != 0) {
        details.present_modes.resize(present_mode_count);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &present_mode_count, details.present_modes.data());
    }

    return details;
}

bool checkSwapChainAdequate(VkPhysicalDevice device, VkSurfaceKHR surface) {
    SwapChainSupportDetails swapchain_support = querySwapChainSupport(device, surface);
    return !swapchain_support.formats.empty() && !swapchain_support.present_modes.empty();
}

bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
    uint32_t extension_count;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, nullptr);

    std::vector<VkExtensionProperties> available_extensions(extension_count);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, available_extensions.data());

    std::set<std::string_view> required_extensions(App::REQUIRED_EXTENSIONS.begin(), App::REQUIRED_EXTENSIONS.end());

    for (const auto &extension : available_extensions) {
        required_extensions.erase(extension.extensionName);
    }

    return required_extensions.empty();
}

bool checkValidationLayerSupport() {
    uint32_t layer_count;
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
    std::vector<VkLayerProperties> available_layers(layer_count);
    vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());

    for (const std::string_view layer_name : App::VALIDATION_LAYERS) {
        bool layer_found = false;
        for (const auto &available_layer : available_layers) {
            if (layer_name == available_layer.layerName) {
                layer_found = true;
                break;
            }
        }
        if (!layer_found) return false;
    }

    return true;
}

int rateDeviceSuitability(VkPhysicalDevice device, VkSurfaceKHR surface) {
    VkPhysicalDeviceProperties device_properties;
    vkGetPhysicalDeviceProperties(device, &device_properties);
    VkPhysicalDeviceFeatures device_features;
    vkGetPhysicalDeviceFeatures(device, &device_features);

    if (device_features.geometryShader == VK_FALSE || device_features.samplerAnisotropy == VK_FALSE) {
        return 0; // forciably requires GS and anisotropic filtering
    }

    if (!checkDeviceExtensionSupport(device)) return 0;
    if (!checkSwapChainAdequate(device, surface)) return 0;

    const QueueFamilyIndicesOptional queue_indices = findQueueFamilies(device, surface);
    if (!queue_indices.isComplete()) return 0;

    int score = 0;

    if (device_properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) score += 2000;
    score += static_cast<int>(device_properties.limits.maxImageDimension3D);

    return score;
}

VkSurfaceFormatKHR chooseSwapSurfaceFormat(std::span<const VkSurfaceFormatKHR> available_formats) {
    if (available_formats.size() == 1 && available_formats[0].format == VK_FORMAT_UNDEFINED) {
        return {VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
    }

    for (const auto &available_format : available_formats) {
        if (available_format.format == VK_FORMAT_B8G8R8A8_UNORM &&
            available_format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return available_format;
        }
    }

    return available_formats[0];
}

VkPresentModeKHR chooseSwapPresentMode(std::span<const VkPresentModeKHR> available_present_modes) {
    for (const auto available_present_mode : available_present_modes) {
        if (available_present_mode == VK_PRESENT_MODE_MAILBOX_KHR) return available_present_mode;
    }

    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D chooseSwapExtent(GLFWwindow *window, const VkSurfaceCapabilitiesKHR &capa) {
    if (capa.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return capa.currentExtent;
    }

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    const auto actual_extent_width = std::clamp(uint32_t(width), capa.minImageExtent.width, capa.maxImageExtent.width);
    const auto actual_extent_height =
        std::clamp(uint32_t(height), capa.minImageExtent.height, capa.maxImageExtent.height);

    return {actual_extent_width, actual_extent_height};
}

VkShaderModule createShaderModule(VkDevice device, std::span<const std::byte> code) {
    const VkShaderModuleCreateInfo create_info{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code.size(),
        .pCode = reinterpret_cast<const uint32_t *>(code.data()),
    };

    VkShaderModule shader_module;
    if (vkCreateShaderModule(device, &create_info, nullptr, &shader_module) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module!");
    }

    return shader_module;
}

std::vector<std::byte> readFile(const std::string &filename) {
    std::basic_ifstream<std::byte> file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file!");
    }

    const auto file_size = file.tellg();
    std::vector<std::byte> buffer(file_size);

    file.seekg(0);
    file.read(buffer.data(), file_size);
    file.close();

    return buffer;
}

uint32_t findMemoryType(VkPhysicalDevice physical_device, uint32_t type_filter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties mem_properties;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);

    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
        if (((type_filter & (1 << i)) != 0u) && // XXX: why is this necessary?
            (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("Failed to find suitable memory type!");
}

constexpr VkVertexInputBindingDescription BINDING_DESCRIPTION{
    .binding = 0,
    .stride = sizeof(Vertex),
    .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
};

constexpr std::array ATTRIBUTE_DESCRIPTIONS{
    VkVertexInputAttributeDescription{
                                      .location = 0,
                                      .binding = 0,
                                      .format = VK_FORMAT_R32G32B32_SFLOAT,
                                      .offset = offsetof(Vertex,   pos),
                                      },
    VkVertexInputAttributeDescription{
                                      .location = 1,
                                      .binding = 0,
                                      .format = VK_FORMAT_R32G32B32_SFLOAT,
                                      .offset = offsetof(Vertex, color),
                                      },

    VkVertexInputAttributeDescription{
                                      .location = 2,
                                      .binding = 0,
                                      .format = VK_FORMAT_R32G32_SFLOAT,
                                      .offset = offsetof(Vertex,    uv),
                                      },
};

constexpr std::array VERTICES{
    Vertex{{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
    Vertex{ {0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
    Vertex{  {0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
    Vertex{ {-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}},
};

constexpr std::array INDICES{
    0u, 1u, 2u, 2u, 3u, 0u,
};

QueueFamilyIndices queue_indices;

} // namespace

void App::framebufferResizeCallback(GLFWwindow *window, int /* width */, int /* height */) {
    auto *app = reinterpret_cast<App *>(glfwGetWindowUserPointer(window));
    app->framebuffer_resized_ = true;
}

App::App() {
    initWindow();
    initVulkan();
}

App::~App() {
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        vkDestroySemaphore(device_, image_available_semaphores_[i], nullptr);
        vkDestroySemaphore(device_, render_finished_semaphores_[i], nullptr);
        vkDestroyFence(device_, in_flight_fences_[i], nullptr);
    }

    for (VkCommandPool command_pool : command_pools_) {
        vkDestroyCommandPool(device_, command_pool, nullptr);
    }

    cleanupSwapchain();

    vkDestroySampler(device_, texture_sampler_, nullptr);
    vkDestroyImageView(device_, texture_image_view_, nullptr);

    vkDestroyImage(device_, texture_image_, nullptr);
    vkFreeMemory(device_, texture_image_memory_, nullptr);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        vkDestroyBuffer(device_, uniform_buffers_[i], nullptr);
        vkFreeMemory(device_, uniform_buffers_memory_[i], nullptr);
    }

    vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);

    vkDestroyDescriptorSetLayout(device_, descriptor_set_layout_, nullptr);

    vkDestroyBuffer(device_, index_buffer_, nullptr);
    vkFreeMemory(device_, index_buffer_memory_, nullptr);

    vkDestroyBuffer(device_, vertex_buffer_, nullptr);
    vkFreeMemory(device_, vertex_buffer_memory_, nullptr);

    vkDestroyPipeline(device_, graphics_pipeline_, nullptr);
    vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
    vkDestroyRenderPass(device_, render_pass_, nullptr);

    vkDestroyDevice(device_, nullptr);
    vkDestroySurfaceKHR(instance_, surface_, nullptr);
    vkDestroyInstance(instance_, nullptr);

    glfwDestroyWindow(window_);
    glfwTerminate();
}

void App::displayFPS() {
    static int frame_counter = 0;

    const double current_time = glfwGetTime();
    const double time_slice = current_time - last_time_;
    if (time_slice >= 1.0) {
        const double delta = time_slice * 1000.0 / double(frame_counter);
        const double fps = double(frame_counter) / time_slice;

        glfwSetWindowTitle(window_, fmt::format("Vulkan Rectangle - delta: {:.2f}ms FPS: {:.2f}", delta, fps).c_str());

        frame_counter = 0;
        last_time_ = current_time;
    }

    frame_counter++;
}

void App::run() {
    while (glfwWindowShouldClose(window_) == 0) {
        glfwPollEvents();
        drawFrame();

        displayFPS();
    }

    vkDeviceWaitIdle(device_);
}

void App::drawFrame() {
    vkWaitForFences(device_, 1, &in_flight_fences_[current_frame_], VK_TRUE, UINT64_MAX);

    uint32_t image_index;

    if (auto result = vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX,
                                            image_available_semaphores_[current_frame_], VK_NULL_HANDLE, &image_index);
        result == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapchain();
    } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("Failed to acquire swapchain image!");
    }

    updateUniformBuffer(current_frame_);

    vkResetCommandPool(device_, command_pools_[current_frame_], 0);
    recordCommandBuffer(command_buffers_[current_frame_], image_index);

    const VkPipelineStageFlags wait_stages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};

    const VkSubmitInfo submit_info{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &image_available_semaphores_[current_frame_],
        .pWaitDstStageMask = wait_stages,
        .commandBufferCount = 1,
        .pCommandBuffers = &command_buffers_[current_frame_],
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &render_finished_semaphores_[current_frame_],
    };

    vkResetFences(device_, 1, &in_flight_fences_[current_frame_]);
    if (vkQueueSubmit(graphics_queue_, 1, &submit_info, in_flight_fences_[current_frame_]) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit draw command buffer!");
    }

    const VkPresentInfoKHR present_info{
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &render_finished_semaphores_[current_frame_],
        .swapchainCount = 1,
        .pSwapchains = &swapchain_,
        .pImageIndices = &image_index,
        .pResults = nullptr, // Optional
    };

    if (auto result = vkQueuePresentKHR(present_queue_, &present_info);
        result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebuffer_resized_) {
        framebuffer_resized_ = false;
        recreateSwapchain();
    } else if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to present swapchain image!");
    }

    current_frame_ = (current_frame_ + 1) % MAX_FRAMES_IN_FLIGHT;
}

void App::initWindow() {
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // tell glfw not to create opengl context

    window_ = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan Rectangle", nullptr, nullptr);

    glfwSetWindowUserPointer(window_, this);
    glfwSetFramebufferSizeCallback(window_, framebufferResizeCallback);

    last_time_ = glfwGetTime();
}

void App::initVulkan() {
    createInstance();

    /// Create Window Surface
    if (glfwCreateWindowSurface(instance_, window_, nullptr, &surface_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create window surface!");
    }

    createPhysicalDevice();
    createLogicalDevice();

    createSwapchain();
    createImageViews();

    createRenderPass();
    createDescriptorSetLayout();
    createGraphicsPipeline();
    createFramebuffers();

    createCommandPool();
    createTextureImage();
    createTextureImageView();
    createTextureSampler();
    createVertexBuffer();
    createIndexBuffer();
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createCommandBuffers();

    createSyncObjects();
}

void App::createInstance() {
    if (ENABLE_VALIDATION_LAYERS && !checkValidationLayerSupport()) {
        throw std::runtime_error("Validation layers requested, but not available!");
    }

    const VkApplicationInfo app_info{
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "Hello Triangle",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "No Engine",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_3,
    };

    uint32_t vk_extension_count = 0;
    const char **vk_extensions = glfwGetRequiredInstanceExtensions(&vk_extension_count);

    const VkInstanceCreateInfo create_info{
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info,
        .enabledLayerCount = ENABLE_VALIDATION_LAYERS ? VALIDATION_LAYERS.size() : 0,
        .ppEnabledLayerNames = ENABLE_VALIDATION_LAYERS ? VALIDATION_LAYERS.data() : nullptr,
        .enabledExtensionCount = vk_extension_count,
        .ppEnabledExtensionNames = vk_extensions,
    };

    if (vkCreateInstance(&create_info, nullptr, &this->instance_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create VK instance!");
    }
}

void App::createPhysicalDevice() {
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);
    if (device_count == 0) {
        throw std::runtime_error("Failed to find GPUs with Vulkan support!");
    }
    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());

    std::multimap<int, VkPhysicalDevice> candidates;

    for (const auto &device : devices) {
        const int score = rateDeviceSuitability(device, surface_);
        if (score > 0) {
            candidates.emplace(score, device);
        }
    }

    if (candidates.empty()) {
        throw std::runtime_error("Failed to find a suitable GPU!");
    }

    physical_device_ = candidates.rbegin()->second;
}

void App::createLogicalDevice() {
    const auto queue_indices_optional = findQueueFamilies(physical_device_, surface_);

    if (!queue_indices_optional.isComplete()) {
        throw std::runtime_error("Failed to find appropriate queue families!");
    }

    queue_indices = queue_indices_optional.value();

    std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
    {
        const std::set<uint32_t> unique_queue_families{
            queue_indices.graphics_family,
            queue_indices.present_family,
            queue_indices.transfer_family,
        };

        float queue_priority = 1.0f;

        for (const auto queue_family : unique_queue_families) {
            VkDeviceQueueCreateInfo queue_create_info{
                .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                .queueFamilyIndex = queue_family,
                .queueCount = 1,
                .pQueuePriorities = &queue_priority,
            };
            queue_create_infos.push_back(queue_create_info);
        }
    }

    const VkPhysicalDeviceFeatures device_features{
        .samplerAnisotropy = VK_TRUE,
    };

    const VkDeviceCreateInfo device_create_info{
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size()),
        .pQueueCreateInfos = queue_create_infos.data(),
        .enabledExtensionCount = REQUIRED_EXTENSIONS.size(),
        .ppEnabledExtensionNames = REQUIRED_EXTENSIONS.data(),
        .pEnabledFeatures = &device_features,
    };

    if (vkCreateDevice(physical_device_, &device_create_info, nullptr, &device_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create logical device!");
    }

    /// Retrive handles of queues

    vkGetDeviceQueue(device_, queue_indices.graphics_family, 0, &graphics_queue_);
    vkGetDeviceQueue(device_, queue_indices.present_family, 0, &present_queue_);
    vkGetDeviceQueue(device_, queue_indices.transfer_family, 0, &transfer_queue_);
}

void App::createSwapchain() {
    const SwapChainSupportDetails swap_chain_support = querySwapChainSupport(physical_device_, surface_);

    const VkSurfaceFormatKHR surface_format = chooseSwapSurfaceFormat(swap_chain_support.formats);
    const VkPresentModeKHR present_mode = chooseSwapPresentMode(swap_chain_support.present_modes);
    const VkExtent2D extent = chooseSwapExtent(window_, swap_chain_support.capabilities);

    uint32_t image_count = swap_chain_support.capabilities.minImageCount + 1;
    if (swap_chain_support.capabilities.maxImageCount > 0 &&
        image_count > swap_chain_support.capabilities.maxImageCount) {
        image_count = swap_chain_support.capabilities.maxImageCount;
    }

    const std::array unique_queue_families = {
        queue_indices.graphics_family,
        queue_indices.present_family,
    };

    const bool is_same_graphic_present_queue = unique_queue_families[0] == unique_queue_families[1];

    const VkSwapchainCreateInfoKHR swapchain_create_info{
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = surface_,
        .minImageCount = image_count,
        .imageFormat = surface_format.format,
        .imageColorSpace = surface_format.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .imageSharingMode = is_same_graphic_present_queue ? VK_SHARING_MODE_EXCLUSIVE : VK_SHARING_MODE_CONCURRENT,
        .queueFamilyIndexCount = is_same_graphic_present_queue ? 0u : 2u,
        .pQueueFamilyIndices = is_same_graphic_present_queue ? nullptr : unique_queue_families.data(),
        .preTransform = swap_chain_support.capabilities.currentTransform,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = present_mode,
        .clipped = VK_TRUE,
        .oldSwapchain = VK_NULL_HANDLE,
    };

    if (vkCreateSwapchainKHR(device_, &swapchain_create_info, nullptr, &swapchain_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create swap chain!");
    }

    vkGetSwapchainImagesKHR(device_, swapchain_, &image_count, nullptr);
    swapchain_images_.resize(image_count);
    vkGetSwapchainImagesKHR(device_, swapchain_, &image_count, swapchain_images_.data());

    swapchain_image_format_ = surface_format.format;
    swapchain_extent_ = extent;
}

void App::cleanupSwapchain() {
    for (VkFramebuffer framebuffer : swapchain_framebuffers_) {
        vkDestroyFramebuffer(device_, framebuffer, nullptr);
    }
    for (VkImageView image_view : swapchain_image_views_) {
        vkDestroyImageView(device_, image_view, nullptr);
    }
    vkDestroySwapchainKHR(device_, swapchain_, nullptr);
}

void App::recreateSwapchain() {
    { /// handle minimized window
        int width = 0, height = 0;
        glfwGetFramebufferSize(window_, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window_, &width, &height);
            glfwWaitEvents();
        }
    }

    vkDeviceWaitIdle(device_);

    cleanupSwapchain();

    createSwapchain();
    createImageViews();
    createFramebuffers();
}

VkImageView App::createImageView(VkImage image, VkFormat format) {
    const VkImageViewCreateInfo image_view_create_info{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = format,
        .components{
                    VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                    VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                    },
        .subresourceRange{
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                    },
    };

    VkImageView image_view;

    if (vkCreateImageView(device_, &image_view_create_info, nullptr, &image_view) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create image view!");
    }

    return image_view;
}

void App::createImageViews() {
    swapchain_image_views_.resize(swapchain_images_.size());

    for (size_t i = 0; i < swapchain_images_.size(); ++i) {
        swapchain_image_views_[i] = createImageView(swapchain_images_[i], swapchain_image_format_);
    }
}

void App::createRenderPass() {
    const VkAttachmentDescription color_attachment{
        .format = swapchain_image_format_,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
    };

    const VkAttachmentReference color_attachment_ref{
        .attachment = 0,
        .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };

    const VkSubpassDescription subpass{
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_attachment_ref,
    };

    const VkSubpassDependency dependency{
        .srcSubpass = VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
    };

    const VkRenderPassCreateInfo render_pass_create_info{
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &color_attachment,
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 1,
        .pDependencies = &dependency,
    };

    if (vkCreateRenderPass(device_, &render_pass_create_info, nullptr, &render_pass_) != VK_SUCCESS) {
        throw std::runtime_error("failed to create render pass!");
    }
}

void App::createDescriptorSetLayout() {
    const VkDescriptorSetLayoutBinding ubo_layout_binding{
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        .pImmutableSamplers = nullptr,
    };

    const VkDescriptorSetLayoutBinding sampler_layout_binding{
        .binding = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        .pImmutableSamplers = nullptr,
    };

    const std::array bindings = {ubo_layout_binding, sampler_layout_binding};

    const VkDescriptorSetLayoutCreateInfo layout_create_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = bindings.size(),
        .pBindings = bindings.data(),
    };

    if (vkCreateDescriptorSetLayout(device_, &layout_create_info, nullptr, &descriptor_set_layout_) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor set layout!");
    }
}

void App::createGraphicsPipeline() {
    constexpr static uint8_t VS_CODE[] = {
#include "base.vs.spv.h"
    };

    constexpr static uint8_t PS_CODE[] = {
#include "base.ps.spv.h"
    };

    VkShaderModule vs_module = createShaderModule(device_, std::as_bytes(std::span(VS_CODE)));
    VkShaderModule ps_module = createShaderModule(device_, std::as_bytes(std::span(PS_CODE)));

    const VkPipelineShaderStageCreateInfo vertex_shader_stage_create_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT,
        .module = vs_module,
        .pName = "main", // should be consistent with entry point in shader
    };

    const VkPipelineShaderStageCreateInfo pixel_shader_stage_create_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
        .module = ps_module,
        .pName = "main",
    };

    const std::array shader_stages = {vertex_shader_stage_create_info, pixel_shader_stage_create_info};

    const VkPipelineDynamicStateCreateInfo dynamic_state_create_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = DYNAMIC_STATES.size(),
        .pDynamicStates = DYNAMIC_STATES.data(),
    };

    const VkPipelineVertexInputStateCreateInfo vertex_input_create_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &BINDING_DESCRIPTION,
        .vertexAttributeDescriptionCount = ATTRIBUTE_DESCRIPTIONS.size(),
        .pVertexAttributeDescriptions = ATTRIBUTE_DESCRIPTIONS.data(),
    };

    const VkPipelineInputAssemblyStateCreateInfo input_assembly_create_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = VK_FALSE,
    };

    const VkPipelineViewportStateCreateInfo viewport_state{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1,
    };

    const VkPipelineRasterizationStateCreateInfo rasterizer_create_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .depthClampEnable = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_BACK_BIT,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .depthBiasEnable = VK_FALSE,
        .depthBiasConstantFactor = 0.0f, // Optional
        .depthBiasClamp = 0.0f,          // Optional
        .depthBiasSlopeFactor = 0.0f,    // Optional
        .lineWidth = 1.0f,
    };

    const VkPipelineMultisampleStateCreateInfo multisampling_create_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        .sampleShadingEnable = VK_FALSE,
        .minSampleShading = 1.0f,          // Optional
        .pSampleMask = nullptr,            // Optional
        .alphaToCoverageEnable = VK_FALSE, // Optional
        .alphaToOneEnable = VK_FALSE,      // Optional
    };

    // TODO: depth & stencil

    constexpr VkColorComponentFlags VK_RGBA =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    const VkPipelineColorBlendAttachmentState color_blend_attachment{
        .blendEnable = VK_FALSE,
        .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,  // Optional
        .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO, // Optional
        .colorBlendOp = VK_BLEND_OP_ADD,             // Optional
        .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,  // Optional
        .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO, // Optional
        .alphaBlendOp = VK_BLEND_OP_ADD,             // Optional
        .colorWriteMask = VK_RGBA,
    };

    const VkPipelineColorBlendStateCreateInfo color_blending_create_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .logicOpEnable = VK_FALSE,
        .logicOp = VK_LOGIC_OP_COPY, // Optional
        .attachmentCount = 1,
        .pAttachments = &color_blend_attachment,
        .blendConstants = {0.0f, 0.0f, 0.0f, 0.0f}, // Optional
    };

    const VkPipelineLayoutCreateInfo pipeline_layout_create_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &descriptor_set_layout_,
        .pushConstantRangeCount = 0,    // Optional
        .pPushConstantRanges = nullptr, // Optional
    };

    if (vkCreatePipelineLayout(device_, &pipeline_layout_create_info, nullptr, &pipeline_layout_) != VK_SUCCESS) {
        throw std::runtime_error("failed to create pipeline layout!");
    }

    const VkGraphicsPipelineCreateInfo pipeline_create_info{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = shader_stages.size(),
        .pStages = shader_stages.data(),
        .pVertexInputState = &vertex_input_create_info,
        .pInputAssemblyState = &input_assembly_create_info,
        .pViewportState = &viewport_state,
        .pRasterizationState = &rasterizer_create_info,
        .pMultisampleState = &multisampling_create_info,
        .pDepthStencilState = nullptr, // Optional
        .pColorBlendState = &color_blending_create_info,
        .pDynamicState = &dynamic_state_create_info,
        .layout = pipeline_layout_,
        .renderPass = render_pass_,
        .subpass = 0,                         // index of subpass to be used with this pipeline
        .basePipelineHandle = VK_NULL_HANDLE, // Optional
        .basePipelineIndex = -1,              // Optional
    };

    if (vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipeline_create_info, nullptr, &graphics_pipeline_) !=
        VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    vkDestroyShaderModule(device_, vs_module, nullptr);
    vkDestroyShaderModule(device_, ps_module, nullptr);
}

void App::createFramebuffers() {
    swapchain_framebuffers_.resize(swapchain_image_views_.size());

    for (size_t i = 0; i < swapchain_image_views_.size(); i++) {
        const std::array attachments = {swapchain_image_views_[i]};

        const VkFramebufferCreateInfo framebuffer_create_info{
            .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .renderPass = render_pass_,
            .attachmentCount = attachments.size(),
            .pAttachments = attachments.data(),
            .width = swapchain_extent_.width,
            .height = swapchain_extent_.height,
            .layers = 1,
        };

        if (vkCreateFramebuffer(device_, &framebuffer_create_info, nullptr, &swapchain_framebuffers_[i]) !=
            VK_SUCCESS) {
            throw std::runtime_error("failed to create framebuffer!");
        }
    }
}

void App::createCommandPool() {
    const VkCommandPoolCreateInfo pool_create_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = 0,
        .queueFamilyIndex = queue_indices.graphics_family,
    };

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        if (vkCreateCommandPool(device_, &pool_create_info, nullptr, command_pools_.data() + i) != VK_SUCCESS) {
            throw std::runtime_error("failed to create command pool!");
        }
    }

    const VkCommandPoolCreateInfo transfer_pool_create_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = 0,
        .queueFamilyIndex = queue_indices.transfer_family,
    };

    if (vkCreateCommandPool(device_, &transfer_pool_create_info, nullptr, &command_pools_.back()) != VK_SUCCESS) {
        throw std::runtime_error("failed to create command pool!");
    }
}

void App::createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage,
                      VkMemoryPropertyFlags properties, VkImage &image, VkDeviceMemory &image_memory) {

    const VkImageCreateInfo image_create_info{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = format,
        .extent = {.width = static_cast<uint32_t>(width), .height = static_cast<uint32_t>(height), .depth = 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = tiling,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };

    if (vkCreateImage(device_, &image_create_info, nullptr, &texture_image_) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture image!");
    }

    VkMemoryRequirements mem_requirements;
    vkGetImageMemoryRequirements(device_, texture_image_, &mem_requirements);

    const VkMemoryAllocateInfo alloc_info{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mem_requirements.size,
        .memoryTypeIndex = findMemoryType(physical_device_, mem_requirements.memoryTypeBits, properties),
    };

    if (vkAllocateMemory(device_, &alloc_info, nullptr, &image_memory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate texture image memory!");
    }
    vkBindImageMemory(device_, image, image_memory, 0);
}

void App::createTextureImage() {
    int tex_width, tex_height, tex_channels;
    stbi_uc *pixels = stbi_load("texture/kemomimi.png", &tex_width, &tex_height, &tex_channels, STBI_rgb_alpha);
    VkDeviceSize image_size = VkDeviceSize(tex_width) * tex_height * 4;

    if (pixels == nullptr) {
        throw std::runtime_error("failed to load texture image!");
    }

    VkBuffer staging_buffer;
    VkDeviceMemory staging_buffer_memory;
    createBuffer(image_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging_buffer,
                 staging_buffer_memory);

    void *data;
    vkMapMemory(device_, staging_buffer_memory, 0, image_size, 0, &data);
    memcpy(data, pixels, static_cast<size_t>(image_size));
    vkUnmapMemory(device_, staging_buffer_memory);

    stbi_image_free(pixels);

    createImage(tex_width, tex_height, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL,
                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                texture_image_, texture_image_memory_);

    transitionImageLayout(texture_image_, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    copyBufferToImage(staging_buffer, texture_image_, static_cast<uint32_t>(tex_width),
                      static_cast<uint32_t>(tex_height));

    vkDestroyBuffer(device_, staging_buffer, nullptr);
    vkFreeMemory(device_, staging_buffer_memory, nullptr);
}

void App::createTextureImageView() { texture_image_view_ = createImageView(texture_image_, VK_FORMAT_R8G8B8A8_SRGB); }

void App::createTextureSampler() {
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(physical_device_, &properties);

    const VkSamplerCreateInfo sampler_create_info{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .mipLodBias = 0.0f,
        .anisotropyEnable = VK_TRUE,
        .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
        .compareEnable = VK_FALSE,
        .compareOp = VK_COMPARE_OP_ALWAYS,
        .minLod = 0.0f,
        .maxLod = 0.0f,
        .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
    };

    if (vkCreateSampler(device_, &sampler_create_info, nullptr, &texture_sampler_) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture sampler!");
    }
}

void App::transitionImageLayout(VkImage image, VkFormat format, VkImageLayout old_layout, VkImageLayout new_layout) {
    VkCommandBuffer command_buffer = beginSingleTimeCommands();

    VkImageMemoryBarrier barrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .oldLayout = old_layout,
        .newLayout = new_layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                             .baseMipLevel = 0,
                             .levelCount = 1,
                             .baseArrayLayer = 0,
                             .layerCount = 1},
    };

    VkPipelineStageFlags source_stage;
    VkPipelineStageFlags destination_stage;

    if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED && new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        source_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destination_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
               new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        source_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destination_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else {
        throw std::invalid_argument("unsupported layout transition!");
    }

    vkCmdPipelineBarrier(command_buffer, source_stage, destination_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);

    endSingleTimeCommands(command_buffer);
}

void App::copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
    VkCommandBuffer command_buffer = beginSingleTimeCommands();

    VkBufferImageCopy region{
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                             .mipLevel = 0,
                             .baseArrayLayer = 0,
                             .layerCount = 1},
        .imageOffset = {.x = 0, .y = 0, .z = 0},
        .imageExtent = {.width = width, .height = height, .depth = 1},
    };

    vkCmdCopyBufferToImage(command_buffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    endSingleTimeCommands(command_buffer);
}

void App::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer &buffer,
                       VkDeviceMemory &buffer_memory) {
    const VkBufferCreateInfo buffer_create_info{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };

    if (vkCreateBuffer(device_, &buffer_create_info, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create buffer!");
    }

    VkMemoryRequirements mem_requirements;
    vkGetBufferMemoryRequirements(device_, buffer, &mem_requirements);

    const VkMemoryAllocateInfo alloc_info{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mem_requirements.size,
        .memoryTypeIndex = findMemoryType(physical_device_, mem_requirements.memoryTypeBits, properties),
    };

    if (vkAllocateMemory(device_, &alloc_info, nullptr, &buffer_memory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate buffer memory!");
    }

    vkBindBufferMemory(device_, buffer, buffer_memory, 0);
}

VkCommandBuffer App::beginSingleTimeCommands() const {
    const VkCommandBufferAllocateInfo alloc_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = command_pools_.back(), // command pool for host transferring
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };

    VkCommandBuffer command_buffer;
    vkAllocateCommandBuffers(device_, &alloc_info, &command_buffer);

    const VkCommandBufferBeginInfo begin_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };

    if (vkBeginCommandBuffer(command_buffer, &begin_info) != VK_SUCCESS) {
        throw std::runtime_error("failed to begin recording command buffer!");
    }

    return command_buffer;
}

void App::endSingleTimeCommands(VkCommandBuffer command_buffer) const {
    if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to record command buffer!");
    }

    const VkSubmitInfo submit_info{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &command_buffer,
    };

    vkQueueSubmit(transfer_queue_, 1, &submit_info, VK_NULL_HANDLE);
    vkQueueWaitIdle(transfer_queue_); // should use fence if multiple queues are used

    vkFreeCommandBuffers(device_, command_pools_.back(), 1, &command_buffer);
}

void App::copyBuffer(VkBuffer src_buffer, VkBuffer dst_buffer, VkDeviceSize size) {
    VkCommandBuffer command_buffer = beginSingleTimeCommands();

    const VkBufferCopy copy_region{
        .srcOffset = 0,
        .dstOffset = 0,
        .size = size,
    };
    vkCmdCopyBuffer(command_buffer, src_buffer, dst_buffer, 1, &copy_region);

    endSingleTimeCommands(command_buffer);
}

void App::createVertexBuffer() {
    const VkDeviceSize buffer_size = sizeof(VERTICES[0]) * VERTICES.size();

    VkBuffer staging_buffer;
    VkDeviceMemory staging_buffer_memory;

    createBuffer(buffer_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging_buffer,
                 staging_buffer_memory);

    void *data;
    vkMapMemory(device_, staging_buffer_memory, 0, buffer_size, 0, &data);
    memcpy(data, VERTICES.data(), buffer_size);
    vkUnmapMemory(device_, staging_buffer_memory);

    createBuffer(buffer_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertex_buffer_, vertex_buffer_memory_);

    copyBuffer(staging_buffer, vertex_buffer_, buffer_size);

    vkDestroyBuffer(device_, staging_buffer, nullptr);
    vkFreeMemory(device_, staging_buffer_memory, nullptr);
}

void App::createIndexBuffer() {
    const VkDeviceSize buffer_size = sizeof(INDICES[0]) * INDICES.size();

    VkBuffer staging_buffer;
    VkDeviceMemory staging_buffer_memory;

    createBuffer(buffer_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging_buffer,
                 staging_buffer_memory);

    void *data;
    vkMapMemory(device_, staging_buffer_memory, 0, buffer_size, 0, &data);
    memcpy(data, INDICES.data(), buffer_size);
    vkUnmapMemory(device_, staging_buffer_memory);

    createBuffer(buffer_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, index_buffer_, index_buffer_memory_);

    copyBuffer(staging_buffer, index_buffer_, buffer_size);

    vkDestroyBuffer(device_, staging_buffer, nullptr);
    vkFreeMemory(device_, staging_buffer_memory, nullptr);
}

void App::createUniformBuffers() {
    const VkDeviceSize buffer_size = sizeof(UniformBufferObject);

    uniform_buffers_.resize(MAX_FRAMES_IN_FLIGHT);
    uniform_buffers_memory_.resize(MAX_FRAMES_IN_FLIGHT);
    uniform_buffers_mapped_.resize(MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        createBuffer(buffer_size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniform_buffers_[i],
                     uniform_buffers_memory_[i]);
        vkMapMemory(device_, uniform_buffers_memory_[i], 0, buffer_size, 0, &uniform_buffers_mapped_[i]);
    }
    // TODO
}

void App::updateUniformBuffer(uint32_t current_image) {
    using namespace std::chrono;

    static auto start_time = high_resolution_clock::now();

    auto current_time = high_resolution_clock::now();
    float time = duration<float, seconds::period>(current_time - start_time).count();

    UniformBufferObject ubo{
        .model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
        .view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
        .proj = glm::perspective(glm::radians(45.0f), float(swapchain_extent_.width) / float(swapchain_extent_.height),
                                 0.1f, 10.0f),
    };

    ubo.proj[1][1] *= -1;

    memcpy(uniform_buffers_mapped_[current_image], &ubo, sizeof(ubo));
}

void App::createDescriptorPool() {
    const VkDescriptorPoolSize ub_pool_size{
        .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
    };

    const VkDescriptorPoolSize img_sampler_pool_size{
        .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
    };

    const std::array pool_sizes = {ub_pool_size, img_sampler_pool_size};

    const VkDescriptorPoolCreateInfo pool_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = MAX_FRAMES_IN_FLIGHT,
        .poolSizeCount = pool_sizes.size(),
        .pPoolSizes = pool_sizes.data(),
    };

    if (vkCreateDescriptorPool(device_, &pool_info, nullptr, &descriptor_pool_) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor pool!");
    }
}

void App::createDescriptorSets() {
    std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptor_set_layout_);

    const VkDescriptorSetAllocateInfo alloc_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptor_pool_,
        .descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
        .pSetLayouts = layouts.data(),
    };

    descriptor_sets_.resize(MAX_FRAMES_IN_FLIGHT);
    if (vkAllocateDescriptorSets(device_, &alloc_info, descriptor_sets_.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        const VkDescriptorBufferInfo buffer_info{
            .buffer = uniform_buffers_[i],
            .offset = 0,
            .range = sizeof(UniformBufferObject),
        };

        const VkDescriptorImageInfo image_info{
            .sampler = texture_sampler_,
            .imageView = texture_image_view_,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };

        const std::array descriptor_writes = {
            VkWriteDescriptorSet{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                 .dstSet = descriptor_sets_[i],
                                 .dstBinding = 0,
                                 .dstArrayElement = 0,
                                 .descriptorCount = 1,
                                 .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                 .pImageInfo = nullptr,
                                 .pBufferInfo = &buffer_info,
                                 .pTexelBufferView = nullptr},
            VkWriteDescriptorSet{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                 .dstSet = descriptor_sets_[i],
                                 .dstBinding = 1,
                                 .dstArrayElement = 0,
                                 .descriptorCount = 1,
                                 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                 .pImageInfo = &image_info,
                                 .pBufferInfo = nullptr,
                                 .pTexelBufferView = nullptr},
        };

        vkUpdateDescriptorSets(device_, descriptor_writes.size(), descriptor_writes.data(), 0, nullptr);
    }
}

void App::createCommandBuffers() {
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        const VkCommandBufferAllocateInfo alloc_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = command_pools_[i],
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };

        if (vkAllocateCommandBuffers(device_, &alloc_info, command_buffers_.data() + i) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }
    }
}

void App::recordCommandBuffer(VkCommandBuffer command_buffer, uint32_t image_index) {
    const VkCommandBufferBeginInfo begin_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr, // Optional
    };

    if (vkBeginCommandBuffer(command_buffer, &begin_info) != VK_SUCCESS) {
        throw std::runtime_error("failed to begin recording command buffer!");
    }

    const VkClearValue clear_color = {{{0.0f, 0.0f, 0.0f, 1.0f}}};

    const VkRenderPassBeginInfo render_pass_begin_info{
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = render_pass_,
        .framebuffer = swapchain_framebuffers_[image_index],
        .renderArea{
                    .offset = {0, 0},
                    .extent = swapchain_extent_,
                    },
        .clearValueCount = 1,
        .pClearValues = &clear_color,
    };

    vkCmdBeginRenderPass(command_buffer, &render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline_);

    const VkViewport viewport{
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(swapchain_extent_.width),
        .height = static_cast<float>(swapchain_extent_.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    vkCmdSetViewport(command_buffer, 0, 1, &viewport);

    const VkRect2D scissor{
        .offset = {0, 0},
        .extent = swapchain_extent_,
    };
    vkCmdSetScissor(command_buffer, 0, 1, &scissor);

    const VkBuffer vertex_buffers[] = {vertex_buffer_};
    const VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(command_buffer, 0, 1, vertex_buffers, offsets);
    vkCmdBindIndexBuffer(command_buffer, index_buffer_, 0, VK_INDEX_TYPE_UINT32);

    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_, 0, 1,
                            &descriptor_sets_[current_frame_], 0, nullptr);
    vkCmdDrawIndexed(command_buffer, INDICES.size(), 1, 0, 0, 0);

    vkCmdEndRenderPass(command_buffer);

    if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to record command buffer!");
    }
}

void App::createSyncObjects() {
    const VkSemaphoreCreateInfo semaphore_create_info{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    };

    const VkFenceCreateInfo fence_create_info{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        if (vkCreateSemaphore(device_, &semaphore_create_info, nullptr, &image_available_semaphores_[i]) !=
                VK_SUCCESS ||
            vkCreateSemaphore(device_, &semaphore_create_info, nullptr, &render_finished_semaphores_[i]) !=
                VK_SUCCESS ||
            vkCreateFence(device_, &fence_create_info, nullptr, &in_flight_fences_[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create synchronization objects for a frame!");
        }
    }
}