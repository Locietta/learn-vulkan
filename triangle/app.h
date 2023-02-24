#pragma once

#include <GLFW/glfw3.h>
#include <array>
#include <span>
#include <vector>

class App {
public:
    App();
    ~App();
    void run();

    constexpr static uint32_t WIDTH = 800;
    constexpr static uint32_t HEIGHT = 600;

    constexpr static std::array VALIDATION_LAYERS{"VK_LAYER_KHRONOS_validation"};
    constexpr static std::array DEVICE_EXTENSIONS{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    constexpr static std::array DYNAMIC_STATES{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};

#ifdef NDEBUG
    constexpr static bool ENABLE_VALIDATION_LAYERS = false;
#else
    constexpr static bool ENABLE_VALIDATION_LAYERS = true;
#endif

private:
    void drawFrame();

    void initWindow();
    void initVulkan();

    void createGraphicsPipeline();
    void createRenderPass();
    void createFramebuffers();
    void createCommandPool();
    void createCommandBuffer();
    void createSyncObjects();

    void recordCommandBuffer(VkCommandBuffer command_buffer, uint32_t image_index);

    GLFWwindow *window_ = nullptr;
    VkInstance instance_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;

    VkQueue graphics_queue_;
    VkQueue present_queue_;

    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat swapchain_image_format_;
    VkExtent2D swapchain_extent_;

    std::vector<VkImage> swapchain_images_;
    std::vector<VkImageView> swapchain_image_views_;

    VkPipelineLayout pipeline_layout_;
    VkRenderPass render_pass_;
    VkPipeline graphics_pipeline_;

    std::vector<VkFramebuffer> swapchain_framebuffers_;
    VkCommandPool command_pool_;
    VkCommandBuffer command_buffer_;

    VkSemaphore image_available_semaphore_;
    VkSemaphore render_finished_semaphore_;
    VkFence in_flight_fence_;
};
