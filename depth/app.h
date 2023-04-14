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
    constexpr static std::array REQUIRED_EXTENSIONS{
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME,
        VK_EXT_PAGEABLE_DEVICE_LOCAL_MEMORY_EXTENSION_NAME,
    };

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

    void createInstance();
    void createPhysicalDevice();
    void createLogicalDevice();

    void createSwapchain();
    void recreateSwapchain();
    void cleanupSwapchain();
    void createImageViews();

    void createRenderPass();
    void createDescriptorSetLayout();
    void createGraphicsPipeline();
    void createFramebuffers();
    void createCommandPool();

    void createDepthResources();

    void createTextureImage();
    void createTextureImageView();
    void createTextureSampler();
    void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage,
                     VkMemoryPropertyFlags properties, VkImage &image, VkDeviceMemory &image_memory);
    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout old_layout, VkImageLayout new_layout);
    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);

    void createVertexBuffer();
    void createIndexBuffer();
    void createUniformBuffers();
    void updateUniformBuffer(uint32_t current_image);
    void createDescriptorPool();
    void createDescriptorSets();

    void createCommandBuffers();
    void createSyncObjects();

    void recordCommandBuffer(VkCommandBuffer command_buffer, uint32_t image_index);
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer &buffer,
                      VkDeviceMemory &buffer_memory);

    [[nodiscard]] VkCommandBuffer beginSingleTimeCommands() const;
    void copyBuffer(VkBuffer src_buffer, VkBuffer dst_buffer, VkDeviceSize size);
    void endSingleTimeCommands(VkCommandBuffer command_buffer) const;
    VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspect_flags);

    [[nodiscard]] VkFormat findSupportedFormat(std::initializer_list<const VkFormat> candidates, VkImageTiling tiling,
                                               VkFormatFeatureFlags features) const;

    [[nodiscard]] VkFormat findDepthFormat() const;

    void displayFPS();

    static void framebufferResizeCallback(GLFWwindow *window, int width, int height);

    constexpr static int MAX_FRAMES_IN_FLIGHT = 2;
    constexpr static int NUM_COMMAND_POOL = MAX_FRAMES_IN_FLIGHT + 1; // extra one for transfer

    uint32_t current_frame_ = 0;
    bool framebuffer_resized_ = false;
    double last_time_ = 0.0;

    GLFWwindow *window_ = nullptr;
    VkInstance instance_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;

    VkQueue graphics_queue_;
    VkQueue present_queue_;
    VkQueue transfer_queue_;

    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat swapchain_image_format_;
    VkExtent2D swapchain_extent_;

    std::vector<VkImage> swapchain_images_;
    std::vector<VkImageView> swapchain_image_views_;

    VkPipelineLayout pipeline_layout_;
    VkRenderPass render_pass_;
    VkDescriptorSetLayout descriptor_set_layout_;
    VkPipeline graphics_pipeline_;

    std::vector<VkFramebuffer> swapchain_framebuffers_;
    std::array<VkCommandPool, NUM_COMMAND_POOL> command_pools_;
    std::array<VkCommandBuffer, MAX_FRAMES_IN_FLIGHT> command_buffers_;

    std::array<VkSemaphore, MAX_FRAMES_IN_FLIGHT> image_available_semaphores_;
    std::array<VkSemaphore, MAX_FRAMES_IN_FLIGHT> render_finished_semaphores_;
    std::array<VkFence, MAX_FRAMES_IN_FLIGHT> in_flight_fences_;

    // TODO: combine vertex/index buffers & memorys into one
    VkBuffer vertex_buffer_;
    VkDeviceMemory vertex_buffer_memory_;
    VkBuffer index_buffer_;
    VkDeviceMemory index_buffer_memory_;

    std::vector<VkBuffer> uniform_buffers_;
    std::vector<VkDeviceMemory> uniform_buffers_memory_;
    std::vector<void *> uniform_buffers_mapped_;

    VkDescriptorPool descriptor_pool_;
    std::vector<VkDescriptorSet> descriptor_sets_;

    VkImage texture_image_;
    VkDeviceMemory texture_image_memory_;
    VkImageView texture_image_view_;
    VkSampler texture_sampler_;

    VkImage depth_image_;
    VkDeviceMemory depth_image_memory_;
    VkImageView depth_image_view_;
};
