#include <GLFW/glfw3.h>

#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>

#include <fmt/core.h>

constexpr int window_width = 800, window_height = 600;

int main() {
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow *window = glfwCreateWindow(window_width, window_height, "Vulkan window", nullptr, nullptr);

    uint32_t extension_count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr);

    fmt::print("{} extensions supported\n", extension_count);

    glm::mat4 matrix;
    glm::vec4 vec;
    auto test = matrix * vec;

    while (glfwWindowShouldClose(window) == 0) {
        glfwPollEvents();
    }

    glfwDestroyWindow(window);

    glfwTerminate();

    return 0;
}