#include <cstdint>
#include <climits>
#include <iostream>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <backends/imgui_impl_vulkan.h>

#include "graphics_internal.hpp"
#include "application.hpp"

namespace {

constexpr int32_t default_window_width = 1280;
constexpr int32_t default_window_height = 720;

constexpr char default_window_title[] = "1 lab po CG";

GLFWwindow* glfw_window = nullptr;

} // namespace

int main() {
    int status = EXIT_SUCCESS;

    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return EXIT_FAILURE;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    glfw_window = glfwCreateWindow(default_window_width, default_window_height,
                                   default_window_title, nullptr, nullptr);

    uint32_t extensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
    std::cout << extensionCount << "extension supported\n";
    
    if (glfw_window == nullptr) {
        status = EXIT_FAILURE;
        goto err_null_window;
    }

    glfwSetFramebufferSizeCallback(glfw_window, [](GLFWwindow*, int width, int height){
        if (width == 0 || height == 0) {
            return;
        }

        graphics::internal::resize(width, height);
    });

    if (ImGui::CreateContext() == nullptr) {
        std::cerr << "Failed to create ImGUI context\n";
        status = EXIT_FAILURE;
        goto err_imgui_init;
    }

    if (!ImGui_ImplGlfw_InitForVulkan(glfw_window, true)) {
        std::cerr << "Failed to initialize ImGUI GLFW backend for Vulkan renderer\n";
        status = EXIT_FAILURE;
        goto err_imgui_glfw_init;
    }

    if (!graphics::internal::initialize(glfw_window)) {
        std::cerr << "Failed to initialize graphics\n";
        status = EXIT_FAILURE;
        goto err_graphics_init;
    }

    if (!application::initialize()) {
        std::cerr << "Failed to initialize application\n";
        status = EXIT_FAILURE;
        goto err_application_init;
    }


    while (!glfwWindowShouldClose(glfw_window)) {
        int width = 0, height = 0;
        glfwGetFramebufferSize(glfw_window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(glfw_window, &width, &height);
            glfwWaitEvents();
        }

        const double time = glfwGetTime();

        glfwPollEvents();
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();

        ImGui::NewFrame();

        // Wait for the previous frame to finish before the application updates
        // host-visible uniform/storage buffers. prepare() waits on the in-flight
        // fence and begins the render pass.
        graphics::internal::FrameData fd = graphics::internal::prepare();
        application::update(time);
        ImGui::Render();
        application::render(fd);
        graphics::internal::submitAndPresent();
    }

    application::shutdown();
err_application_init:
    graphics::internal::shutdown();
err_graphics_init:
    ImGui_ImplGlfw_Shutdown();
err_imgui_glfw_init:
    ImGui::DestroyContext();
err_imgui_init:
    glfwDestroyWindow(glfw_window);
err_null_window:
    glfwTerminate();

    return status;
}