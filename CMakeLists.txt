cmake_minimum_required(VERSION 3.20)
include(FetchContent)

project(vulkan-starter-app LANGUAGES C CXX)

add_executable(${PROJECT_NAME}
    source/main.cpp
    source/graphics_internal.cpp
    source/application.cpp
)

if(UNIX AND NOT APPLE)
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
elseif(APPLE)
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
elseif(WIN32)
    if(MSVC)
        target_compile_options(${PROJECT_NAME} PRIVATE /W4 /wd4201)
        target_link_options(${PROJECT_NAME} PRIVATE /subsystem:windows /entry:mainCRTStartup)
    else()
        target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
        target_link_options(${PROJECT_NAME} PRIVATE -mwindows)
    endif()
    target_compile_definitions(${PROJECT_NAME} PRIVATE -D_USE_MATH_DEFINES)
else()
    message(FATAL_ERROR "${CMAKE_SYSTEM_NAME} is not supported")
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED TRUE
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.5.1
)
FetchContent_Declare(
    vk-bootstrap
    GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap.git
    GIT_TAG main
)
FetchContent_Declare(
    VulkanMemoryAllocator
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
    GIT_TAG v3.4.0
)
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.92.9
)
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 1.0.1
)
# stb_image is a single-header image decoder; it is fetched as source only.
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG master
)

find_package(Vulkan REQUIRED)

set(GLFW_LIBRARY_TYPE STATIC)
set(GLFW_BUILD_EXAMPLES OFF)
set(GLFW_BUILD_TESTS OFF)
set(GLFW_BUILD_DOCS OFF)

FetchContent_MakeAvailable(
    glfw
    vk-bootstrap
    VulkanMemoryAllocator
    imgui
    glm
)
FetchContent_GetProperties(stb)
if(NOT stb_POPULATED)
    FetchContent_Populate(stb)
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE
    glfw
    Vulkan::Vulkan
    vk-bootstrap::vk-bootstrap
    GPUOpen::VulkanMemoryAllocator
    glm::glm
)
target_include_directories(${PROJECT_NAME} PRIVATE
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
    ${stb_SOURCE_DIR}
)
target_sources(${PROJECT_NAME} PRIVATE
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
)

# Keep the image files next to the executable / in the debugger working directory.
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/textures" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

# Compile GLSL to SPIR-V as part of the normal build.
find_program(SHADER_COMPILER
    NAMES glslc glslangValidator
    HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin"
)
if(NOT SHADER_COMPILER)
    message(FATAL_ERROR
        "Could not find glslc or glslangValidator. Install the Vulkan SDK and reconfigure CMake.")
endif()

get_filename_component(SHADER_COMPILER_NAME "${SHADER_COMPILER}" NAME)
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/shaders")

function(compile_glsl SOURCE_FILE OUTPUT_FILE STAGE)
    if(SHADER_COMPILER_NAME STREQUAL "glslc" OR SHADER_COMPILER_NAME STREQUAL "glslc.exe")
        add_custom_command(
            OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/shaders/${OUTPUT_FILE}"
            COMMAND "${SHADER_COMPILER}" "-fshader-stage=${STAGE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SOURCE_FILE}"
                    -o "${CMAKE_CURRENT_BINARY_DIR}/shaders/${OUTPUT_FILE}"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SOURCE_FILE}"
            VERBATIM
        )
    else()
        add_custom_command(
            OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/shaders/${OUTPUT_FILE}"
            COMMAND "${SHADER_COMPILER}" -V -S "${STAGE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SOURCE_FILE}"
                    -o "${CMAKE_CURRENT_BINARY_DIR}/shaders/${OUTPUT_FILE}"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SOURCE_FILE}"
            VERBATIM
        )
    endif()
endfunction()

compile_glsl(shader.vert vert.spv vert)
compile_glsl(shader.frag frag.spv frag)
add_custom_target(compile_shaders DEPENDS
    "${CMAKE_CURRENT_BINARY_DIR}/shaders/vert.spv"
    "${CMAKE_CURRENT_BINARY_DIR}/shaders/frag.spv"
)
add_dependencies(${PROJECT_NAME} compile_shaders)
