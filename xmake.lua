set_project("learn-vulkan")
add_rules("mode.debug", "mode.release")
set_languages("cxx20")

add_requires("glfw", "vulkansdk", "glm", "fmt")

add_defines("GLFW_INCLUDE_VULKAN", "GLM_FORCE_RADIANS", "GLM_FORCE_DEPTH_ZERO_TO_ONE")

-- include subprojects
includes("*/xmake.lua")

