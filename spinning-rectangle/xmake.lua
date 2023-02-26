
target("spinning-rectangle")
    set_kind("binary")
    add_rules("hlsl2spv", { bin2c = true })
    add_files("*.cpp", "shader/*.hlsl")
    add_packages("glfw", "vulkansdk", "glm", "fmt")
