add_rules("hlsl2spv")

target("triangle")
    set_kind("binary")
    add_files("*.cpp", "shader/*.vert", "shader/*.frag")
    add_packages("glfw", "vulkansdk", "glm", "fmt")
