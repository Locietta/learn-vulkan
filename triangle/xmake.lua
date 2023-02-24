target("triangle")
    set_kind("binary")
    add_files("*.cpp")
    add_packages("glfw", "vulkansdk", "glm", "fmt")
