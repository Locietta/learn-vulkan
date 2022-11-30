
target("check")
    set_kind("binary")
    add_files("main.cpp")
    add_packages("glfw", "vulkansdk", "glm", "fmt")