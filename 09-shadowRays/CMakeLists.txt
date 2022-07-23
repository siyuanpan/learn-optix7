
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/device_programs.ptx
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/device_programs.cu
    COMMAND nvcc ${CMAKE_CURRENT_SOURCE_DIR}/device_programs.cu -allow-unsupported-compiler
        -ptx -o ${CMAKE_CURRENT_SOURCE_DIR}/device_programs.ptx -std=c++17
        -I ${OptiX_ROOT_DIR}/include 
        -I ${PROJECT_SOURCE_DIR}/dep
)

add_executable(09_shadowRays main.cpp CUDABuffer.h LaunchParams.h SampleRenderer.h SampleRenderer.cpp GLFWindow.h GLFWindow.cpp Model.h Model.cpp ${CMAKE_CURRENT_SOURCE_DIR}/device_programs.ptx)

target_link_libraries(09_shadowRays PUBLIC opengl32 glfw CUDA::cudart CUDA::cuda_driver OptiX::OptiX)
target_include_directories(09_shadowRays PUBLIC ${CMAKE_SOURCE_DIR}/dep)