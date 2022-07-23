
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/device_programs.ptx
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/device_programs.cu
    COMMAND nvcc ${CMAKE_CURRENT_SOURCE_DIR}/device_programs.cu -allow-unsupported-compiler
        -ptx -o ${CMAKE_CURRENT_SOURCE_DIR}/device_programs.ptx -std=c++17
        -I ${OptiX_ROOT_DIR}/include 
        -I ${PROJECT_SOURCE_DIR}/dep
)

add_executable(03_inGLFWindow main.cpp CUDABuffer.h LaunchParams.h SampleRenderer.h SampleRenderer.cpp GLFWindow.h GLFWindow.cpp ${CMAKE_CURRENT_SOURCE_DIR}/device_programs.ptx)

target_link_libraries(03_inGLFWindow PUBLIC opengl32 glfw CUDA::cudart CUDA::cuda_driver OptiX::OptiX)
target_include_directories(03_inGLFWindow PUBLIC ${CMAKE_SOURCE_DIR}/dep)