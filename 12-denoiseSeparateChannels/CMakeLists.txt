
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/device_programs.ptx
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/device_programs.cu
    COMMAND nvcc ${CMAKE_CURRENT_SOURCE_DIR}/device_programs.cu -allow-unsupported-compiler
        -ptx -o ${CMAKE_CURRENT_SOURCE_DIR}/device_programs.ptx -std=c++17
        -I ${OptiX_ROOT_DIR}/include 
        -I ${PROJECT_SOURCE_DIR}/dep
)

# add_custom_command(
#     OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/toneMap.obj
#     DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/toneMap.cu
#     COMMAND nvcc ${CMAKE_CURRENT_SOURCE_DIR}/toneMap.cu -allow-unsupported-compiler
#         -c -o ${CMAKE_CURRENT_SOURCE_DIR}/toneMap.obj -std=c++17
#         -I ${OptiX_ROOT_DIR}/include 
#         -I ${PROJECT_SOURCE_DIR}/dep
#         -I ${GLFW_SOURCE_DIR}/include
# )

# CUDA_COMPILE(generated_files toneMap.cu DEBUG -allow-unsupported-compiler)
# add_library(toneMap ${generated_files})
# set(CMAKE_CUDA_FLAGS_DEBUG "-allow-unsupported-compiler")
# cuda_add_library(toneMap STATIC toneMap.cu OPTIONS -allow-unsupported-compiler)
# target_compile_options(toneMap PRIVATE  "-allow-unsupported-compiler")

# set(COMPILE_OPTIONS)
# list(APPEND COMPILE_OPTIONS -allow-unsupported-compiler)
# add_library(toneMap toneMap.cu)
# target_include_directories(toneMap PUBLIC ${CMAKE_SOURCE_DIR}/dep)
# target_compile_options(toneMap PUBLIC cuda_std_17 cxx_std_17)
# set_target_properties(toneMap PROPERTIES CUDA_ARCHITECTURES "35;50;52;72;75" CUDA_SEPARABLE_COMPILATION ON)
# target_compile_options(toneMap PRIVATE ${COMPILE_OPTIONS})
# target_link_libraries(toneMap CUDA::cudart CUDA::cuda_driver)
# set_target_properties(toneMap PROPERTIES CUDA_ARCHITECTURES "35;50;72")

add_executable(12_denoiseSeparateChannels main.cpp CUDABuffer.h CUDAModule.h CUDAKernel.h LaunchParams.h SampleRenderer.h SampleRenderer.cpp GLFWindow.h GLFWindow.cpp Model.h Model.cpp ${CMAKE_CURRENT_SOURCE_DIR}/device_programs.ptx)

target_link_libraries(12_denoiseSeparateChannels PUBLIC  opengl32 glfw CUDA::cudart CUDA::cuda_driver CUDA::nvrtc OptiX::OptiX)
target_include_directories(12_denoiseSeparateChannels PUBLIC ${CMAKE_SOURCE_DIR}/dep)