cmake_minimum_required(VERSION 3.21)

# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -allow-unsupported-compiler" )
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
# set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86)
# set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5/bin/nvcc.exe")

project(learn-optix7 LANGUAGES CXX )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

foreach (lang CXX CUDA)
  set(CMAKE_${lang}_STANDARD 17)
  set(CMAKE_${lang}_STANDARD_REQUIRED ON)
  set(CMAKE_${lang}_EXTENSIONS OFF)
endforeach()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)
# enable_language(CUDA)
find_package(OptiX REQUIRED)

configure_file(
	${CMAKE_CURRENT_SOURCE_DIR}/defines.h.in
	${CMAKE_CURRENT_SOURCE_DIR}/defines.h
)

find_package(OpenGL REQUIRED)

add_subdirectory(dep)

add_subdirectory(01-helloOptix)
add_subdirectory(02-pipelineAndRayGen)
add_subdirectory(03-inGLFWindow)
add_subdirectory(04-firstTriangleMesh)
add_subdirectory(05-firstSBTData)
add_subdirectory(06-multipleObjects)
add_subdirectory(07-firstRealModel)
add_subdirectory(08-addingTextures)
add_subdirectory(09-shadowRays)
add_subdirectory(10-softShadows)
add_subdirectory(11-denoiseColorOnly)
add_subdirectory(12-denoiseSeparateChannels)