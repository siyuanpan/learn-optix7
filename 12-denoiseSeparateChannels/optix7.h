#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <sstream>

#ifndef PRINT
#  define PRINT(var) std::cout << #  var << "=" << var << std::endl;
#  define PING                                                        \
    std::cout << __FILE__ << "::" << __LINE__ << ": " << __FUNCTION__ \
              << std::endl;
#endif

#define WARP_SIZE 32
#define MAX_REGISTERS 64

template <typename T, int N>
constexpr int array_count(const T (&array)[N]) {
  return N;
}

#define CU_CHECK(call)                                            \
  {                                                               \
    CUresult rc = cu##call;                                       \
    if (rc != CUDA_SUCCESS) {                                     \
      std::stringstream txt;                                      \
      const char* error_name;                                     \
      const char* error_string;                                   \
                                                                  \
      cuGetErrorName(rc, &error_name);                            \
      cuGetErrorString(rc, &error_string);                        \
      txt << "CUDA call failed with error " << error_name << " (" \
          << error_string << ")";                                 \
      throw std::runtime_error(txt.str());                        \
    }                                                             \
  }

#define CUDA_CHECK(call)                                    \
  {                                                         \
    cudaError_t rc = cuda##call;                            \
    if (rc != cudaSuccess) {                                \
      std::stringstream txt;                                \
      cudaError_t err = rc; /*cudaGetLastError();*/         \
      txt << "CUDA Error " << cudaGetErrorName(err) << " (" \
          << cudaGetErrorString(err) << ")";                \
      throw std::runtime_error(txt.str());                  \
    }                                                       \
  }

#define OPTIX_CHECK(call)                                                \
  {                                                                      \
    OptixResult res = call;                                              \
    if (res != OPTIX_SUCCESS) {                                          \
      fprintf(stderr, "Optix call (%s) failed with code %d (line %d)\n", \
              #call, res, __LINE__);                                     \
      exit(2);                                                           \
    }                                                                    \
  }

#define CUDA_SYNC_CHECK()                                              \
  {                                                                    \
    cudaDeviceSynchronize();                                           \
    cudaError_t error = cudaGetLastError();                            \
    if (error != cudaSuccess) {                                        \
      fprintf(stderr, "error (%s: line %d): %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(error));                              \
      exit(2);                                                         \
    }                                                                  \
  }

#define NVRTC_CHECK(call)                                                    \
  {                                                                          \
    nvrtcResult result = call;                                               \
    if (result != NVRTC_SUCCESS) {                                           \
      fprintf(stderr, "%s:%d: NVRTC call failed with error %s!\n", __FILE__, \
              __LINE__, nvrtcGetErrorString(result));                        \
      exit(2);                                                               \
    }                                                                        \
  }