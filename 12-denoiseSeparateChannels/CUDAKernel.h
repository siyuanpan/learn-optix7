#pragma once

#include "optix7.h"

#include "CUDAModule.h"

struct CUDAKernel {
  inline void init(const CUDAModule* module, const char* kernel_name) {
    CUresult result =
        cuModuleGetFunction(&kernel, module->cuda_module, kernel_name);
    if (result == CUDA_ERROR_NOT_FOUND) {
      printf("No Kernel with name '%s' was found in the Module!\n",
             kernel_name);
      exit(1);
    }

    if (result != CUDA_SUCCESS) {
      std::stringstream txt;
      const char* error_name;
      const char* error_string;

      cuGetErrorName(result, &error_name);
      cuGetErrorString(result, &error_string);
      txt << "CUDA call failed with error " << error_name << " ("
          << error_string << ")";
      throw std::runtime_error(txt.str());
    }

    CU_CHECK(FuncSetCacheConfig(kernel, CU_FUNC_CACHE_PREFER_L1));
    CU_CHECK(FuncSetSharedMemConfig(kernel,
                                    CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE));
  }

  // Execute kernel without parameters
  inline void execute() { execute_internal(0); }

  // Exectute kernel with one ore more parameters
  template <typename... T>
  inline void execute(const T&... parameters) {
    int buffer_size = fill_buffer(0, parameters...);
    assert(buffer_size < PARAMETER_BUFFER_SIZE);

    execute_internal(buffer_size);
  }

  inline void set_grid_dim(int x, int y, int z) {
    grid_dim_x = x;
    grid_dim_y = y;
    grid_dim_z = z;
  }

  inline void set_block_dim(int x, int y, int z) {
    block_dim_x = x;
    block_dim_y = y;
    block_dim_z = z;
  }

  inline void occupancy_max_block_size_1d() {
    int grid, block;
    CU_CHECK(
        OccupancyMaxPotentialBlockSize(&grid, &block, kernel, nullptr, 0, 0));

    set_block_dim(block, 1, 1);
  }

  inline void occupancy_max_block_size_2d() {
    int grid, block;
    CU_CHECK(
        OccupancyMaxPotentialBlockSize(&grid, &block, kernel, nullptr, 0, 0));

    // Take sqrt because we want block_x x block_y to be as square as possible
    int block_x = int(sqrt(block));
    // Make sure block_x is a multiple of 32
    block_x += (32 - block_x) & 31;

    if (block_x == 0) block_x = 32;

    int block_y = block / block_x;

    set_block_dim(block_x, block_y, 1);
  }

  inline void set_shared_memory(unsigned bytes) { shared_memory_bytes = bytes; }

  CUfunction kernel;

  int grid_dim_x = 64, grid_dim_y = 1, grid_dim_z = 1;
  int block_dim_x = 64, block_dim_y = 1, block_dim_z = 1;

  unsigned shared_memory_bytes = 0;

  static constexpr int PARAMETER_BUFFER_SIZE = 256;  // In bytes

  mutable unsigned char parameter_buffer[PARAMETER_BUFFER_SIZE];

 private:
  template <typename T>
  inline int fill_buffer(int buffer_offset, const T& parameter) {
    int size = sizeof(T);
    int align = alignof(T);

    int alignment = buffer_offset & (align - 1);
    if (alignment != 0) {
      buffer_offset += align - alignment;
    }

    memcpy(parameter_buffer + buffer_offset, &parameter, size);

    return buffer_offset + size;
  }

  template <typename T, typename... Ts>
  inline int fill_buffer(int buffer_offset, const T& parameter,
                         const Ts&... parameters) {
    int offset = fill_buffer(buffer_offset, parameter);

    return fill_buffer(offset, parameters...);
  }

  inline void execute_internal(size_t parameter_buffer_size) const {
    void* params[] = {
        CU_LAUNCH_PARAM_BUFFER_POINTER,
        parameter_buffer,
        CU_LAUNCH_PARAM_BUFFER_SIZE,
        &parameter_buffer_size,
        CU_LAUNCH_PARAM_END,
    };

    CU_CHECK(LaunchKernel(kernel, grid_dim_x, grid_dim_y, grid_dim_z,
                          block_dim_x, block_dim_y, block_dim_z,
                          shared_memory_bytes, nullptr, nullptr, params));
  }
};