#pragma once

#include "optix7.h"
#include <vector>
#include <assert.h>
#include <string.h>
#include <filesystem>
#include <nvrtc.h>
#include <fstream>

struct CUDAModule {
  void init(const std::string &module_name, const std::string &filename,
            int compute_capability, int max_registers) {
#ifdef _DEBUG
    std::string ptx_filename = filename + ".debug.ptx";
#else
    std::string ptx_filename = filename + ".release.ptx";
#endif

    bool should_recompile = true;
    if (std::filesystem::exists(ptx_filename)) {
      std::filesystem::file_time_type last_write_time_source =
          std::filesystem::last_write_time(filename);
      std::filesystem::file_time_type last_write_time_ptx =
          std::filesystem::last_write_time(ptx_filename);

      // Recompile if the source file is newer than the binary
      should_recompile = last_write_time_ptx < last_write_time_source;
    }

    if (should_recompile) {
      std::ifstream src_fin(filename);
      const std::string src_str((std::istreambuf_iterator<char>(src_fin)),
                                std::istreambuf_iterator<char>());

      nvrtcProgram program;
      NVRTC_CHECK(nvrtcCreateProgram(&program, src_str.c_str(),
                                     filename.c_str(), 0, NULL, NULL));

      // Configure options
      std::string option_compute =
          "--gpu-architecture=compute_" + std::to_string(compute_capability);
      std::string option_maxregs =
          "--maxrregcount=" + std::to_string(max_registers);

      const char *options[] = {
          "--std=c++17", option_compute.c_str(), option_maxregs.c_str(),
          "--use_fast_math", "--extra-device-vectorization",
          //"--device-debug",
          "-lineinfo", "-restrict",
          // "-I C:/Program Files/NVIDIA GPU Computing
          // Toolkit/CUDA/v11.5/include",
          // "-I C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.4.0/include",
          // "-I C:/Users/pansiyuan/dev/learn-optix/dep",
      };

      // Compile to PTX
      nvrtcResult result =
          nvrtcCompileProgram(program, array_count(options), options);

      size_t log_size;
      NVRTC_CHECK(nvrtcGetProgramLogSize(program, &log_size));

      if (log_size > 1) {
        std::string log;
        log.resize(log_size);
        NVRTC_CHECK(nvrtcGetProgramLog(program, log.data()));

        std::cout << "NVRTC output:\n" << log << "\n";
      }

      if (result != NVRTC_SUCCESS) __debugbreak();  // Compile error

      // Obtain PTX from NVRTC
      size_t ptx_size;
      NVRTC_CHECK(nvrtcGetPTXSize(program, &ptx_size));
      std::string ptx;
      ptx.resize(ptx_size);
      NVRTC_CHECK(nvrtcGetPTX(program, ptx.data()));
      std::cout << "ptx size : " << ptx_size << std::endl;

      NVRTC_CHECK(nvrtcDestroyProgram(&program));

      std::ofstream fs(ptx_filename);
      fs << ptx;
      fs.close();
    }

    char log_buffer[8192];
    log_buffer[0] = NULL;

    CUjit_option options[] = {
        CU_JIT_MAX_REGISTERS,
        CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_INFO_LOG_BUFFER,
        CU_JIT_LOG_VERBOSE,
    };

    void *values[] = {
        reinterpret_cast<void *>(max_registers),
        reinterpret_cast<void *>(sizeof(log_buffer)),
        reinterpret_cast<void *>(log_buffer),
        reinterpret_cast<void *>(
            should_recompile)  // Only verbose if we just recompiled
    };

    CUlinkState link_state;
    CU_CHECK(LinkCreate(array_count(options), options, values, &link_state));
    CU_CHECK(LinkAddFile(link_state, CU_JIT_INPUT_PTX, ptx_filename.c_str(), 0,
                         nullptr, nullptr));

    void *cubin;
    size_t cubin_size;
    cuLinkComplete(link_state, &cubin, &cubin_size);

    CU_CHECK(ModuleLoadData(&cuda_module, cubin));

    CU_CHECK(LinkDestroy(link_state));

    if (should_recompile) {
      printf("%s\n", log_buffer);
    }
  }

  void free() { CU_CHECK(ModuleUnload(cuda_module)); }

  CUmodule cuda_module;
};