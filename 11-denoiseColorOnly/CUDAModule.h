#pragma once

#include "optix7.h"
#include <vector>
#include <assert.h>
#include <string.h>

struct CUDAModule {
  void init(const std::string& module_name, const std::string& filename,
            int compute_capability, int max_registers) {}
  //   CUdevicePtr ptr;
};