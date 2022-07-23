#include <iostream>

#include "optix7.h"

void InitOptix() {
  cudaFree(0);
  int num_devices;
  cudaGetDeviceCount(&num_devices);
  if (num_devices == 0)
    throw std::runtime_error("no CUDA capable devices found!");

  std::cout << "found " << num_devices << " CUDA devices" << std::endl;

  // initialize optix
  OPTIX_CHECK(optixInit());
}

int main(int ac, char** av) {
  try {
    std::cout << "initializing optix..." << std::endl;

    InitOptix();

    // std::cout << GDT_TERMINAL_GREEN
    //           << "#osc: successfully initialized optix... yay!"
    //           << GDT_TERMINAL_DEFAULT << std::endl;

    // for this simple hello-world example, don't do anything else
    // ...
    std::cout << "done. clean exit." << std::endl;

  } catch (std::runtime_error& e) {
    std::cout << "FATAL ERROR: " << e.what() << std::endl;
    exit(1);
  }
  return 0;
}