#pragma once

#include "CUDABuffer.h"
#include "LaunchParams.h"

class SampleRenderer {
 public:
  /*! constructor - performs all setup, including initializing
    optix, creates module, pipeline, programs, SBT, etc. */
  SampleRenderer();

  /*! render one frame */
  void render();

  /*! resize frame buffer to given resolution */
  void resize(const glm::ivec2 &newSize);

  /*! download the rendered color buffer */
  void downloadPixels(uint32_t h_pixels[]);

 protected:
  // ------------------------------------------------------------------
  // internal helper functions
  // ------------------------------------------------------------------

  /*! helper function that initializes optix and checks for errors */
  void initOptix();

  /*! creates and configures a optix device context (in this simple
    example, only for the primary GPU device) */
  void createContext();

  /*! creates the module that contains all the programs we are going
    to use. in this simple example, we use a single module from a
    single .cu file, using a single embedded ptx string */
  void createModule();

  /*! does all setup for the raygen program(s) we are going to use */
  void createRaygenPrograms();

  /*! does all setup for the miss program(s) we are going to use */
  void createMissPrograms();

  /*! does all setup for the hitgroup program(s) we are going to use */
  void createHitgroupPrograms();

  /*! assembles the full pipeline of all programs */
  void createPipeline();

  /*! constructs the shader binding table */
  void buildSBT();

 protected:
  /*! @{ CUDA device context and stream that optix pipeline will run
      on, as well as device properties for this device */
  CUcontext cudaContext;
  CUstream stream;
  cudaDeviceProp deviceProps;

  OptixDeviceContext optixContext;

  /*! @{ the pipeline we're building */
  OptixPipeline pipeline;
  OptixPipelineCompileOptions pipelineCompileOptions = {};
  OptixPipelineLinkOptions pipelineLinkOptions = {};
  /*! @} */

  /*! @{ the module that contains out device programs */
  OptixModule module;
  OptixModuleCompileOptions moduleCompileOptions = {};
  /* @} */

  std::vector<OptixProgramGroup> raygenPGs;
  CUDABuffer raygenRecordsBuffer;
  std::vector<OptixProgramGroup> missPGs;
  CUDABuffer missRecordsBuffer;
  std::vector<OptixProgramGroup> hitgroupPGs;
  CUDABuffer hitgroupRecordsBuffer;
  OptixShaderBindingTable sbt = {};

  LaunchParams launchParams;
  CUDABuffer launchParamsBuffer;
  /*! @} */

  CUDABuffer colorBuffer;
};