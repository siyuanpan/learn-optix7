#pragma once

#include "CUDABuffer.h"
#include "LaunchParams.h"

#include "GLFWindow.h"
#include "Model.h"

struct Camera {
  /*! camera position - *from* where we are looking */
  vec3f from;
  /*! which point we are looking *at* */
  vec3f at;
  /*! general up-vector */
  vec3f up;
};

class SampleRenderer {
 public:
  /*! constructor - performs all setup, including initializing
    optix, creates module, pipeline, programs, SBT, etc. */
  SampleRenderer(const Model *model, const QuadLight &light);

  /*! render one frame */
  void render();

  /*! resize frame buffer to given resolution */
  void resize(const glm::ivec2 &newSize);

  /*! download the rendered color buffer */
  void downloadPixels(vec4f h_pixels[]);

  /*! set camera to render with */
  void setCamera(const Camera &camera);

  bool denoiserOn = true;
  bool accumulate = true;

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

  /*! build an acceleration structure for the given triangle mesh */
  OptixTraversableHandle buildAccel();

  /*! upload textures, and create cuda texture objects for them */
  void createTextures();

 protected:
  /*! @{ CUDA device context and stream that optix pipeline will run
      on, as well as device properties for this device */
  CUcontext cudaContext;
  CUstream stream;
  cudaDeviceProp deviceProps;

  OptixDeviceContext optixContext;

  int compute_capability;

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

 public:
  LaunchParams launchParams;
  CUDABuffer launchParamsBuffer;
  /*! @} */

  /*! the color buffer we use during _rendering_, which is a bit
      larger than the actual displayed frame buffer (to account for
      the border), and in float4 format (the denoiser requires
      floats) */
  CUDABuffer renderBuffer;

  /*! the actual final color buffer used for display, in rgba8 */
  CUDABuffer denoisedBuffer;

  OptixDenoiser denoiser = nullptr;
  CUDABuffer denoiserScratch;
  CUDABuffer denoiserState;

  /*! the camera we are to render with. */
  Camera lastSetCamera;

  /*! the model we are going to trace rays against */
  const Model *model;

  /*! one buffer per input mesh */
  std::vector<CUDABuffer> vertexBuffer;
  std::vector<CUDABuffer> normalBuffer;
  std::vector<CUDABuffer> texcoordBuffer;
  std::vector<CUDABuffer> indexBuffer;
  //! buffer that keeps the (final, compacted) accel structure
  CUDABuffer asBuffer;

  /*! @{ one texture object and pixel array per used texture */
  std::vector<cudaArray_t> textureArrays;
  std::vector<cudaTextureObject_t> textureObjects;
  /*! @} */
};