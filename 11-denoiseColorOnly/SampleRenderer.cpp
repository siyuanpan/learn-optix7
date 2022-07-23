#include "SampleRenderer.h"
#include <optix_function_table_definition.h>
#include <iostream>
#include "../defines.h"
#include <fstream>
#include <filesystem>
#include <nvrtc.h>
#include "CUDAModule.h"

/*! SBT record for a raygen program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord {
  __align__(
      OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  // just a dummy value - later examples will use more interesting
  // data here
  void *data;
};

/*! SBT record for a miss program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord {
  __align__(
      OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  // just a dummy value - later examples will use more interesting
  // data here
  void *data;
};

/*! SBT record for a hitgroup program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord {
  __align__(
      OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  TriangleMeshSBTData data;
};

SampleRenderer::SampleRenderer(const Model *model, const QuadLight &light)
    : model(model) {
  initOptix();

  launchParams.light.origin = light.origin;
  launchParams.light.du = light.du;
  launchParams.light.dv = light.dv;
  launchParams.light.power = light.power;

  std::cout << "creating optix context ..." << std::endl;
  createContext();

  std::cout << "setting up module ..." << std::endl;
  createModule();

  std::cout << "creating raygen programs ..." << std::endl;
  createRaygenPrograms();
  std::cout << "creating miss programs ..." << std::endl;
  createMissPrograms();
  std::cout << "creating hitgroup programs ..." << std::endl;
  createHitgroupPrograms();

  launchParams.traversable = buildAccel();

  std::cout << "setting up optix pipeline ..." << std::endl;
  createPipeline();

  createTextures();

  std::cout << "#building SBT ..." << std::endl;
  buildSBT();

  launchParamsBuffer.alloc(sizeof(launchParams));
  std::cout << "context, module, pipeline, etc, all set up ..." << std::endl;

  std::cout << "Optix 7 Sample fully set up" << std::endl;
}

void SampleRenderer::createTextures() {
  int numTextures = (int)model->textures.size();

  textureArrays.resize(numTextures);
  textureObjects.resize(numTextures);

  for (int textureID = 0; textureID < numTextures; textureID++) {
    auto texture = model->textures[textureID];

    cudaResourceDesc res_desc = {};

    cudaChannelFormatDesc channel_desc;
    int32_t width = texture->resolution.x;
    int32_t height = texture->resolution.y;
    int32_t numComponents = 4;
    int32_t pitch = width * numComponents * sizeof(uint8_t);
    channel_desc = cudaCreateChannelDesc<uchar4>();

    cudaArray_t &pixelArray = textureArrays[textureID];
    CUDA_CHECK(MallocArray(&pixelArray, &channel_desc, width, height));

    CUDA_CHECK(Memcpy2DToArray(pixelArray,
                               /* offset */ 0, 0, texture->pixel, pitch, pitch,
                               height, cudaMemcpyHostToDevice));

    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = pixelArray;

    cudaTextureDesc tex_desc = {};
    tex_desc.addressMode[0] = cudaAddressModeWrap;
    tex_desc.addressMode[1] = cudaAddressModeWrap;
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.readMode = cudaReadModeNormalizedFloat;
    tex_desc.normalizedCoords = 1;
    tex_desc.maxAnisotropy = 1;
    tex_desc.maxMipmapLevelClamp = 99;
    tex_desc.minMipmapLevelClamp = 0;
    tex_desc.mipmapFilterMode = cudaFilterModePoint;
    tex_desc.borderColor[0] = 1.0f;
    tex_desc.sRGB = 0;

    // Create texture object
    cudaTextureObject_t cuda_tex = 0;
    CUDA_CHECK(CreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr));
    textureObjects[textureID] = cuda_tex;
  }
}

OptixTraversableHandle SampleRenderer::buildAccel() {
  const int numMeshes = (int)model->meshes.size();
  vertexBuffer.resize(numMeshes);
  normalBuffer.resize(numMeshes);
  texcoordBuffer.resize(numMeshes);
  indexBuffer.resize(numMeshes);

  OptixTraversableHandle asHandle{0};

  // ==================================================================
  // triangle inputs
  // ==================================================================
  std::vector<OptixBuildInput> triangleInput(model->meshes.size());
  std::vector<CUdeviceptr> d_vertices(model->meshes.size());
  std::vector<CUdeviceptr> d_indices(model->meshes.size());
  std::vector<uint32_t> triangleInputFlags(model->meshes.size());

  for (int meshID = 0; meshID < model->meshes.size(); meshID++) {
    TriangleMesh &mesh = *model->meshes[meshID];
    vertexBuffer[meshID].alloc_and_upload(mesh.vertex);
    indexBuffer[meshID].alloc_and_upload(mesh.index);
    if (!mesh.normal.empty())
      normalBuffer[meshID].alloc_and_upload(mesh.normal);
    if (!mesh.texcoord.empty())
      texcoordBuffer[meshID].alloc_and_upload(mesh.texcoord);

    triangleInput[meshID] = {};
    triangleInput[meshID].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    // create local variables, because we need a *pointer* to the
    // device pointers
    d_vertices[meshID] = vertexBuffer[meshID].d_pointer();
    d_indices[meshID] = indexBuffer[meshID].d_pointer();

    triangleInput[meshID].triangleArray.vertexFormat =
        OPTIX_VERTEX_FORMAT_FLOAT3;
    triangleInput[meshID].triangleArray.vertexStrideInBytes = sizeof(vec3f);
    triangleInput[meshID].triangleArray.numVertices = (int)mesh.vertex.size();
    triangleInput[meshID].triangleArray.vertexBuffers = &d_vertices[meshID];

    triangleInput[meshID].triangleArray.indexFormat =
        OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangleInput[meshID].triangleArray.indexStrideInBytes = sizeof(vec3i);
    triangleInput[meshID].triangleArray.numIndexTriplets =
        (int)mesh.index.size();
    triangleInput[meshID].triangleArray.indexBuffer = d_indices[meshID];

    triangleInputFlags[meshID] = 0;

    // in this example we have one SBT entry, and no per-primitive
    // materials:
    triangleInput[meshID].triangleArray.flags = &triangleInputFlags[meshID];
    triangleInput[meshID].triangleArray.numSbtRecords = 1;
    triangleInput[meshID].triangleArray.sbtIndexOffsetBuffer = 0;
    triangleInput[meshID].triangleArray.sbtIndexOffsetSizeInBytes = 0;
    triangleInput[meshID].triangleArray.sbtIndexOffsetStrideInBytes = 0;
  }

  // ==================================================================
  // BLAS setup
  // ==================================================================

  OptixAccelBuildOptions accelOptions = {};
  accelOptions.buildFlags =
      OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
  accelOptions.motionOptions.numKeys = 1;
  accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes blasBufferSizes;
  OPTIX_CHECK(optixAccelComputeMemoryUsage(
      optixContext, &accelOptions, triangleInput.data(),
      (int)model->meshes.size(),  // num_build_inputs
      &blasBufferSizes));

  // ==================================================================
  // prepare compaction
  // ==================================================================

  CUDABuffer compactedSizeBuffer;
  compactedSizeBuffer.alloc(sizeof(uint64_t));

  OptixAccelEmitDesc emitDesc;
  emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
  emitDesc.result = compactedSizeBuffer.d_pointer();

  // ==================================================================
  // execute build (main stage)
  // ==================================================================

  CUDABuffer tempBuffer;
  tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

  CUDABuffer outputBuffer;
  outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

  OPTIX_CHECK(optixAccelBuild(
      optixContext,
      /* stream */ 0, &accelOptions, triangleInput.data(),
      (int)model->meshes.size(), tempBuffer.d_pointer(), tempBuffer.sizeInBytes,

      outputBuffer.d_pointer(), outputBuffer.sizeInBytes,

      &asHandle,

      &emitDesc, 1));

  CUDA_SYNC_CHECK();

  // ==================================================================
  // perform compaction
  // ==================================================================
  uint64_t compactedSize;
  compactedSizeBuffer.download(&compactedSize, 1);

  asBuffer.alloc(compactedSize);
  OPTIX_CHECK(optixAccelCompact(optixContext,
                                /*stream:*/ 0, asHandle, asBuffer.d_pointer(),
                                asBuffer.sizeInBytes, &asHandle));
  CUDA_SYNC_CHECK();

  // ==================================================================
  // aaaaaand .... clean up
  // ==================================================================
  outputBuffer.free();  // << the UNcompacted, temporary output buffer
  tempBuffer.free();
  compactedSizeBuffer.free();

  return asHandle;
}

void SampleRenderer::initOptix() {
  cudaFree(0);
  int num_devices;
  cudaGetDeviceCount(&num_devices);
  if (num_devices == 0)
    throw std::runtime_error("no CUDA capable devices found!");

  std::cout << "found " << num_devices << " CUDA devices" << std::endl;

  // int major, minor;
  // cuDeviceGetAttribute;
  std::vector<int> devices(num_devices);
  cudaGetDevice(devices.data());
  // std::cout << "device : " << devices[0] << std::endl;

  int best_device;
  int best_compute_capability = 0;
  for (size_t i = 0; i < num_devices; ++i) {
    int major, minor;
    CUDA_CHECK(DeviceGetAttribute(
        &major, cudaDeviceAttr::cudaDevAttrComputeCapabilityMajor, devices[i]));
    CUDA_CHECK(DeviceGetAttribute(
        &minor, cudaDeviceAttr::cudaDevAttrComputeCapabilityMinor, devices[i]));

    int device_compute_capability = major * 10 + minor;
    if (device_compute_capability > best_compute_capability) {
      best_device = devices[i];
      best_compute_capability = device_compute_capability;
    }
  }

  std::cout << "compute capability : " << best_compute_capability << std::endl;
  compute_capability = best_compute_capability;

  // initialize optix
  OPTIX_CHECK(optixInit());
}

static void context_log_cb(unsigned int level, const char *tag,
                           const char *message, void *) {
  fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
}

static void test(int compute_capability) {
  std::string filename =
      "C:/Users/pansiyuan/dev/learn-optix/12-denoiseSeparateChannels/"
      "toneMap.cu";
  std::cout << filename << std::endl;

#ifdef _DEBUG
  std::string ptx_file = ".debug.ptx";
#else
  std::string ptx_file = ".release.ptx";
#endif
  auto output_filename = filename + ptx_file;
  std::cout << output_filename << std::endl;

  bool should_recompile = true;
  if (std::filesystem::exists(output_filename)) {
    std::filesystem::file_time_type last_write_time_source =
        std::filesystem::last_write_time(filename);
    std::filesystem::file_time_type last_write_time_ptx =
        std::filesystem::last_write_time(output_filename);

    // Recompile if the source file is newer than the binary
    should_recompile = last_write_time_ptx < last_write_time_source;
  }

  std::cout << "should recompile : " << should_recompile << std::endl;
  int max_registers = 64;
  if (should_recompile) {
    std::ifstream src_fin(filename);
    const std::string src_str((std::istreambuf_iterator<char>(src_fin)),
                              std::istreambuf_iterator<char>());

    nvrtcProgram program;
    NVRTC_CHECK(nvrtcCreateProgram(&program, src_str.c_str(), filename.c_str(),
                                   0, NULL, NULL));

    // Configure options
    std::string option_compute =
        "--gpu-architecture=compute_" + std::to_string(compute_capability);
    std::string option_maxregs =
        "--maxrregcount=" + std::to_string(max_registers);

    const char *options[] = {
        "--std=c++17",
        option_compute.c_str(),
        option_maxregs.c_str(),
        "--use_fast_math",
        "--extra-device-vectorization",
        //"--device-debug",
        "-lineinfo",
        "-restrict",
        "-I C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5/include",
        "-I C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.4.0/include",
        "-I C:/Users/pansiyuan/dev/learn-optix/dep",
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

    std::ofstream fs(output_filename);
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
  CU_CHECK(LinkAddFile(link_state, CU_JIT_INPUT_PTX, output_filename.c_str(), 0,
                       nullptr, nullptr));

  void *cubin;
  size_t cubin_size;
  cuLinkComplete(link_state, &cubin, &cubin_size);

  CUmodule cuda_module;
  CU_CHECK(ModuleLoadData(&cuda_module, cubin));

  CU_CHECK(LinkDestroy(link_state));

  if (should_recompile) {
    printf("%s\n", log_buffer);
  }
}

void SampleRenderer::createContext() {
  // for this sample, do everything on one device
  const int deviceID = 0;
  CUDA_CHECK(SetDevice(deviceID));
  CUDA_CHECK(StreamCreate(&stream));

  cudaGetDeviceProperties(&deviceProps, deviceID);
  std::cout << "running on device: " << deviceProps.name << std::endl;

  CUresult cuRes = cuCtxGetCurrent(&cudaContext);
  if (cuRes != CUDA_SUCCESS)
    fprintf(stderr, "Error querying current context: error code %d\n", cuRes);

  OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
  OPTIX_CHECK(optixDeviceContextSetLogCallback(optixContext, context_log_cb,
                                               nullptr, 4));

  test(compute_capability);
}

void SampleRenderer::createModule() {
  moduleCompileOptions.maxRegisterCount = 50;
  moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
  moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

  pipelineCompileOptions = {};
  pipelineCompileOptions.traversableGraphFlags =
      OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  pipelineCompileOptions.usesMotionBlur = false;
  pipelineCompileOptions.numPayloadValues = 2;
  pipelineCompileOptions.numAttributeValues = 2;
  pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
  pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

  pipelineLinkOptions.maxTraceDepth = 2;

  //   const std::string ptxCode = embedded_ptx_code;
  std::ifstream ptx_fin(
      "C:/Users/pansiyuan/dev/learn-optix/11-denoiseColorOnly/"
      "device_programs.ptx");
  const std::string ptx_str((std::istreambuf_iterator<char>(ptx_fin)),
                            std::istreambuf_iterator<char>());

  char log[2048];
  size_t sizeof_log = sizeof(log);
  OPTIX_CHECK(optixModuleCreateFromPTX(
      optixContext, &moduleCompileOptions, &pipelineCompileOptions,
      ptx_str.c_str(), ptx_str.size(), log, &sizeof_log, &module));
  if (sizeof_log > 1) PRINT(log);
}

void SampleRenderer::createRaygenPrograms() {
  // we do a single ray gen program in this example:
  raygenPGs.resize(1);

  OptixProgramGroupOptions pgOptions = {};
  OptixProgramGroupDesc pgDesc = {};
  pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  pgDesc.raygen.module = module;
  pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";

  // OptixProgramGroup raypg;
  char log[2048];
  size_t sizeof_log = sizeof(log);
  OPTIX_CHECK(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions, log,
                                      &sizeof_log, &raygenPGs[0]));
  if (sizeof_log > 1) PRINT(log);
}

/*! does all setup for the miss program(s) we are going to use */
void SampleRenderer::createMissPrograms() {
  char log[2048];
  size_t sizeof_log = sizeof(log);

  // we do a single ray gen program in this example:
  missPGs.resize(RAY_TYPE_COUNT);

  OptixProgramGroupOptions pgOptions = {};
  OptixProgramGroupDesc pgDesc = {};
  pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
  pgDesc.miss.module = module;

  // ------------------------------------------------------------------
  // radiance rays
  // ------------------------------------------------------------------
  pgDesc.miss.entryFunctionName = "__miss__radiance";

  // OptixProgramGroup raypg;

  OPTIX_CHECK(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions, log,
                                      &sizeof_log,
                                      &missPGs[RADIANCE_RAY_TYPE]));
  if (sizeof_log > 1) PRINT(log);

  // ------------------------------------------------------------------
  // shadow rays
  // ------------------------------------------------------------------
  pgDesc.miss.entryFunctionName = "__miss__shadow";

  OPTIX_CHECK(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions, log,
                                      &sizeof_log, &missPGs[SHADOW_RAY_TYPE]));
  if (sizeof_log > 1) PRINT(log);
}

/*! does all setup for the hitgroup program(s) we are going to use */
void SampleRenderer::createHitgroupPrograms() {
  // for this simple example, we set up a single hit group
  hitgroupPGs.resize(RAY_TYPE_COUNT);

  char log[2048];
  size_t sizeof_log = sizeof(log);

  OptixProgramGroupOptions pgOptions = {};
  OptixProgramGroupDesc pgDesc = {};
  pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  pgDesc.hitgroup.moduleCH = module;
  pgDesc.hitgroup.moduleAH = module;

  // -------------------------------------------------------
  // radiance rays
  // -------------------------------------------------------
  pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
  pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

  OPTIX_CHECK(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions, log,
                                      &sizeof_log,
                                      &hitgroupPGs[RADIANCE_RAY_TYPE]));
  if (sizeof_log > 1) PRINT(log);

  // -------------------------------------------------------
  // shadow rays: technically we don't need this hit group,
  // since we just use the miss shader to check if we were not
  // in shadow
  // -------------------------------------------------------
  pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__shadow";
  pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__shadow";

  OPTIX_CHECK(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions, log,
                                      &sizeof_log,
                                      &hitgroupPGs[SHADOW_RAY_TYPE]));
  if (sizeof_log > 1) PRINT(log);
}

/*! assembles the full pipeline of all programs */
void SampleRenderer::createPipeline() {
  std::vector<OptixProgramGroup> programGroups;
  for (auto pg : raygenPGs) programGroups.push_back(pg);
  for (auto pg : missPGs) programGroups.push_back(pg);
  for (auto pg : hitgroupPGs) programGroups.push_back(pg);

  char log[2048];
  size_t sizeof_log = sizeof(log);
  OPTIX_CHECK(optixPipelineCreate(optixContext, &pipelineCompileOptions,
                                  &pipelineLinkOptions, programGroups.data(),
                                  (int)programGroups.size(), log, &sizeof_log,
                                  &pipeline));
  if (sizeof_log > 1) PRINT(log);

  OPTIX_CHECK(
      optixPipelineSetStackSize(/* [in] The pipeline to configure the stack size
                                   for */
                                pipeline,
                                /* [in] The direct stack size requirement for
                                   direct callables invoked from IS or AH. */
                                2 * 1024,
                                /* [in] The direct stack size requirement for
                                   direct
                                   callables invoked from RG, MS, or CH.  */
                                2 * 1024,
                                /* [in] The continuation stack requirement. */
                                2 * 1024,
                                /* [in] The maximum depth of a traversable graph
                                   passed to trace. */
                                1));
  if (sizeof_log > 1) PRINT(log);
}

/*! constructs the shader binding table */
void SampleRenderer::buildSBT() {
  // ------------------------------------------------------------------
  // build raygen records
  // ------------------------------------------------------------------
  std::vector<RaygenRecord> raygenRecords;
  for (int i = 0; i < raygenPGs.size(); i++) {
    RaygenRecord rec;
    OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[i], &rec));
    rec.data = nullptr; /* for now ... */
    raygenRecords.push_back(rec);
  }
  raygenRecordsBuffer.alloc_and_upload(raygenRecords);
  sbt.raygenRecord = raygenRecordsBuffer.d_pointer();

  // ------------------------------------------------------------------
  // build miss records
  // ------------------------------------------------------------------
  std::vector<MissRecord> missRecords;
  for (int i = 0; i < missPGs.size(); i++) {
    MissRecord rec;
    OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i], &rec));
    rec.data = nullptr; /* for now ... */
    missRecords.push_back(rec);
  }
  missRecordsBuffer.alloc_and_upload(missRecords);
  sbt.missRecordBase = missRecordsBuffer.d_pointer();
  sbt.missRecordStrideInBytes = sizeof(MissRecord);
  sbt.missRecordCount = (int)missRecords.size();

  // ------------------------------------------------------------------
  // build hitgroup records
  // ------------------------------------------------------------------
  int numObjects = (int)model->meshes.size();
  std::vector<HitgroupRecord> hitgroupRecords;
  for (int meshID = 0; meshID < numObjects; meshID++) {
    for (int rayID = 0; rayID < RAY_TYPE_COUNT; rayID++) {
      auto mesh = model->meshes[meshID];

      HitgroupRecord rec;
      // all meshes use the same code, so all same hit group
      OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[rayID], &rec));
      rec.data.color = mesh->diffuse;
      if (mesh->diffuseTextureID >= 0) {
        rec.data.hasTexture = true;
        rec.data.texture = textureObjects[mesh->diffuseTextureID];
      } else {
        rec.data.hasTexture = false;
      }
      rec.data.vertex = (vec3f *)vertexBuffer[meshID].d_pointer();
      rec.data.index = (vec3i *)indexBuffer[meshID].d_pointer();
      rec.data.normal = (vec3f *)normalBuffer[meshID].d_pointer();
      rec.data.texcoord = (vec2f *)texcoordBuffer[meshID].d_pointer();
      hitgroupRecords.push_back(rec);
    }
  }
  hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
  sbt.hitgroupRecordBase = hitgroupRecordsBuffer.d_pointer();
  sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
  sbt.hitgroupRecordCount = (int)hitgroupRecords.size();
}

/*! render one frame */
void SampleRenderer::render() {
  // sanity check: make sure we launch only after first resize is
  // already done:
  if (launchParams.frame.size.x == 0) return;

  if (!accumulate) launchParams.frame.frameID = 0;

  launchParamsBuffer.upload(&launchParams, 1);
  launchParams.frame.frameID++;

  OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
                          pipeline, stream,
                          /*! parameters and SBT */
                          launchParamsBuffer.d_pointer(),
                          launchParamsBuffer.sizeInBytes, &sbt,
                          /*! dimensions of the launch: */
                          launchParams.frame.size.x, launchParams.frame.size.y,
                          1));

  OptixDenoiserParams denoiserParams;
  denoiserParams.denoiseAlpha = 1;
  denoiserParams.hdrIntensity = (CUdeviceptr)0;
  if (accumulate)
    denoiserParams.blendFactor = 1.f / (launchParams.frame.frameID);
  else
    denoiserParams.blendFactor = 0.0f;

  // -------------------------------------------------------
  OptixImage2D inputLayer;
  inputLayer.data = renderBuffer.d_pointer();
  /// Width of the image (in pixels)
  inputLayer.width = launchParams.frame.size.x;
  /// Height of the image (in pixels)
  inputLayer.height = launchParams.frame.size.y;
  /// Stride between subsequent rows of the image (in bytes).
  inputLayer.rowStrideInBytes = launchParams.frame.size.x * sizeof(float4);
  /// Stride between subsequent pixels of the image (in bytes).
  /// For now, only 0 or the value that corresponds to a dense packing of pixels
  /// (no gaps) is supported.
  inputLayer.pixelStrideInBytes = sizeof(float4);
  /// Pixel format.
  inputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;

  // -------------------------------------------------------
  OptixImage2D outputLayer;
  outputLayer.data = denoisedBuffer.d_pointer();
  /// Width of the image (in pixels)
  outputLayer.width = launchParams.frame.size.x;
  /// Height of the image (in pixels)
  outputLayer.height = launchParams.frame.size.y;
  /// Stride between subsequent rows of the image (in bytes).
  outputLayer.rowStrideInBytes = launchParams.frame.size.x * sizeof(float4);
  /// Stride between subsequent pixels of the image (in bytes).
  /// For now, only 0 or the value that corresponds to a dense packing of pixels
  /// (no gaps) is supported.
  outputLayer.pixelStrideInBytes = sizeof(float4);
  /// Pixel format.
  outputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;

  // -------------------------------------------------------
  if (denoiserOn) {
#if OPTIX_VERSION >= 70300
    OptixDenoiserGuideLayer denoiserGuideLayer = {};

    OptixDenoiserLayer denoiserLayer = {};
    denoiserLayer.input = inputLayer;
    denoiserLayer.output = outputLayer;

    OPTIX_CHECK(optixDenoiserInvoke(
        denoiser,
        /*stream*/ 0, &denoiserParams, denoiserState.d_pointer(),
        denoiserState.size(), &denoiserGuideLayer, &denoiserLayer, 1,
        /*inputOffsetX*/ 0,
        /*inputOffsetY*/ 0, denoiserScratch.d_pointer(),
        denoiserScratch.size()));
#else
    OPTIX_CHECK(optixDenoiserInvoke(
        denoiser,
        /*stream*/ 0, &denoiserParams, denoiserState.d_pointer(),
        denoiserState.size(), &inputLayer, 1,
        /*inputOffsetX*/ 0,
        /*inputOffsetY*/ 0, &outputLayer, denoiserScratch.d_pointer(),
        denoiserScratch.size()));
#endif
  } else {
    cudaMemcpy((void *)outputLayer.data, (void *)inputLayer.data,
               outputLayer.width * outputLayer.height * sizeof(float4),
               cudaMemcpyDeviceToDevice);
  }

  // sync - make sure the frame is rendered before we download and
  // display (obviously, for a high-performance application you
  // want to use streams and double-buffering, but for this simple
  // example, this will have to do)
  CUDA_SYNC_CHECK();
}

/*! set camera to render with */
void SampleRenderer::setCamera(const Camera &camera) {
  lastSetCamera = camera;
  // reset accumulation
  launchParams.frame.frameID = 0;
  launchParams.camera.position = camera.from;
  launchParams.camera.direction = normalize(camera.at - camera.from);
  const float cosFovy = 0.66f;
  const float aspect =
      float(launchParams.frame.size.x) / float(launchParams.frame.size.y);
  launchParams.camera.horizontal =
      cosFovy * aspect *
      normalize(cross(launchParams.camera.direction, camera.up));
  launchParams.camera.vertical =
      cosFovy * normalize(cross(launchParams.camera.horizontal,
                                launchParams.camera.direction));
}

/*! resize frame buffer to given resolution */
void SampleRenderer::resize(const glm::ivec2 &newSize) {
  if (denoiser) {
    OPTIX_CHECK(optixDenoiserDestroy(denoiser));
  };

  // ------------------------------------------------------------------
  // create the denoiser:
  OptixDenoiserOptions denoiserOptions = {};

#if OPTIX_VERSION >= 70300
  OPTIX_CHECK(optixDenoiserCreate(optixContext, OPTIX_DENOISER_MODEL_KIND_LDR,
                                  &denoiserOptions, &denoiser));
#else
  denoiserOptions.inputKind = OPTIX_DENOISER_INPUT_RGB;

#  if OPTIX_VERSION < 70100
  // these only exist in 7.0, not 7.1
  denoiserOptions.pixelFormat = OPTIX_PIXEL_FORMAT_FLOAT4;
#  endif

  OPTIX_CHECK(optixDenoiserCreate(optixContext, &denoiserOptions, &denoiser));
  OPTIX_CHECK(
      optixDenoiserSetModel(denoiser, OPTIX_DENOISER_MODEL_KIND_LDR, NULL, 0));
#endif

  // .. then compute and allocate memory resources for the denoiser
  OptixDenoiserSizes denoiserReturnSizes;
  OPTIX_CHECK(optixDenoiserComputeMemoryResources(
      denoiser, newSize.x, newSize.y, &denoiserReturnSizes));

#if OPTIX_VERSION < 70100
  denoiserScratch.resize(denoiserReturnSizes.recommendedScratchSizeInBytes);
#else
  denoiserScratch.resize(
      std::max(denoiserReturnSizes.withOverlapScratchSizeInBytes,
               denoiserReturnSizes.withoutOverlapScratchSizeInBytes));
#endif
  denoiserState.resize(denoiserReturnSizes.stateSizeInBytes);

  // ------------------------------------------------------------------
  // resize our cuda frame buffer
  denoisedBuffer.resize(newSize.x * newSize.y * sizeof(float4));
  renderBuffer.resize(newSize.x * newSize.y * sizeof(float4));

  // update the launch parameters that we'll pass to the optix
  // launch:
  launchParams.frame.size = newSize;
  launchParams.frame.colorBuffer = (float4 *)renderBuffer.d_pointer();

  // and re-set the camera, since aspect may have changed
  setCamera(lastSetCamera);

  // ------------------------------------------------------------------
  OPTIX_CHECK(
      optixDenoiserSetup(denoiser, 0, newSize.x, newSize.y,
                         denoiserState.d_pointer(), denoiserState.size(),
                         denoiserScratch.d_pointer(), denoiserScratch.size()));
}

void SampleRenderer::downloadPixels(vec4f h_pixels[]) {
  denoisedBuffer.download(
      h_pixels, launchParams.frame.size.x * launchParams.frame.size.y);
}