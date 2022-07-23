#include <optix_device.h>

#include "LaunchParams.h"

extern "C" __constant__ LaunchParams optixLaunchParams;

// for this simple example, we have a single ray type
enum { SURFACE_RAY_TYPE = 0, RAY_TYPE_COUNT };

static __forceinline__ __device__ void* unpackPointer(uint32_t i0,
                                                      uint32_t i1) {
  const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
  void* ptr = reinterpret_cast<void*>(uptr);
  return ptr;
}

static __forceinline__ __device__ void packPointer(void* ptr, uint32_t& i0,
                                                   uint32_t& i1) {
  const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
  i0 = uptr >> 32;
  i1 = uptr & 0x00000000ffffffff;
}

template <typename T>
static __forceinline__ __device__ T* getPRD() {
  const uint32_t u0 = optixGetPayload_0();
  const uint32_t u1 = optixGetPayload_1();
  return reinterpret_cast<T*>(unpackPointer(u0, u1));
}

static __forceinline__ __device__ vec3f randomColor(int i) {
  int r = unsigned(i) * 13 * 17 + 0x234235;
  int g = unsigned(i) * 7 * 3 * 5 + 0x773477;
  int b = unsigned(i) * 11 * 19 + 0x223766;
  return vec3f((r & 255) / 255.f, (g & 255) / 255.f, (b & 255) / 255.f);
}

extern "C" __global__ void __closesthit__radiance() {
  const TriangleMeshSBTData& sbtData =
      *(const TriangleMeshSBTData*)optixGetSbtDataPointer();

  // compute normal:
  const int primID = optixGetPrimitiveIndex();
  const vec3i index = sbtData.index[primID];
  const vec3f& A = sbtData.vertex[index.x];
  const vec3f& B = sbtData.vertex[index.y];
  const vec3f& C = sbtData.vertex[index.z];
  const vec3f Ng = normalize(cross(B - A, C - A));

  const float3 _ray_dir = optixGetWorldRayDirection();
  const vec3f rayDir{_ray_dir.x, _ray_dir.y, _ray_dir.z};
  const float cosDN = 0.2f + .8f * fabsf(dot(rayDir, Ng));
  vec3f& prd = *(vec3f*)getPRD<vec3f>();
  prd = cosDN * sbtData.color;
  // prd = randomColor(primID);
}

extern "C" __global__ void
__anyhit__radiance() { /*! for this simple example, this will remain empty */
}

extern "C" __global__ void __miss__radiance() {
  glm::fvec3& prd = *(glm::fvec3*)getPRD<glm::fvec3>();
  // set to constant white as background color
  prd = glm::fvec3(1.f);
}

extern "C" __global__ void __raygen__renderFrame() {
  // compute a test pattern based on pixel ID
  const int ix = optixGetLaunchIndex().x;
  const int iy = optixGetLaunchIndex().y;

  const auto& camera = optixLaunchParams.camera;

  // our per-ray data for this example. what we initialize it to
  // won't matter, since this value will be overwritten by either
  // the miss or hit program, anyway
  vec3f pixelColorPRD = vec3f(0.f);

  // the values we store the PRD pointer in:
  uint32_t u0, u1;
  packPointer(&pixelColorPRD, u0, u1);

  // normalized screen plane position, in [0,1]^2
  const vec2f screen(vec2f(ix + .5f, iy + .5f) /
                     vec2f(optixLaunchParams.frame.size));

  // generate ray direction
  vec3f rayDir =
      normalize(camera.direction + (screen.x - 0.5f) * camera.horizontal +
                (screen.y - 0.5f) * camera.vertical);

  float3 cam_pos{camera.position.x, camera.position.y, camera.position.z};
  float3 ray_dir{rayDir.x, rayDir.y, rayDir.z};

  optixTrace(optixLaunchParams.traversable, cam_pos, ray_dir,
             0.f,    // tmin
             1e20f,  // tmax
             0.0f,   // rayTime
             OptixVisibilityMask(255),
             OPTIX_RAY_FLAG_DISABLE_ANYHIT,  // OPTIX_RAY_FLAG_NONE,
             SURFACE_RAY_TYPE,               // SBT offset
             RAY_TYPE_COUNT,                 // SBT stride
             SURFACE_RAY_TYPE,               // missSBTIndex
             u0, u1);

  const int r = int(255.99f * pixelColorPRD.x);
  const int g = int(255.99f * pixelColorPRD.y);
  const int b = int(255.99f * pixelColorPRD.z);

  // convert to 32-bit rgba value (we explicitly set alpha to 0xff
  // to make stb_image_write happy ...
  const uint32_t rgba = 0xff000000 | (r << 0) | (g << 8) | (b << 16);

  // and write to frame buffer ...
  const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;
  optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
}