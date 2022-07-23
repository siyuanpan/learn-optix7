#include <optix_device.h>

#include "LaunchParams.h"
#include "random.h"

#define NUM_LIGHT_SAMPLES 1
#define NUM_PIXEL_SAMPLES 16

typedef LCG<16> Random;

extern "C" __constant__ LaunchParams optixLaunchParams;

/*! per-ray data now captures random number generator, so programs
    can access RNG state */
struct PRD {
  Random random;
  vec3f pixelColor;
};

// for this simple example, we have a single ray type

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

static __forceinline__ __device__ vec3f float3_to_vec3f(float3 v) {
  return vec3f{v.x, v.y, v.z};
}

static __forceinline__ __device__ float3 vec3f_to_float3(vec3f v) {
  return float3{v.x, v.y, v.z};
}

extern "C" __global__ void __closesthit__shadow() {
  /* not going to be used ... */
}

extern "C" __global__ void __closesthit__radiance() {
  const TriangleMeshSBTData& sbtData =
      *(const TriangleMeshSBTData*)optixGetSbtDataPointer();
  PRD& prd = *getPRD<PRD>();

  // ------------------------------------------------------------------
  // gather some basic hit information
  // ------------------------------------------------------------------
  const int primID = optixGetPrimitiveIndex();
  const vec3i index = sbtData.index[primID];
  const float u = optixGetTriangleBarycentrics().x;
  const float v = optixGetTriangleBarycentrics().y;

  // ------------------------------------------------------------------
  // compute normal, using either shading normal (if avail), or
  // geometry normal (fallback)
  // ------------------------------------------------------------------
  const vec3f& A = sbtData.vertex[index.x];
  const vec3f& B = sbtData.vertex[index.y];
  const vec3f& C = sbtData.vertex[index.z];
  vec3f Ng = cross(B - A, C - A);
  vec3f Ns = (sbtData.normal)
                 ? ((1.f - u - v) * sbtData.normal[index.x] +
                    u * sbtData.normal[index.y] + v * sbtData.normal[index.z])
                 : Ng;

  // ------------------------------------------------------------------
  // face-forward and normalize normals
  // ------------------------------------------------------------------
  const vec3f rayDir = float3_to_vec3f(optixGetWorldRayDirection());

  if (dot(rayDir, Ng) > 0.f) Ng = -Ng;
  Ng = normalize(Ng);

  if (dot(Ng, Ns) < 0.f) Ns -= 2.f * dot(Ng, Ns) * Ng;
  Ns = normalize(Ns);

  // ------------------------------------------------------------------
  // compute diffuse material color, including diffuse texture, if
  // available
  // ------------------------------------------------------------------
  vec3f diffuseColor = sbtData.color;
  if (sbtData.hasTexture && sbtData.texcoord) {
    const vec2f tc = (1.f - u - v) * sbtData.texcoord[index.x] +
                     u * sbtData.texcoord[index.y] +
                     v * sbtData.texcoord[index.z];

    float4 fromTexture = tex2D<float4>(sbtData.texture, tc.x, tc.y);
    diffuseColor *= vec3f(fromTexture.x, fromTexture.y, fromTexture.z);
  }

  // start with some ambient term
  vec3f pixelColor = (0.1f + 0.2f * fabsf(dot(Ns, rayDir))) * diffuseColor;

  // ------------------------------------------------------------------
  // compute shadow
  // ------------------------------------------------------------------
  const vec3f surfPos = (1.f - u - v) * sbtData.vertex[index.x] +
                        u * sbtData.vertex[index.y] +
                        v * sbtData.vertex[index.z];
  const int numLightSamples = NUM_LIGHT_SAMPLES;
  for (int lightSampleID = 0; lightSampleID < numLightSamples;
       lightSampleID++) {
    // produce random light sample
    const vec3f lightPos = optixLaunchParams.light.origin +
                           prd.random() * optixLaunchParams.light.du +
                           prd.random() * optixLaunchParams.light.dv;
    vec3f lightDir = lightPos - surfPos;
    float lightDist = length(lightDir);
    lightDir = normalize(lightDir);

    // trace shadow ray:
    const float NdotL = dot(lightDir, Ns);
    if (NdotL >= 0.f) {
      vec3f lightVisibility = vec3f(0.f);
      // the values we store the PRD pointer in:
      uint32_t u0, u1;
      packPointer(&lightVisibility, u0, u1);
      optixTrace(optixLaunchParams.traversable,
                 vec3f_to_float3(surfPos + 1e-3f * Ng),
                 vec3f_to_float3(lightDir),
                 1e-3f,                      // tmin
                 lightDist * (1.f - 1e-3f),  // tmax
                 0.0f,                       // rayTime
                 OptixVisibilityMask(255),
                 // For shadow rays: skip any/closest hit shaders and terminate
                 // on first intersection with anything. The miss shader is used
                 // to mark if the light was visible.
                 OPTIX_RAY_FLAG_DISABLE_ANYHIT |
                     OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT |
                     OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                 SHADOW_RAY_TYPE,  // SBT offset
                 RAY_TYPE_COUNT,   // SBT stride
                 SHADOW_RAY_TYPE,  // missSBTIndex
                 u0, u1);
      pixelColor += lightVisibility * optixLaunchParams.light.power *
                    diffuseColor *
                    (NdotL / (lightDist * lightDist * numLightSamples));
    }
  }

  prd.pixelColor = pixelColor;
}

extern "C" __global__ void
__anyhit__radiance() { /*! for this simple example, this will remain empty */
}

extern "C" __global__ void __anyhit__shadow() { /*! not going to be used */
}

extern "C" __global__ void __miss__radiance() {
  PRD& prd = *getPRD<PRD>();
  // set to constant white as background color
  prd.pixelColor = vec3f(1.f);
}

extern "C" __global__ void __miss__shadow() {
  // we didn't hit anything, so the light is visible
  vec3f& prd = *(vec3f*)getPRD<vec3f>();
  prd = vec3f(1.f);
}

extern "C" __global__ void __raygen__renderFrame() {
  // compute a test pattern based on pixel ID
  const int ix = optixGetLaunchIndex().x;
  const int iy = optixGetLaunchIndex().y;
  const auto& camera = optixLaunchParams.camera;

  PRD prd;
  prd.random.init(ix + optixLaunchParams.frame.size.x * iy,
                  optixLaunchParams.frame.frameID);
  prd.pixelColor = vec3f(0.f);

  // the values we store the PRD pointer in:
  uint32_t u0, u1;
  packPointer(&prd, u0, u1);

  int numPixelSamples = optixLaunchParams.numPixelSamples;

  vec3f pixelColor(0.f);
  for (int sampleID = 0; sampleID < numPixelSamples; sampleID++) {
    // normalized screen plane position, in [0,1]^2
    const vec2f screen(vec2f(ix + prd.random(), iy + prd.random()) /
                       vec2f(optixLaunchParams.frame.size));

    // generate ray direction
    vec3f rayDir =
        normalize(camera.direction + (screen.x - 0.5f) * camera.horizontal +
                  (screen.y - 0.5f) * camera.vertical);
    optixTrace(optixLaunchParams.traversable, vec3f_to_float3(camera.position),
               vec3f_to_float3(rayDir),
               0.f,    // tmin
               1e20f,  // tmax
               0.0f,   // rayTime
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT,  // OPTIX_RAY_FLAG_NONE,
               RADIANCE_RAY_TYPE,              // SBT offset
               RAY_TYPE_COUNT,                 // SBT stride
               RADIANCE_RAY_TYPE,              // missSBTIndex
               u0, u1);
    pixelColor += prd.pixelColor;
  }

  vec4f rgba{pixelColor.x / numPixelSamples, pixelColor.y / numPixelSamples,
             pixelColor.z / numPixelSamples, 1.f};

  // and write to frame buffer ...
  const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;
  if (optixLaunchParams.frame.frameID > 0) {
    rgba += float(optixLaunchParams.frame.frameID) *
            vec4f(optixLaunchParams.frame.colorBuffer[fbIndex].x,
                  optixLaunchParams.frame.colorBuffer[fbIndex].y,
                  optixLaunchParams.frame.colorBuffer[fbIndex].z,
                  optixLaunchParams.frame.colorBuffer[fbIndex].w);
    rgba /= (optixLaunchParams.frame.frameID + 1.f);
  }

  optixLaunchParams.frame.colorBuffer[fbIndex] =
      float4{rgba.r, rgba.g, rgba.b, rgba.a};
}