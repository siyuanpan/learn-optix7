// #include "SampleRenderer.h"
// #include <glm/glm.hpp>

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned long uint32_t;
typedef unsigned long long uint64_t;

inline __device__ float4 sqrt(float4 f) {
  return make_float4(sqrtf(f.x), sqrtf(f.y), sqrtf(f.z), sqrtf(f.w));
}
inline __device__ float clampf(float f) { return min(1.f, max(0.f, f)); }
inline __device__ float4 clamp(float4 f) {
  return make_float4(clampf(f.x), clampf(f.y), clampf(f.z), clampf(f.w));
}

/*! runs a cuda kernel that performs gamma correction and float4-to-rgba
 * conversion */
extern "C" __global__ void computeFinalPixelColorsKernel(
    uint32_t *finalColorBuffer, float4 *denoisedBuffer, int2 size) {
  int pixelX = threadIdx.x + blockIdx.x * blockDim.x;
  int pixelY = threadIdx.y + blockIdx.y * blockDim.y;
  if (pixelX >= size.x) return;
  if (pixelY >= size.y) return;

  int pixelID = pixelX + size.x * pixelY;

  float4 f4 = denoisedBuffer[pixelID];
  f4 = clamp(sqrt(f4));
  uint32_t rgba = 0;
  rgba |= (uint32_t)(f4.x * 255.9f) << 0;
  rgba |= (uint32_t)(f4.y * 255.9f) << 8;
  rgba |= (uint32_t)(f4.z * 255.9f) << 16;
  rgba |= (uint32_t)255 << 24;
  finalColorBuffer[pixelID] = rgba;
}

// void SampleRenderer::computeFinalPixelColors() {
//   vec2i fbSize = launchParams.frame.size;
//   vec2i blockSize = vec2i(32);
//   vec2i numBlocks = divRoundUp(fbSize, blockSize);
//   computeFinalPixelColorsKernel<<<dim3(numBlocks.x, numBlocks.y),
//                                   dim3(blockSize.x, blockSize.y)>>>(
//       (uint32_t *)finalColorBuffer.d_pointer(),
//       (float4 *)denoisedBuffer.d_pointer(), fbSize);
// }