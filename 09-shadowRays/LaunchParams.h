#pragma once

#include <stdint.h>
#include <glm/glm.hpp>

using vec2f = glm::fvec2;
using vec3f = glm::fvec3;
using vec3i = glm::ivec3;

enum { RADIANCE_RAY_TYPE = 0, SHADOW_RAY_TYPE, RAY_TYPE_COUNT };

struct TriangleMeshSBTData {
  vec3f color;
  vec3f *vertex;
  vec3f *normal;
  vec2f *texcoord;
  vec3i *index;
  bool hasTexture;
  cudaTextureObject_t texture;
};

struct LaunchParams {
  struct {
    uint32_t *colorBuffer;
    vec2f size;
  } frame;

  // struct {
  //   uint32_t *colorBuffer;
  //   float2 size;
  // } frame;

  struct {
    vec3f position;
    vec3f direction;
    vec3f horizontal;
    vec3f vertical;
  } camera;

  // struct {
  //   float3 position;
  //   float3 direction;
  //   float3 horizontal;
  //   float3 vertical;
  // } camera;

  OptixTraversableHandle traversable;
};