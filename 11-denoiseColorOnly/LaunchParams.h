#pragma once

#include <stdint.h>
#include <glm/glm.hpp>

using vec2i = glm::ivec2;
using vec2f = glm::fvec2;
using vec3f = glm::fvec3;
using vec3i = glm::ivec3;
using vec4f = glm::fvec4;

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
  int numPixelSamples = 1;
  struct {
    int frameID = 0;
    float4 *colorBuffer;

    /*! the size of the frame buffer to render */
    vec2i size;
  } frame;

  struct {
    vec3f position;
    vec3f direction;
    vec3f horizontal;
    vec3f vertical;
  } camera;

  struct {
    vec3f origin, du, dv, power;
  } light;

  OptixTraversableHandle traversable;
};