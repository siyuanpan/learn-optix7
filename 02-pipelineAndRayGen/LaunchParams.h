#pragma once

#include <stdint.h>
#include <glm/glm.hpp>

struct LaunchParams {
  int frameID{0};
  uint32_t *colorBuffer;
  glm::ivec2 fbSize;
};