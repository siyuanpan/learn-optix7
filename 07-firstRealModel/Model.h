#pragma once

#include <vector>

#include "GLFWindow.h"

struct TriangleMesh {
  std::vector<vec3f> vertex;
  std::vector<vec3f> normal;
  std::vector<vec2f> texcoord;
  std::vector<vec3i> index;

  // material data:
  vec3f diffuse;
};

struct Model {
  ~Model() {
    for (auto mesh : meshes) delete mesh;
  }

  std::vector<TriangleMesh *> meshes;
  //! bounding box of all vertices in the model
  box3f bounds;
};

Model *loadOBJ(const std::string &objFile);