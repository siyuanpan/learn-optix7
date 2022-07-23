#include <iostream>

#include "SampleRenderer.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "GLFWindow.h"
#include <gl/GL.h>

struct SampleWindow : public GLFCameraWindow {
  SampleWindow(const std::string& title, const std::vector<TriangleMesh>& model,
               const Camera& camera, const float worldScale)
      : GLFCameraWindow(title, camera.from, camera.at, camera.up, worldScale),
        sample(model) {
    sample.setCamera(camera);
  }

  virtual void render() override {
    if (cameraFrame.modified) {
      sample.setCamera(Camera{cameraFrame.get_from(), cameraFrame.get_at(),
                              cameraFrame.get_up()});
      cameraFrame.modified = false;
    }
    sample.render();
  }

  virtual void draw() override {
    sample.downloadPixels(pixels.data());
    if (fbTexture == 0) glGenTextures(1, &fbTexture);

    glBindTexture(GL_TEXTURE_2D, fbTexture);
    GLenum texFormat = GL_RGBA;
    GLenum texelType = GL_UNSIGNED_BYTE;
    glTexImage2D(GL_TEXTURE_2D, 0, texFormat, fbSize.x, fbSize.y, 0, GL_RGBA,
                 texelType, pixels.data());

    glDisable(GL_LIGHTING);
    glColor3f(1, 1, 1);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, fbTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glDisable(GL_DEPTH_TEST);

    glViewport(0, 0, fbSize.x, fbSize.y);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.f, (float)fbSize.x, 0.f, (float)fbSize.y, -1.f, 1.f);

    glBegin(GL_QUADS);
    {
      glTexCoord2f(0.f, 0.f);
      glVertex3f(0.f, 0.f, 0.f);

      glTexCoord2f(0.f, 1.f);
      glVertex3f(0.f, (float)fbSize.y, 0.f);

      glTexCoord2f(1.f, 1.f);
      glVertex3f((float)fbSize.x, (float)fbSize.y, 0.f);

      glTexCoord2f(1.f, 0.f);
      glVertex3f((float)fbSize.x, 0.f, 0.f);
    }
    glEnd();
  }

  virtual void resize(const vec2i& newSize) {
    fbSize = newSize;
    sample.resize(newSize);
    pixels.resize(newSize.x * newSize.y);
  }

  vec2i fbSize;
  GLuint fbTexture{0};
  SampleRenderer sample;
  std::vector<uint32_t> pixels;
};

int main(int ac, char** av) {
  try {
    std::vector<TriangleMesh> model(2);
    // 100x100 thin ground plane
    model[0].color = vec3f(0.f, 1.f, 0.f);
    model[0].addCube(vec3f(0.f, -1.5f, 0.f), vec3f(10.f, .1f, 10.f));
    // a unit cube centered on top of that
    model[1].color = vec3f(0.f, 1.f, 1.f);
    model[1].addCube(vec3f(0.f, 0.f, 0.f), vec3f(2.f, 2.f, 2.f));

    Camera camera = {/*from*/ vec3f(-10.f, 2.f, -12.f),
                     /* at */ vec3f(0.f, 0.f, 0.f),
                     /* up */ vec3f(0.f, 1.f, 0.f)};

    // something approximating the scale of the world, so the
    // camera knows how much to move for any given user interaction:
    const float worldScale = 10.f;

    SampleWindow* window =
        new SampleWindow("Optix 7 Course Example", model, camera, worldScale);
    window->run();

  } catch (std::runtime_error& e) {
    std::cout << "FATAL ERROR: " << e.what() << std::endl;
    exit(1);
  }
  return 0;
}