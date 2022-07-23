#include <iostream>

#include "SampleRenderer.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main(int ac, char** av) {
  try {
    std::cout << "initializing optix..." << std::endl;

    SampleRenderer sample;

    const glm::ivec2 fbSize(1200, 1024);
    sample.resize(fbSize);
    sample.render();

    std::vector<uint32_t> pixels(fbSize.x * fbSize.y);
    sample.downloadPixels(pixels.data());

    const std::string fileName = "example2.png";
    stbi_write_png(fileName.c_str(), fbSize.x, fbSize.y, 4, pixels.data(),
                   fbSize.x * sizeof(uint32_t));
    std::cout << "Image rendered, and saved to " << fileName << " ... done."
              << std::endl;

  } catch (std::runtime_error& e) {
    std::cout << "FATAL ERROR: " << e.what() << std::endl;
    exit(1);
  }
  return 0;
}