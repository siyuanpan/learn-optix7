#pragma once

#include <glm/glm.hpp>
#include <glm/ext.hpp>
// #include <glm/gtx/string_cast.hpp>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <iostream>

#define M_PI 3.14159265358979323846

using vec2i = glm::ivec2;
using vec2f = glm::fvec2;
using vec3f = glm::fvec3;
using vec3i = glm::ivec3;
using vec4f = glm::fvec4;

struct linear3f {
  vec3f vx, vy, vz;

  inline linear3f() = default;
  inline linear3f(const linear3f &other) {
    vx = other.vx;
    vy = other.vy;
    vz = other.vz;
  }
  inline linear3f &operator=(const linear3f &other) {
    vx = other.vx;
    vy = other.vy;
    vz = other.vz;
    return *this;
  }

  inline linear3f(const vec3f &vx, const vec3f &vy, const vec3f &vz)
      : vx(vx), vy(vy), vz(vz) {}

  inline linear3f(const float &m00, const float &m01, const float &m02,
                  const float &m10, const float &m11, const float &m12,
                  const float &m20, const float &m21, const float &m22)
      : vx(m00, m10, m20), vy(m01, m11, m21), vz(m02, m12, m22) {}

  static inline linear3f rotate(const vec3f &_u, const float r) {
    vec3f u = normalize(_u);
    float s = sin(r), c = cos(r);
    return linear3f(
        u.x * u.x + (1 - u.x * u.x) * c, u.x * u.y * (1 - c) - u.z * s,
        u.x * u.z * (1 - c) + u.y * s, u.x * u.y * (1 - c) + u.z * s,
        u.y * u.y + (1 - u.y * u.y) * c, u.y * u.z * (1 - c) - u.x * s,
        u.x * u.z * (1 - c) - u.y * s, u.y * u.z * (1 - c) + u.x * s,
        u.z * u.z + (1 - u.z * u.z) * c);
  }
};

struct affine3f {
  linear3f l;
  vec3f p;
};

struct box3f {
  vec3f lower{std::numeric_limits<float>::max()};
  vec3f upper{std::numeric_limits<float>::lowest()};

  box3f &extend(const vec3f &other) {
    lower = min(lower, other);
    upper = max(upper, other);
    return *this;
  }

  vec3f center() const { return (lower + upper) / 2.f; }

  vec3f span() const { return upper - lower; }
};

inline vec3f operator*(const linear3f &a, const vec3f &b) {
  return b.x * a.vx + b.y * a.vy + b.z * a.vz;
}

inline linear3f operator*(const linear3f &a, const linear3f &b) {
  return linear3f(a * b.vx, a * b.vy, a * b.vz);
}

inline vec3f madd(const vec3f &a, const vec3f &b, const vec3f &c) {
  return a * b + c;
}

inline const vec3f xfmPoint(const affine3f &m, const vec3f &p) {
  return madd(vec3f(p.x), m.l.vx,
              madd(vec3f(p.y), m.l.vy, madd(vec3f(p.z), m.l.vz, m.p)));
}

struct GLFWindow {
  GLFWindow(const std::string &title);
  ~GLFWindow();

  /*! put pixels on the screen ... */
  virtual void draw() { /* empty - to be subclassed by user */
  }

  /*! callback that window got resized */
  virtual void resize(
      const vec2i &newSize) { /* empty - to be subclassed by user */
  }

  virtual void key(int key, int mods) {}

  /*! callback that window got resized */
  virtual void mouseMotion(const vec2i &newPos) {}

  /*! callback that window got resized */
  virtual void mouseButton(int button, int action, int mods) {}

  inline vec2i getMousePos() const {
    double x, y;
    glfwGetCursorPos(handle, &x, &y);
    return vec2i((int)x, (int)y);
  }

  /*! re-render the frame - typically part of draw(), but we keep
    this a separate function so render() can focus on optix
    rendering, and now have to deal with opengl pixel copies
    etc */
  virtual void render() { /* empty - to be subclassed by user */
  }

  /*! opens the actual window, and runs the window's events to
    completion. This function will only return once the window
    gets closed */
  void run();

  /*! the glfw window handle */
  GLFWwindow *handle{nullptr};
};

struct CameraFrame {
  CameraFrame(const float worldScale) : motionSpeed(worldScale) {}

  vec3f getPOI() const { return position - poiDistance * frame.vz; }

  /*! re-compute all orientation related fields from given
    'user-style' camera parameters */
  void setOrientation(/* camera origin    : */ const vec3f &origin,
                      /* point of interest: */ const vec3f &interest,
                      /* up-vector        : */ const vec3f &up) {
    position = origin;
    upVector = up;
    frame.vz = (interest == origin)
                   ? vec3f(0, 0, 1)
                   : /* negative because we use NEGATIZE z axis */ -normalize(
                         interest - origin);
    frame.vx = cross(up, frame.vz);
    if (dot(frame.vx, frame.vx) < 1e-8f)
      frame.vx = vec3f(0, 1, 0);
    else
      frame.vx = normalize(frame.vx);
    // frame.vx
    //   = (fabs(dot(up,frame.vz)) < 1e-6f)
    //   ? vec3f(0,1,0)
    //   : normalize(cross(up,frame.vz));
    frame.vy = normalize(cross(frame.vz, frame.vx));
    poiDistance = length(interest - origin);
    forceUpFrame();
  }

  /*! tilt the frame around the z axis such that the y axis is "facing upwards"
   */
  void forceUpFrame() {
    // frame.vz remains unchanged
    if (fabsf(dot(frame.vz, upVector)) < 1e-6f)
      // looking along upvector; not much we can do here ...
      return;
    frame.vx = normalize(cross(upVector, frame.vz));
    frame.vy = normalize(cross(frame.vz, frame.vx));
    modified = true;
  }

  void setUpVector(const vec3f &up) {
    upVector = up;
    forceUpFrame();
  }

  inline float computeStableEpsilon(float f) const {
    return abs(f) * float(1. / (1 << 21));
  }

  inline float computeStableEpsilon(const vec3f v) const {
    return std::max(
        std::max(computeStableEpsilon(v.x), computeStableEpsilon(v.y)),
        computeStableEpsilon(v.z));
  }

  inline vec3f get_from() const { return position; }
  inline vec3f get_at() const { return getPOI(); }
  inline vec3f get_up() const { return upVector; }

  // linear3f frame{one};
  linear3f frame{};
  vec3f position{0, -1, 0};
  /*! distance to the 'point of interst' (poi); e.g., the point we
    will rotate around */
  float poiDistance{1.f};
  vec3f upVector{0, 1, 0};
  /* if set to true, any change to the frame will always use to
     upVector to 'force' the frame back upwards; if set to false,
     the upVector will be ignored */
  bool forceUp{true};

  /*! multiplier how fast the camera should move in world space
    for each unit of "user specifeid motion" (ie, pixel
    count). Initial value typically should depend on the world
    size, but can also be adjusted. This is actually something
    that should be more part of the manipulator widget(s), but
    since that same value is shared by multiple such widgets
    it's easiest to attach it to the camera here ...*/
  float motionSpeed{1.f};

  /*! gets set to true every time a manipulator changes the camera
    values */
  bool modified{true};
};

struct CameraFrameManip {
  CameraFrameManip(CameraFrame *cameraFrame) : cameraFrame(cameraFrame) {}

  /*! this gets called when the user presses a key on the keyboard ... */
  virtual void key(int key, int mods) {
    CameraFrame &fc = *cameraFrame;

    switch (key) {
      case '+':
      case '=':
        fc.motionSpeed *= 1.5f;
        std::cout << "# viewer: new motion speed is " << fc.motionSpeed
                  << std::endl;
        break;
      case '-':
      case '_':
        fc.motionSpeed /= 1.5f;
        std::cout << "# viewer: new motion speed is " << fc.motionSpeed
                  << std::endl;
        break;
      case 'C':
        std::cout << "(C)urrent camera:" << std::endl;
        // std::cout << "- from :" << glm::to_string(fc.position) << std::endl;
        // std::cout << "- poi  :" << glm::to_string(fc.getPOI()) << std::endl;
        // std::cout << "- upVec:" << glm::to_string(fc.upVector) << std::endl;
        // std::cout << "- frame:" << glm::to_string(fc.frame) << std::endl;
        // std::cout << "- frame x:" << glm::to_string(fc.frame.vx) <<
        // std::endl; std::cout << "- frame y:" << glm::to_string(fc.frame.vy)
        // << std::endl; std::cout << "- frame z:" <<
        // glm::to_string(fc.frame.vz) << std::endl;
        break;
      case 'x':
      case 'X':
        fc.setUpVector(fc.upVector == vec3f(1, 0, 0) ? vec3f(-1, 0, 0)
                                                     : vec3f(1, 0, 0));
        break;
      case 'y':
      case 'Y':
        fc.setUpVector(fc.upVector == vec3f(0, 1, 0) ? vec3f(0, -1, 0)
                                                     : vec3f(0, 1, 0));
        break;
      case 'z':
      case 'Z':
        fc.setUpVector(fc.upVector == vec3f(0, 0, 1) ? vec3f(0, 0, -1)
                                                     : vec3f(0, 0, 1));
        break;
      default:
        break;
    }
  }

  virtual void strafe(const vec3f &howMuch) {
    cameraFrame->position += howMuch;
    cameraFrame->modified = true;
  }
  /*! strafe, in screen space */
  virtual void strafe(const vec2f &howMuch) {
    strafe(+howMuch.x * cameraFrame->frame.vx -
           howMuch.y * cameraFrame->frame.vy);
  }

  virtual void move(const float step) = 0;
  virtual void rotate(const float dx, const float dy) = 0;

  // /*! this gets called when the user presses a key on the keyboard ... */
  // virtual void special(int key, const vec2i &where) { };

  /*! mouse got dragged with left button pressedn, by 'delta'
    pixels, at last position where */
  virtual void mouseDragLeft(const vec2f &delta) {
    rotate(delta.x * degrees_per_drag_fraction,
           delta.y * degrees_per_drag_fraction);
  }

  /*! mouse got dragged with left button pressedn, by 'delta'
    pixels, at last position where */
  virtual void mouseDragMiddle(const vec2f &delta) {
    strafe(delta * pixels_per_move * cameraFrame->motionSpeed);
  }

  /*! mouse got dragged with left button pressedn, by 'delta'
    pixels, at last position where */
  virtual void mouseDragRight(const vec2f &delta) {
    move(delta.y * pixels_per_move * cameraFrame->motionSpeed);
  }

  // /*! mouse button got either pressed or released at given location */
  // virtual void mouseButtonLeft  (const vec2i &where, bool pressed) {}

  // /*! mouse button got either pressed or released at given location */
  // virtual void mouseButtonMiddle(const vec2i &where, bool pressed) {}

  // /*! mouse button got either pressed or released at given location */
  // virtual void mouseButtonRight (const vec2i &where, bool pressed) {}

 protected:
  CameraFrame *cameraFrame;
  const float kbd_rotate_degrees{10.f};
  const float degrees_per_drag_fraction{150.f};
  const float pixels_per_move{10.f};
};

struct GLFCameraWindow : public GLFWindow {
  GLFCameraWindow(const std::string &title, const vec3f &camera_from,
                  const vec3f &camera_at, const vec3f &camera_up,
                  const float worldScale)
      : GLFWindow(title), cameraFrame(worldScale) {
    cameraFrame.setOrientation(camera_from, camera_at, camera_up);
    enableFlyMode();
    enableInspectMode();
  }

  void enableFlyMode();
  void enableInspectMode();

  // /*! put pixels on the screen ... */
  // virtual void draw()
  // { /* empty - to be subclassed by user */ }

  // /*! callback that window got resized */
  // virtual void resize(const vec2i &newSize)
  // { /* empty - to be subclassed by user */ }

  virtual void key(int key, int mods) override {
    switch (key) {
      case 'f':
      case 'F':
        std::cout << "Entering 'fly' mode" << std::endl;
        if (flyModeManip) cameraFrameManip = flyModeManip;
        break;
      case 'i':
      case 'I':
        std::cout << "Entering 'inspect' mode" << std::endl;
        if (inspectModeManip) cameraFrameManip = inspectModeManip;
        break;
      default:
        if (cameraFrameManip) cameraFrameManip->key(key, mods);
    }
  }

  /*! callback that window got resized */
  virtual void mouseMotion(const vec2i &newPos) override {
    vec2i windowSize;
    glfwGetWindowSize(handle, &windowSize.x, &windowSize.y);

    if (isPressed.leftButton && cameraFrameManip)
      cameraFrameManip->mouseDragLeft(vec2f(newPos - lastMousePos) /
                                      vec2f(windowSize));
    if (isPressed.rightButton && cameraFrameManip)
      cameraFrameManip->mouseDragRight(vec2f(newPos - lastMousePos) /
                                       vec2f(windowSize));
    if (isPressed.middleButton && cameraFrameManip)
      cameraFrameManip->mouseDragMiddle(vec2f(newPos - lastMousePos) /
                                        vec2f(windowSize));
    lastMousePos = newPos;
    /* empty - to be subclassed by user */
  }

  /*! callback that window got resized */
  virtual void mouseButton(int button, int action, int mods) override {
    const bool pressed = (action == GLFW_PRESS);
    switch (button) {
      case GLFW_MOUSE_BUTTON_LEFT:
        isPressed.leftButton = pressed;
        break;
      case GLFW_MOUSE_BUTTON_MIDDLE:
        isPressed.middleButton = pressed;
        break;
      case GLFW_MOUSE_BUTTON_RIGHT:
        isPressed.rightButton = pressed;
        break;
    }
    lastMousePos = getMousePos();
  }

  // /*! mouse got dragged with left button pressedn, by 'delta'
  //   pixels, at last position where */
  // virtual void mouseDragLeft  (const vec2i &where, const vec2i &delta) {}

  // /*! mouse got dragged with left button pressedn, by 'delta'
  //   pixels, at last position where */
  // virtual void mouseDragRight (const vec2i &where, const vec2i &delta) {}

  // /*! mouse got dragged with left button pressedn, by 'delta'
  //   pixels, at last position where */
  // virtual void mouseDragMiddle(const vec2i &where, const vec2i &delta) {}

  /*! a (global) pointer to the currently active window, so we can
    route glfw callbacks to the right GLFWindow instance (in this
    simplified library we only allow on window at any time) */
  // static GLFWindow *current;

  struct {
    bool leftButton{false}, middleButton{false}, rightButton{false};
  } isPressed;
  vec2i lastMousePos = {-1, -1};

  friend struct CameraFrameManip;

  CameraFrame cameraFrame;
  std::shared_ptr<CameraFrameManip> cameraFrameManip;
  std::shared_ptr<CameraFrameManip> inspectModeManip;
  std::shared_ptr<CameraFrameManip> flyModeManip;
};

struct InspectModeManip : public CameraFrameManip {
  InspectModeManip(CameraFrame *cameraFrame) : CameraFrameManip(cameraFrame) {}

 private:
  /*! helper function: rotate camera frame by given degrees, then
    make sure the frame, poidistance etc are all properly set,
    the widget gets notified, etc */
  virtual void rotate(const float deg_u, const float deg_v) override {
    float rad_u = -M_PI / 180.f * deg_u;
    float rad_v = -M_PI / 180.f * deg_v;

    CameraFrame &fc = *cameraFrame;

    const vec3f poi = fc.getPOI();
    fc.frame = linear3f::rotate(fc.frame.vy, rad_u) *
               linear3f::rotate(fc.frame.vx, rad_v) * fc.frame;

    if (fc.forceUp) fc.forceUpFrame();

    fc.position = poi + fc.poiDistance * fc.frame.vz;
    fc.modified = true;
  }

  /*! helper function: move forward/backwards by given multiple of
    motion speed, then make sure the frame, poidistance etc are
    all properly set, the widget gets notified, etc */
  virtual void move(const float step) override {
    const vec3f poi = cameraFrame->getPOI();
    // inspectmode can't get 'beyond' the look-at point:
    const float minReqDistance = 0.1f * cameraFrame->motionSpeed;
    cameraFrame->poiDistance =
        std::max(minReqDistance, cameraFrame->poiDistance - step);
    cameraFrame->position =
        poi + cameraFrame->poiDistance * cameraFrame->frame.vz;
    cameraFrame->modified = true;
  }
};

struct FlyModeManip : public CameraFrameManip {
  FlyModeManip(CameraFrame *cameraFrame) : CameraFrameManip(cameraFrame) {}

 private:
  /*! helper function: rotate camera frame by given degrees, then
    make sure the frame, poidistance etc are all properly set,
    the widget gets notified, etc */
  virtual void rotate(const float deg_u, const float deg_v) override {
    float rad_u = -M_PI / 180.f * deg_u;
    float rad_v = -M_PI / 180.f * deg_v;

    CameraFrame &fc = *cameraFrame;

    // const vec3f poi  = fc.getPOI();
    fc.frame = linear3f::rotate(fc.frame.vy, rad_u) *
               linear3f::rotate(fc.frame.vx, rad_v) * fc.frame;

    if (fc.forceUp) fc.forceUpFrame();

    fc.modified = true;
  }

  /*! helper function: move forward/backwards by given multiple of
    motion speed, then make sure the frame, poidistance etc are
    all properly set, the widget gets notified, etc */
  virtual void move(const float step) override {
    cameraFrame->position += step * cameraFrame->frame.vz;
    cameraFrame->modified = true;
  }
};

inline void GLFCameraWindow::enableFlyMode() {
  flyModeManip = std::make_shared<FlyModeManip>(&cameraFrame);
  cameraFrameManip = flyModeManip;
}

inline void GLFCameraWindow::enableInspectMode() {
  inspectModeManip = std::make_shared<InspectModeManip>(&cameraFrame);
  cameraFrameManip = inspectModeManip;
}