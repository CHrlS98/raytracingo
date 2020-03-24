#pragma once

#include <shapefactory.h>

#include <vector>
#include <memory>

#include <glm/vec3.hpp>
#include <sutil/Camera.h>

namespace engine 
{
namespace host
{
class Shape;
class PointLight;

class Scene
{
public:
    Scene(const unsigned int& camWidth, const unsigned int& camHeight);
    ~Scene() = default;

    inline std::vector<std::shared_ptr<Shape>> GetShapes() const { return m_shapes; }
    inline std::vector<PointLight> GetLights() const { return m_lights; }
    inline glm::vec3 GetAmbientLight() const { return m_ambientLight; }
    inline std::shared_ptr<sutil::Camera> GetCamera() const { return m_camera; }
    inline unsigned int GetCameraWidth() const { return m_cameraWidth; }
    inline unsigned int GetCameraHeight() const { return m_cameraHeight; }
    inline glm::vec3 GetBackgroundColor() const { return m_backgroundColor; }
    inline int GetNbObjects() const { return m_nbObjects; }

private:
    std::vector<std::shared_ptr<Shape>> m_shapes;
    std::vector<PointLight> m_lights;
    std::shared_ptr<sutil::Camera> m_camera;
    glm::vec3 m_ambientLight;
    glm::vec3 m_backgroundColor;
    ShapeFactory m_factory;
    int m_nbObjects;
    unsigned int m_cameraWidth;
    unsigned int m_cameraHeight;
    void SetupCamera();
    void SetupObjects();
    void SetupLights();
    void AddObject(const std::pair<std::shared_ptr<Shape>, int>& object);
    sutil::Matrix4x4 GetScale(float sx, float sy, float sz) const;
    sutil::Matrix4x4 GetTranslate(float tx, float ty, float tz) const;
    sutil::Matrix4x4 GetRotate(float angleRad, float vx, float vy, float vz) const;
};
}
} // namespace engine
