#pragma once

#include <vector>
#include <memory>
#include <glm/vec3.hpp>
#include <sutil/Camera.h>

namespace engine 
{
namespace host
{
class IShape;
class PointLight;

class Scene
{
public:
    Scene(const unsigned int& camWidth, const unsigned int& camHeight);
    ~Scene() = default;

    inline std::vector<std::shared_ptr<IShape>> GetShapes() const { return m_shapes; }
    inline std::vector<PointLight> GetLights() const { return m_lights; }
    inline glm::vec3 GetAmbientLight() const { return m_ambientLight; }
    inline std::shared_ptr<sutil::Camera> GetCamera() const { return m_camera; }
    inline unsigned int GetCameraWidth() const { return m_cameraWidth; }
    inline unsigned int GetCameraHeight() const { return m_cameraHeight; }
    inline glm::vec3 GetBackgroundColor() const { return m_backgroundColor; }

private:
    std::vector<std::shared_ptr<IShape>> m_shapes;
    std::vector<PointLight> m_lights;
    std::shared_ptr<sutil::Camera> m_camera;
    glm::vec3 m_ambientLight;
    glm::vec3 m_backgroundColor;
    unsigned int m_cameraWidth;
    unsigned int m_cameraHeight;
    void SetupCamera();
    void SetupObjects();
    void SetupLights();
};
}
} // namespace engine
