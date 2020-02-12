#pragma once

#include <vector>
#include <memory>
#include <glm/vec3.hpp>

namespace engine 
{

constexpr unsigned int NB_OBJ = 3;

class IShape;
class PointLight;

class Scene
{
public:
    Scene();
    ~Scene() = default;

    inline std::vector<std::shared_ptr<IShape>> GetShapes() const { return m_shapes; }
    inline std::vector<PointLight> GetLights() const { return m_lights; }
    inline glm::vec3 GetAmbientLight() const { return m_ambientLight; }

private:
    void SetupObjects();
    void SetupLights();
    std::vector<std::shared_ptr<IShape>> m_shapes;
    std::vector<PointLight> m_lights;
    glm::vec3 m_ambientLight;
};

} // namespace engine
