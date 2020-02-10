#include <scene.h>
#include <sphere.h>

namespace engine
{

Scene::Scene()
    : m_shapes()
{
    m_shapes.push_back(std::make_shared<Sphere>(Sphere(glm::vec3(-2.0f, -2.0f, -2.0f), 0.8f)));
    m_shapes.push_back(std::make_shared<Sphere>(Sphere(glm::vec3(0.0f, 0.0f, 0.0f), 0.8f)));
    m_shapes.push_back(std::make_shared<Sphere>(Sphere(glm::vec3(2.0f, 2.0f, -2.0f), 0.8f)));
}

} // namespace engine