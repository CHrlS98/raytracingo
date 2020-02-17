#include <scene.h>
#include <sphere.h>
#include <plane.h>

#include <PointLight.h>

namespace engine
{

Scene::Scene()
    : m_shapes()
    , m_lights()
    , m_ambientLight()
{
    SetupObjects();
    SetupLights();
}

void Scene::SetupObjects()
{
    m_shapes.push_back(std::make_shared<Sphere>(Sphere(glm::vec3(0.45f, 2.0f, -2.0f), 0.8f)));
    m_shapes.push_back(std::make_shared<Sphere>(Sphere(glm::vec3(0.0f, 0.0f, -2.0f), 0.8f)));
    m_shapes.push_back(std::make_shared<Sphere>(Sphere(glm::vec3(2.0f, 2.0f, -3.0f), 1.0f)));

    m_shapes.push_back(std::make_shared<Plane>(Plane(glm::vec3(0.0f, -2.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f))));
}

void Scene::SetupLights()
{
    m_ambientLight = { 0.2, 0.2, 0.2 };

    m_lights.push_back(PointLight({ 0.0, 10.0, -2.0 }, { 0.6, 0.6, 0.6 }));
    m_lights.push_back(PointLight({ -10.0, 10.0, 10.0 }, { 0.6, 0.0, 0.0 }));
}

} // namespace engine
