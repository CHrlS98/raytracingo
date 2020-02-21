#include <scene.h>
#include <sphere.h>
#include <plane.h>

#include <PointLight.h>
#include <BasicMaterial.h>

namespace engine
{
namespace host
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
    BasicMaterial yellow(
        glm::vec3(1.0f, 1.0f, 0.0f), // ka
        glm::vec3(1.0f, 1.0f, 0.0f), // kd
        glm::vec3(1.0f, 1.0f, 1.0f), // ks
        glm::vec3(0.2f, 0.2f, 0.2f), // kr
        30.0f                        // alpha
    );

    BasicMaterial purple(
        glm::vec3(1.0f, 0.0f, 1.0f), // ka
        glm::vec3(1.0f, 0.0f, 1.0f), // kd
        glm::vec3(1.0f, 1.0f, 1.0f), // ks
        glm::vec3(0.0f, 0.0f, 0.0f), // kr
        30.0f                        // alpha
    );

    BasicMaterial cyan(
        glm::vec3(0.0f, 1.0f, 1.0f), // ka
        glm::vec3(0.0f, 1.0f, 1.0f), // kd
        glm::vec3(1.0f, 1.0f, 1.0f), // ks
        glm::vec3(0.5f, 0.5f, 0.5f), // kr
        30.0f                        // alpha
    );

    BasicMaterial darkRed(
        glm::vec3(0.5f, 0.0f, 0.0f), // ka
        glm::vec3(0.5f, 0.0f, 0.0f), // kd
        glm::vec3(1.0f, 1.0f, 1.0f), // ks
        glm::vec3(0.05f, 0.05f, 0.05f), // kr
        30.0f                        // alpha
    );

    m_shapes.push_back(std::make_shared<Sphere>(Sphere(glm::vec3(0.45f, 2.0f, -2.0f), 0.8f, yellow)));
    m_shapes.push_back(std::make_shared<Sphere>(Sphere(glm::vec3(0.0f, 0.0f, -2.0f), 0.8f, purple)));
    m_shapes.push_back(std::make_shared<Sphere>(Sphere(glm::vec3(2.0f, 2.0f, -3.0f), 1.0f, cyan)));

    m_shapes.push_back(std::make_shared<Plane>(Plane(glm::vec3(0.0f, -2.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), darkRed)));
}

void Scene::SetupLights()
{
    m_ambientLight = { 0.2, 0.2, 0.2 };

    m_lights.push_back(PointLight({ 0.0, 10.0, -2.0 }, { 0.3, 0.3, 0.3 }));
    m_lights.push_back(PointLight({ -10.0, 10.0, 10.0 }, { 0.3, 0.3, 0.3 }));
    m_lights.push_back(PointLight({ 1000.0, 0.0, 0.0 }, { 0.3, 0.3, 0.3 }));
}

} // namespace host
} // namespace engine
