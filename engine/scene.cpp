#include <scene.h>
#include <sphere.h>
#include <plane.h>

#include <PointLight.h>
#include <BasicMaterial.h>

namespace engine
{
namespace host
{

Scene::Scene(const unsigned int& camWidth, const unsigned int& camHeight)
    : m_shapes()
    , m_lights()
    , m_cameraWidth(camWidth)
    , m_cameraHeight(camHeight)
    , m_ambientLight()
    , m_backgroundColor()
    , m_camera(nullptr)
{
    SetupObjects();
    SetupLights();
    SetupCamera();
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
        glm::vec3(1.0f, 1.0f, 1.0f), // kr
        30.0f                        // alpha
    );

    BasicMaterial blackMirror(
        glm::vec3(0.1f, 0.1f, 0.1f), // ka
        glm::vec3(0.1f, 0.1f, 0.1f), // kd
        glm::vec3(0.0f, 0.0f, 0.0f), // ks
        glm::vec3(1.0f, 1.0f, 1.0f), // kr
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
        glm::vec3(0.0f, 0.0f, 0.0f), // ks
        glm::vec3(0.1f, 0.1f, 0.1f), // kr
        30.0f                        // alpha
    );

    BasicMaterial white(
        glm::vec3(0.6f, 0.6f, 0.6f), // ka
        glm::vec3(0.6f, 0.6f, 0.6f), // kd
        glm::vec3(1.0f, 1.0f, 1.0f), // ks
        glm::vec3(0.2f, 0.2f, 0.2f), // kr
        30.0f                        // alpha
    );

    //m_shapes.push_back(std::make_shared<Sphere>(Sphere(glm::vec3(0.45f, 2.0f, -2.0f), 0.8f, yellow)));
    //m_shapes.push_back(std::make_shared<Sphere>(Sphere(glm::vec3(0.0f, 0.0f, -2.0f), 0.8f, purple)));
    m_shapes.push_back(std::make_shared<Sphere>(Sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f, blackMirror)));
    //m_shapes.push_back(std::make_shared<Sphere>(Sphere(glm::vec3(2.0f, 2.0f, -3.0f), 1.0f, cyan)));
    //m_shapes.push_back(std::make_shared<Sphere>(Sphere(glm::vec3(-2.0f, 2.0f, -3.0f), 1.2f, white)));

    m_shapes.push_back(std::make_shared<Plane>(Plane(glm::vec3(0.0f, -2.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), darkRed)));
}

void Scene::SetupLights()
{
    m_ambientLight = { 0.4, 0.4, 0.4 };

    m_lights.push_back(PointLight({ 0.0, 10.0, 0.0 }, { 0.6, 0.6, 0.6 }));
}

void Scene::SetupCamera()
{
    m_backgroundColor = { 0.0f, 0.0f, 0.0f };
    m_camera.reset(new sutil::Camera(
            { 0.0f, 1.0f, 7.0f }, // Position de l'oeil
            { 0.0f, 0.0f, 0.0f }, // Point au centre du regard
            { 0.0f, 1.0f, 0.0f }, // Vecteur haut
            60.0f, // Field of view
            static_cast<float>(m_cameraWidth) / static_cast<float>(m_cameraHeight) // aspect ratio
        )
    );
}

} // namespace host
} // namespace engine
