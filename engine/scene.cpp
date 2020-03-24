#include <scene.h>

#include <PointLight.h>
#include <BasicMaterial.h>
#include <materials.h>
#include<sutil/Matrix.h>

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
    , m_factory()
    , m_nbObjects(0)
{
    SetupObjects();
    SetupLights();
    SetupCamera();
}

void Scene::SetupObjects()
{
    // disks
    std::pair<std::shared_ptr<Shape>, int> disk0 = m_factory.CreateDisk(GetScale(4.0f, 1.0f, 4.0f), materials::purple);
    disk0.first->Transform(GetTranslate(0.0f, -1.0f, 0.0f));
    AddObject(disk0);
    
    // spheres
    std::pair<std::shared_ptr<Shape>, int> sphere0 = m_factory.CreateSphere(GetTranslate(-2.5f, 1.0f, -0.5f), materials::cyan);
    AddObject(sphere0);

    std::pair<std::shared_ptr<Shape>, int> sphere1 = m_factory.CreateSphere(GetScale(1.0f, 2.0f, 1.0f), materials::prettyGreen);
    sphere1.first->Transform(GetTranslate(1.0f, 1.0f, -2.5f));
    AddObject(sphere1);

    // Cylinders
    std::pair<std::shared_ptr<Shape>, int> cylinder0 = m_factory.CreateClosedCylinder(GetTranslate(-0.5f, 0.1f, 1.0f), materials::darkRed);
    AddObject(cylinder0);

    // Cubes
    std::pair<std::shared_ptr<Shape>, int> cube0 = m_factory.CreateCube(GetRotate(M_PIf/4.0f, 1.0f, 1.0f, 1.0f), materials::yellow);
    cube0.first->Transform(GetTranslate(2.0f, 0.25f, 0.5f));
    AddObject(cube0);
}

void Scene::SetupLights()
{
    m_ambientLight = { 0.4, 0.4, 0.4 };

    m_lights.push_back(PointLight({ 3.0, 3.0, -3.0 }, { 0.6, 0.6, 0.6 }, 0.2f));
}

void Scene::SetupCamera()
{
    m_backgroundColor = { 0.0f, 0.0f, 0.0f };
    m_camera.reset(new sutil::Camera(
            { 4.0f, 4.5f, 7.0f }, // Position de l'oeil
            { 0.0f, 0.0f, 0.0f }, // Point au centre du regard
            { 0.0f, 1.0f, 0.0f }, // Vecteur haut
            60.0f, // Field of view
            static_cast<float>(m_cameraWidth) / static_cast<float>(m_cameraHeight) // aspect ratio
        )
    );
}

void Scene::AddObject(const std::pair<std::shared_ptr<Shape>, int>& object)
{
    m_shapes.push_back(object.first);
    m_nbObjects += object.second;
}

sutil::Matrix4x4 Scene::GetTranslate(float tx, float ty, float tz) const
{
    return sutil::Matrix4x4::translate(make_float3(tx, ty, tz));
}

sutil::Matrix4x4 Scene::GetScale(float sx, float sy, float sz) const
{
    return sutil::Matrix4x4::scale(make_float3(sx, sy, sz));
}

sutil::Matrix4x4 Scene::GetRotate(float angleRad, float vx, float vy, float vz) const
{
    return sutil::Matrix4x4::rotate(angleRad, make_float3(vx, vy, vz));
}


} // namespace host
} // namespace engine
