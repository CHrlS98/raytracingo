#include <scene.h>

#include <engine/light.h>
#include <BasicMaterial.h>
#include <materials.h>
#include <sutil/Matrix.h>

namespace engine
{
namespace host
{

Scene::Scene(SceneModel sceneModel, const unsigned int& camWidth, const unsigned int& camHeight)
    : m_shapes()
    , m_cameraWidth(camWidth)
    , m_cameraHeight(camHeight)
    , m_backgroundColor()
    , m_camera(nullptr)
    , m_factory()
    , m_nbObjects(0)
    , m_sceneModel(sceneModel)
{
    SetupObjects();
    SetupCamera();
}

void Scene::CreateFunPlate()
{
    // custom shape
    std::pair<std::shared_ptr<Shape>, int> triangle0;
    {
        std::vector<Primitive> primitives;
        primitives.push_back(Primitive(PRIMITIVE_TYPE::CYLINDER,
            GetTranslate(cos(M_PIf / 3.0f), sin(M_PIf / 3.0f), 0.0f)
            * GetRotate(M_PIf / 6.0f, 0.0f, 0.0f, 1.0f)
            * GetScale(0.4f, 1.0f, 0.4f),
            materials::plateMetallicGold)
        );
        primitives.push_back(Primitive(PRIMITIVE_TYPE::CYLINDER,
            GetTranslate(-cos(M_PIf / 3.0f), sin(M_PIf / 3.0f), 0.0f)
            * GetRotate(-M_PIf / 6.0f, 0.0f, 0.0f, 1.0f)
            * GetScale(0.4f, 1.0f, 0.4f),
            materials::plateMetallicGold)
        );
        primitives.push_back(Primitive(PRIMITIVE_TYPE::CYLINDER,
            GetRotate(M_PIf / 2.0f, 0.0f, 0.0f, 1.0f)
            * GetScale(0.4f, 1.0f, 0.4f),
            materials::plateMetallicGold)
        );
        primitives.push_back(Primitive(PRIMITIVE_TYPE::SPHERE,
            GetTranslate(-1.0f, 0.0f, 0.0f)
            * GetScale(0.4f, 0.4f, 0.4f),
            materials::plateMetallicGold)
        );
        primitives.push_back(Primitive(PRIMITIVE_TYPE::SPHERE,
            GetTranslate(1.0f, 0.0f, 0.0f)
            * GetScale(0.4f, 0.4f, 0.4f),
            materials::plateMetallicGold)
        );
        primitives.push_back(Primitive(PRIMITIVE_TYPE::SPHERE,
            GetTranslate(0.0f, sqrtf(3.0f), 0.0f)
            * GetScale(0.4f, 0.4f, 0.4f),
            materials::plateMetallicGold)
        );
        triangle0 = m_factory.CreateCustom(primitives, sutil::Matrix4x4::identity());
    }
    triangle0.first->Transform(GetRotate(-M_PIf / 2.0f, 1.0f, 0.0f, 0.0f));
    triangle0.first->Transform(GetTranslate(0.0f, 2.5f, 0.5f));
    AddObject(triangle0);
    
    // disks
    std::pair<std::shared_ptr<Shape>, int> disk0 = m_factory.CreateDisk(GetScale(4.0f, 1.0f, 4.0f), materials::platePurple);
    disk0.first->Transform(GetTranslate(0.0f, -1.0f, 0.0f));
    AddObject(disk0);

    // spheres
    std::pair<std::shared_ptr<Shape>, int> sphere0 = m_factory.CreateSphere(GetTranslate(-2.5f, 1.0f, -0.5f), materials::plateCyan);
    AddObject(sphere0);
    
    std::pair<std::shared_ptr<Shape>, int> sphere1 = m_factory.CreateSphere(GetScale(1.0f, 2.0f, 1.0f), materials::platePrettyGreen);
    sphere1.first->Transform(GetTranslate(1.0f, 1.0f, -2.5f));
    AddObject(sphere1);
    
    // Cylinders
    std::pair<std::shared_ptr<Shape>, int> cylinder0 = m_factory.CreateClosedCylinder(GetTranslate(-0.5f, 0.1f, 1.0f), materials::plateDarkRed);
    AddObject(cylinder0);
    
    // Cubes
    std::pair<std::shared_ptr<Shape>, int> cube0 = m_factory.CreateCube(GetRotate(M_PIf / 4.0f, 1.0f, 1.0f, 1.0f), materials::plateYellow);
    cube0.first->Transform(GetTranslate(2.0f, 0.25f, 0.5f));
    AddObject(cube0);

    // Setup des lumieres
    std::pair<std::shared_ptr<Shape>, int> lightObj = m_factory.CreateRectangle(
        GetTranslate(0.0f, 6.0f, 0.0f) * GetRotate(M_PIf, 1.0f, 0.0f, 0.0f) * GetScale(4.0f, 1.0f, 4.0f),
        materials::cornellLight
    );

    Primitive primitive = lightObj.first->GetPrimitives()[0];
    SurfaceLight light = SurfaceLight(primitive.GetType(), primitive.GetModelMatrix(), primitive.GetMaterial().GetLe(), 0.0f);

    AddObject(lightObj);
    m_surfaceLights.push_back(light);
}

void Scene::CreateCornellBox()
{
    // Setup des objets
    std::pair<std::shared_ptr<Shape>, int> back = m_factory.CreateRectangle(
        GetTranslate(0.0f, 0.0f, -4.0f) * GetRotate(M_PIf / 2.0f, 1.0f, 0.0f, 0.0f) * GetScale(8.0f, 1.0f, 8.0f),
        materials::cornellWhite
    );
    std::pair<std::shared_ptr<Shape>, int> front = m_factory.CreateRectangle(
        GetTranslate(0.0f, 0.0f, 4.0f) * GetRotate(-M_PIf / 2.0f, 1.0f, 0.0f, 0.0f) * GetScale(8.0f, 1.0f, 8.0f),
        materials::cornellWhite
    );
    std::pair<std::shared_ptr<Shape>, int> left = m_factory.CreateRectangle(
        GetTranslate(-4.0f, 0.0f, 0.0f) * GetRotate(-M_PIf / 2.0f, 0.0f, 0.0f, 1.0f) * GetScale(8.0f, 1.0f, 8.0f),
        materials::cornellRed
    );
    std::pair<std::shared_ptr<Shape>, int> right = m_factory.CreateRectangle(
        GetTranslate(4.0f, 0.0f, 0.0f) * GetRotate(M_PIf / 2.0f, 0.0f, 0.0f, 1.0f) * GetScale(8.0f, 1.0f, 8.0f),
        materials::cornellBlue
    );
    std::pair<std::shared_ptr<Shape>, int> top = m_factory.CreateRectangle(
        GetTranslate(0.0f, 4.0f, 0.0f) * GetRotate(M_PIf, 0.0f, 0.0f, 1.0f) * GetScale(8.0f, 1.0f, 8.0f),
        materials::cornellWhite
    );
    std::pair<std::shared_ptr<Shape>, int> bottom = m_factory.CreateRectangle(
        GetTranslate(0.0f, -4.0f, 0.0f) * GetScale(8.0f, 1.0f, 8.0f),
        materials::cornellWhite
    );
    std::pair<std::shared_ptr<Shape>, int> box1 = m_factory.CreateCube(
        GetTranslate(1.3f, -3.0f, 1.3f) * GetRotate(-M_PIf / 6.0f, 0.0f, 1.0f, 0.0f) * GetScale(2.0f, 2.0f, 2.0f),
        materials::cornellWhite
    );
    std::pair<std::shared_ptr<Shape>, int> box2 = m_factory.CreateCube(
        GetTranslate(-1.3f, -2.0f, -1.3f) * GetRotate(M_PIf / 8.0f, 0.0f, 1.0f, 0.0f) * GetScale(2.0f, 4.0f, 2.0f),
        materials::cornellWhite
    );

    AddObject(back);
    AddObject(front);
    AddObject(left);
    AddObject(right);
    AddObject(top);
    AddObject(bottom);

    AddObject(box1);
    AddObject(box2);

    // Setup des lumieres
    std::pair<std::shared_ptr<Shape>, int> lightObj = m_factory.CreateRectangle(
        GetTranslate(0.0f, 3.95f, 0.0f) * GetRotate(M_PIf, 1.0f, 0.0f, 0.0f) * GetScale(4.0f, 1.0f, 4.0f),
        materials::cornellLight
    );

    Primitive primitive = lightObj.first->GetPrimitives()[0];
    SurfaceLight light = SurfaceLight(primitive.GetType(), primitive.GetModelMatrix(), { 1.0f, 1.0f, 1.0f }, 0.0f);

    AddObject(lightObj);
    m_surfaceLights.push_back(light);
}

void Scene::CreateSlide()
{
    // Plateau
    std::pair<std::shared_ptr<Shape>, int> floor = m_factory.CreateRectangle(GetTranslate(0.0f, -1.0f, 0.0f) * GetScale(150.0f, 1.0f, 8.0f), materials::grey);
    AddObject(floor);

    std::pair<std::shared_ptr<Shape>, int> sphere = m_factory.CreateSphere(GetTranslate(12.0f, 0.0f, 0.0f), materials::plateCyan);
    AddObject(sphere);

    std::pair<std::shared_ptr<Shape>, int> cube = m_factory.CreateCube(GetTranslate(6.0f, 0.0f, 0.0f) * GetScale(2.0f, 2.0f, 2.0f), materials::plateCyan);
    AddObject(cube);

    std::pair<std::shared_ptr<Shape>, int> cylinder = m_factory.CreateClosedCylinder(GetTranslate(0.0f, 0.0f, 0.0f), materials::plateCyan);
    AddObject(cylinder);

    std::pair<std::shared_ptr<Shape>, int> disk = m_factory.CreateDisk(GetTranslate(-6.0f, 0.0f, 0.0f), materials::plateCyan);
    AddObject(disk);

    std::pair<std::shared_ptr<Shape>, int> square = m_factory.CreateRectangle(GetTranslate(-12.0f, 0.0f, 0.0f) * GetScale(2.0f, 1.0f, 2.0f), materials::plateCyan);
    AddObject(square);

    std::pair<std::shared_ptr<Shape>, int> sphereTransf = m_factory.CreateSphere(GetTranslate(-20.0f, 2.0f, 0.0f) * GetRotate(M_PIf / 2.0f, 1.0f, 1.0f, .0f) * GetScale(3.0f, 2.0f, 2.0f), materials::cornellBlue);
    AddObject(sphereTransf);

    std::pair<std::shared_ptr<Shape>, int> cubeTransf = m_factory.CreateCube(GetTranslate(-30.0f, 2.0f, 0.0f) * GetRotate(M_PIf/4.0f, 0.f, 1.f, 1.f) * GetScale(2.0f, 2.0f, 2.0f), materials::platePrettyGreen);
    AddObject(cubeTransf);

    std::pair<std::shared_ptr<Shape>, int> openCylinder = m_factory.CreateOpenCylinder(GetTranslate(-40.0f, 0.3f, 0.0f)* GetRotate(M_PIf / 2.0f, 1.f, 0.f, 0.f)* GetScale(1.0f, 2.0f, 1.0f), materials::plateMetallicGold);
    AddObject(openCylinder);

    std::pair<std::shared_ptr<Shape>, int> triangle0;
    {
        std::vector<Primitive> primitives;
        primitives.push_back(Primitive(PRIMITIVE_TYPE::CYLINDER,
            GetTranslate(cos(M_PIf / 3.0f), sin(M_PIf / 3.0f), 0.0f)
            * GetRotate(M_PIf / 6.0f, 0.0f, 0.0f, 1.0f)
            * GetScale(0.4f, 1.0f, 0.4f),
            materials::cornellRed)
        );
        primitives.push_back(Primitive(PRIMITIVE_TYPE::CYLINDER,
            GetTranslate(-cos(M_PIf / 3.0f), sin(M_PIf / 3.0f), 0.0f)
            * GetRotate(-M_PIf / 6.0f, 0.0f, 0.0f, 1.0f)
            * GetScale(0.4f, 1.0f, 0.4f),
            materials::cornellRed)
        );
        primitives.push_back(Primitive(PRIMITIVE_TYPE::CYLINDER,
            GetRotate(M_PIf / 2.0f, 0.0f, 0.0f, 1.0f)
            * GetScale(0.4f, 1.0f, 0.4f),
            materials::cornellRed)
        );
        primitives.push_back(Primitive(PRIMITIVE_TYPE::SPHERE,
            GetTranslate(-1.0f, 0.0f, 0.0f)
            * GetScale(0.4f, 0.4f, 0.4f),
            materials::cornellRed)
        );
        primitives.push_back(Primitive(PRIMITIVE_TYPE::SPHERE,
            GetTranslate(1.0f, 0.0f, 0.0f)
            * GetScale(0.4f, 0.4f, 0.4f),
            materials::cornellRed)
        );
        primitives.push_back(Primitive(PRIMITIVE_TYPE::SPHERE,
            GetTranslate(0.0f, sqrtf(3.0f), 0.0f)
            * GetScale(0.4f, 0.4f, 0.4f),
            materials::cornellRed)
        );
        triangle0 = m_factory.CreateCustom(primitives, sutil::Matrix4x4::identity());
    }
    triangle0.first->Transform(GetScale(1.5f, 1.5f, 1.5f));
    triangle0.first->Transform(GetRotate(-M_PIf / 6.0f, 1.0f, 0.0f, 0.0f));
    triangle0.first->Transform(GetTranslate(-50.0f, 1.2f, 0.7f));
    AddObject(triangle0);

    std::pair<std::shared_ptr<Shape>, int> cub3 = m_factory.CreateCube(GetTranslate(-60.0f, 2.5f, 0.0f) * GetRotate(M_PIf / 4.0f, 0.f, 1.f, 1.f) * GetScale(1.0f, 6.0f, 0.4f), materials::cream);
    AddObject(cub3);


    // Setup des lumieres
    std::pair<std::shared_ptr<Shape>, int> lightObj = m_factory.CreateRectangle(
        GetTranslate(0.0f, 100.0f, 10.0f) * GetRotate(M_PIf, 1.0f, 0.0f, 0.0f) * GetScale(0.001f, 1.0f, 0.001f),
        materials::cornellLight
    );

    Primitive primitive = lightObj.first->GetPrimitives()[0];
    SurfaceLight light = SurfaceLight(primitive.GetType(), primitive.GetModelMatrix(), { 1.0f, 1.0f, 1.0f }, 0.0f);

    AddObject(lightObj);
    m_surfaceLights.push_back(light);
}

void Scene::SetupObjects()
{
    switch (m_sceneModel)
    {
    case SceneModel::CORNELL:
        CreateCornellBox();
        break;
    case SceneModel::SLIDE:
        CreateSlide();
        break;
    case SceneModel::PLATE:
        CreateFunPlate();
        break;
    }
}

void Scene::SetupCamera()
{
    m_backgroundColor = { 0.0, 0.0, 0.0 };
    m_camera.reset(new sutil::Camera(
            { 0.0f, 3.0f, 14.0f }, // Position de l'oeil
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
