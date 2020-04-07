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
    , m_ambientLight()
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

void Scene::CreateSadJailCell()
{
    /*
    // spheres
    std::pair<std::shared_ptr<Shape>, int> cylinder = m_factory.CreateClosedCylinder(GetTranslate(0.0f, -3.0f, 0.0f), materials::darkRed);
    AddObject(cylinder);
    //std::pair<std::shared_ptr<Shape>, int> sphere1 = m_factory.CreateSphere(GetTranslate(-2.1f, -0.5f, 0.0f), materials::prettyGreen);
    //AddObject(sphere1);
    //std::pair<std::shared_ptr<Shape>, int> sphere2 = m_factory.CreateSphere(GetTranslate(2.1f, -0.5f, 0.0f), materials::metallicGold);
    //AddObject(sphere2);

    std::pair<std::shared_ptr<Shape>, int> box = m_factory.CreateCube(GetScale(2.0f, 2.0f, 2.0f), materials::white);
    box.first->Transform(GetRotate(M_PIf / 3.0f, 0.0f, 1.0f, 0.0f));
    box.first->Transform(GetTranslate(-2.f, -4.0f, 3.0f));
    AddObject(box); // mur fond

    //std::pair<std::shared_ptr<Shape>, int> mirror = m_factory.CreateDisk(GetScale(3.0f, 1.0f, 1.8f), materials::blackMirror);
    //mirror.first->Transform(GetRotate(-M_PIf / 2.0f, 0.0f, 0.0f, 1.0f));
    //mirror.first->Transform(GetTranslate(4.99f, 0.0f, 1.0f));
    //AddObject(mirror); // mur fond

    std::pair<std::shared_ptr<Shape>, int> plane0 = m_factory.CreateRectangle(GetScale(10.0f, 1.0f, 10.0f), materials::blue);
    plane0.first->Transform(GetTranslate(0.0f, -5.0f, 0.0f));
    AddObject(plane0); // sol

    std::pair<std::shared_ptr<Shape>, int> plane1 = m_factory.CreateRectangle(GetScale(10.0f, 1.0f, 10.0f), materials::white);
    plane1.first->Transform(GetTranslate(0.0f, 5.0f, 0.0f));
    AddObject(plane1); // plafond

    std::pair<std::shared_ptr<Shape>, int> plane2 = m_factory.CreateRectangle(GetScale(10.0f, 1.0f, 10.0f), materials::cream);
    plane2.first->Transform(GetRotate(M_PIf / 2.0f, 0.0f, 0.0f, 1.0f));
    plane2.first->Transform(GetTranslate(5.0f, 0.0f, 0.0f));
    AddObject(plane2); // mur droit

    std::pair<std::shared_ptr<Shape>, int> plane3 = m_factory.CreateRectangle(GetScale(10.0f, 1.0f, 10.0f), materials::cream);
    plane3.first->Transform(GetRotate(-M_PIf / 2.0f, 0.0f, 0.0f, 1.0f));
    plane3.first->Transform(GetTranslate(-5.0f, 0.0f, 0.0f));
    AddObject(plane3); // mur gauche

    std::pair<std::shared_ptr<Shape>, int> cube0 = m_factory.CreateCube(GetScale(3.0f, 0.5f, 10.0f), materials::white);
    cube0.first->Transform(GetRotate(M_PIf / 2.0f, 1.0f, 0.0f, 0.0f));
    cube0.first->Transform(GetTranslate(-3.5f, 0.0f, -5.25f));
    AddObject(cube0); // mur fond
    std::pair<std::shared_ptr<Shape>, int> cube1 = m_factory.CreateCube(GetScale(3.0f, 0.5f, 10.0f), materials::white);
    cube1.first->Transform(GetRotate(M_PIf / 2.0f, 1.0f, 0.0f, 0.0f));
    cube1.first->Transform(GetTranslate(3.5f, 0.0f, -5.25f));
    AddObject(cube1); // mur fond
    std::pair<std::shared_ptr<Shape>, int> cube2 = m_factory.CreateCube(GetScale(4.0f, 0.5f, 3.0f), materials::white);
    cube2.first->Transform(GetRotate(M_PIf / 2.0f, 1.0f, 0.0f, 0.0f));
    cube2.first->Transform(GetTranslate(0.f, 3.5f, -5.25f));
    AddObject(cube2); // mur fond
    std::pair<std::shared_ptr<Shape>, int> cube3 = m_factory.CreateCube(GetScale(4.0f, 0.5f, 4.0f), materials::white);
    cube3.first->Transform(GetRotate(M_PIf / 2.0f, 1.0f, 0.0f, 0.0f));
    cube3.first->Transform(GetTranslate(0.f, -3.0f, -5.25f));
    AddObject(cube3); // mur fond

    // barreaux
    std::pair<std::shared_ptr<Shape>, int> cylinder0 = m_factory.CreateClosedCylinder(GetScale(0.18f, 1.5f, 0.18f), materials::grey);
    cylinder0.first->Transform(GetTranslate(0.f, 0.5f, -5.25f));
    AddObject(cylinder0);
    std::pair<std::shared_ptr<Shape>, int> cylinder1 = m_factory.CreateClosedCylinder(GetScale(0.18f, 1.5f, 0.18f), materials::grey);
    cylinder1.first->Transform(GetTranslate(1.f, 0.5f, -5.25f));
    AddObject(cylinder1);
    std::pair<std::shared_ptr<Shape>, int> cylinder2 = m_factory.CreateClosedCylinder(GetScale(0.18f, 1.5f, 0.18f), materials::grey);
    cylinder2.first->Transform(GetTranslate(-1.f, 0.5f, -5.25f));
    AddObject(cylinder2);
    */
}

void Scene::SetupObjects()
{
    switch (m_sceneModel)
    {
    case SceneModel::CORNELL:
        CreateCornellBox();
        break;
    case SceneModel::JAIL:
        CreateSadJailCell();
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
            { 0.0f, 0.0f, 14.0f }, // Position de l'oeil
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
