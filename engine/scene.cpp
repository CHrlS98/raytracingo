#include <scene.h>
#include <sphere.h>

Scene::Scene()
    : m_shapes()
{
    Sphere sphere0 = Sphere(glm::vec3(0.0f), 1.0f);
    m_shapes.push_back(sphere0);
}