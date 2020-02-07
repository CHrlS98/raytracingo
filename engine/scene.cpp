#include <scene.h>

Scene::Scene()
    : m_shapes()
{
}

void Scene::AddShape(const Shape& shape)
{
    m_shapes.push_back(shape);
}