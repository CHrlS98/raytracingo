#include <sphere.h>

Sphere::Sphere()
{
    Sphere(glm::vec3(0.0f), 0.0f);
}

Sphere::Sphere(const glm::vec3& worldPosition, const float radius)
    : m_radius(radius)
{
    m_type = ShapeType::SphereType;
    m_boundingBox = { -m_radius, -m_radius, -m_radius, m_radius, m_radius, m_radius };
    m_worldPosition = worldPosition;
}