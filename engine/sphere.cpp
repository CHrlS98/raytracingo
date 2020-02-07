#include <sphere.h>

Sphere::Sphere()
{
    Sphere(glm::vec3(0.0f));
}

Sphere::Sphere(const glm::vec3& worldPosition)
{
    m_worldPosition = worldPosition;
}