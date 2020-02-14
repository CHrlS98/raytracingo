#include <sphere.h>

namespace engine
{

Sphere::Sphere()
{
    Sphere(glm::vec3(0.0f), 0.0f);
}

Sphere::Sphere(const glm::vec3& worldPosition, const float radius)
    : IShape(ShapeType::SphereType, worldPosition, "__intersection__sphere")
    ,m_radius(radius)
{
    m_boundingBox.minX = worldPosition.x - radius;
    m_boundingBox.minY = worldPosition.y - radius;
    m_boundingBox.minZ = worldPosition.z - radius;
    m_boundingBox.maxX = worldPosition.x + radius;
    m_boundingBox.maxY = worldPosition.y + radius;
    m_boundingBox.maxZ = worldPosition.z + radius;
}

} // namespace engine
