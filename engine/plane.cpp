#include <plane.h>

namespace engine
{

Plane::Plane(const glm::vec3& position, const glm::vec3& normal)
    : IShape(ShapeType::PlaneType, position, "__intersection__plane")
    , m_normal(normal)
{
    BuildAabb();
}

void Plane::BuildAabb()
{
    m_boundingBox.maxX = 1e6;
    m_boundingBox.minX = -1e6;

    m_boundingBox.maxY = 1e6;
    m_boundingBox.minY = -1e6;

    m_boundingBox.maxZ = 1e6;
    m_boundingBox.minZ = -1e6;
}

} // namespace engine