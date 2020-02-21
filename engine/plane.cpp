#include <plane.h>

namespace engine
{
namespace host 
{
namespace
{
const std::string INTERSECTION_PROGRAM = "__intersection__plane";
}

Plane::Plane(const glm::vec3& position, const glm::vec3& normal, const BasicMaterial& material)
    : IShape(ShapeType::PlaneType, position, material, INTERSECTION_PROGRAM)
    , m_normal(normal)
{
    BuildAabb();
}

void Plane::BuildAabb()
{
    const float floatMax = std::numeric_limits<float>::max();

    m_boundingBox.maxX = floatMax;
    m_boundingBox.minX = -floatMax;

    m_boundingBox.maxY = floatMax;
    m_boundingBox.minY = -floatMax;

    m_boundingBox.maxZ = floatMax;
    m_boundingBox.minZ = -floatMax;
}

void Plane::CopyToDevice(device::HitGroupData& data) const
{
    data.geometry.plane.normal = { m_normal.x, m_normal.y, m_normal.z };
    data.geometry.plane.position = { m_worldPosition.x, m_worldPosition.y, m_worldPosition.z };

    const glm::vec3& ka = m_material.GetKa();
    const glm::vec3& kd = m_material.GetKd();
    const glm::vec3& ks = m_material.GetKs();
    const glm::vec3& kr = m_material.GetKr();
    data.material.basicMaterial.ka = { ka.r, ka.g, ka.b };
    data.material.basicMaterial.kd = { kd.r, kd.g, kd.b };
    data.material.basicMaterial.ks = { ks.r, ks.g, ks.b };
    data.material.basicMaterial.kr = { kr.r, kr.g, kr.b };
    data.material.basicMaterial.alpha = m_material.GetAlpha();
}

} // namespace host
} // namespace engine