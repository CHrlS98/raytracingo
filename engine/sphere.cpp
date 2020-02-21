#include <sphere.h>

namespace engine
{
namespace host
{
namespace
{
const std::string INTERSECTION_PROGRAM = "__intersection__sphere";
}

Sphere::Sphere()
    :IShape(ShapeType::SphereType, glm::vec3(0.0f), BasicMaterial(), INTERSECTION_PROGRAM)
    , m_radius(1.0f)
{
    BuildAabb();
}

Sphere::Sphere(const glm::vec3& worldPosition, const float radius, const BasicMaterial& material)
    : IShape(ShapeType::SphereType, worldPosition, material, INTERSECTION_PROGRAM)
    ,m_radius(radius)
{
    BuildAabb();
}

void Sphere::BuildAabb()
{
    m_boundingBox.minX = m_worldPosition.x - m_radius;
    m_boundingBox.minY = m_worldPosition.y - m_radius;
    m_boundingBox.minZ = m_worldPosition.z - m_radius;
    m_boundingBox.maxX = m_worldPosition.x + m_radius;
    m_boundingBox.maxY = m_worldPosition.y + m_radius;
    m_boundingBox.maxZ = m_worldPosition.z + m_radius;
}

void Sphere::CopyToDevice(device::HitGroupData& data) const
{
    // Copie les caracteristiques geometriques de l'objet
    data.geometry.sphere.position = { m_worldPosition.x, m_worldPosition.y, m_worldPosition.z };
    data.geometry.sphere.radius = m_radius;

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
