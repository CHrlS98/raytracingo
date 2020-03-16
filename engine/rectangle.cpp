#include <rectangle.h>
#include <algorithm>

namespace engine
{
namespace host
{
namespace
{
const std::string INTERSECTION_PROGRAM = "__intersection__rectangle";
}

Rectangle::Rectangle(const glm::vec3& a, const glm::vec3& b, const glm::vec3& p0, const BasicMaterial& material)
    : IShape(p0, material, INTERSECTION_PROGRAM)
    , m_a(a)
    , m_b(b)
{
    BuildAabb();
}

void Rectangle::BuildAabb()
{
    if ((m_a.x > 0) == (m_b.x > 0)) // signes identiques
    {
        const float dxMax = std::abs(m_a.x) > std::abs(m_b.x) ? m_a.x : m_b.x;
        m_boundingBox.minX = std::min(m_worldPosition.x, m_worldPosition.x + dxMax);
        m_boundingBox.maxX = std::max(m_worldPosition.x, m_worldPosition.x + dxMax);
    }
    else
    {
        m_boundingBox.minX = std::min(m_worldPosition.x + m_a.x, m_worldPosition.x + m_b.x);
        m_boundingBox.maxX = std::max(m_worldPosition.x + m_a.x, m_worldPosition.x + m_b.x);
    }

    if ((m_a.y > 0) == (m_b.y > 0)) // signes identiques
    {
        const float dyMax = std::abs(m_a.y) > std::abs(m_b.y) ? m_a.y : m_b.y;
        m_boundingBox.minY = std::min(m_worldPosition.y, m_worldPosition.y + dyMax);
        m_boundingBox.maxY = std::max(m_worldPosition.y, m_worldPosition.y + dyMax);
    }
    else
    {
        m_boundingBox.minY = std::min(m_worldPosition.y + m_a.y, m_worldPosition.y + m_b.y);
        m_boundingBox.maxY = std::max(m_worldPosition.y + m_a.y, m_worldPosition.y + m_b.y);
    }

    if ((m_a.z > 0) == (m_b.z > 0)) // signes identiques
    {
        const float dzMax = std::abs(m_a.z) > std::abs(m_b.z) ? m_a.z : m_b.z;
        m_boundingBox.minZ = std::min(m_worldPosition.z, m_worldPosition.z + dzMax);
        m_boundingBox.maxZ = std::max(m_worldPosition.z, m_worldPosition.z + dzMax);
    }
    else
    {
        m_boundingBox.minZ = std::min(m_worldPosition.z + m_a.z, m_worldPosition.z + m_b.z);
        m_boundingBox.maxZ = std::max(m_worldPosition.z + m_a.z, m_worldPosition.z + m_b.z);
    }

    const float epsilon = 0.1f;
    
    m_boundingBox.maxX += epsilon;
    m_boundingBox.minX -= epsilon;
    m_boundingBox.maxY += epsilon;
    m_boundingBox.minY -= epsilon;
    m_boundingBox.maxZ += epsilon;
    m_boundingBox.minZ -= epsilon;
}

void Rectangle::CopyToDevice(device::HitGroupData& data) const
{
    data.geometry.rectangle.a = { m_a.x, m_a.y, m_a.z };
    data.geometry.rectangle.b = { m_b.x, m_b.y, m_b.z };
    data.geometry.rectangle.p0 = { m_worldPosition.x, m_worldPosition.y, m_worldPosition.z };

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

}// namespace host
}// namespace engine