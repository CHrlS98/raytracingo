#include <engine/light.h>
#include <algorithm>

namespace engine
{
namespace host
{

SurfaceLight::SurfaceLight(PRIMITIVE_TYPE type, const sutil::Matrix4x4& modelMatrix, const glm::vec3& color, const float falloff)
    : m_type(type)
    , m_modelMatrix(modelMatrix)
    , m_color(color)
    , m_falloff(falloff)
{
    float4 origin = { -0.5f, 0.0f, 0.5f, 1.0f };
    const float4 corner = modelMatrix * origin;
    m_corner = glm::vec3(corner.x, corner.y, corner.z);

    float4 xAxis = { 1.0f, 0.0f, 0.0f, 0.0f };
    const float4 v1 = modelMatrix * xAxis;
    m_v1 = glm::vec3(v1.x, v1.y, v1.z);

    float4 zAxis = { 0.0f, 0.0f, -1.0f, 0.0f };
    const float4 v2 = modelMatrix * zAxis;
    m_v2 = glm::vec3(v2.x, v2.y, v2.z);

    m_normal = glm::normalize(glm::cross(m_v1, m_v2));

    switch (m_type)
    {
    case engine::host::PRIMITIVE_TYPE::RECTANGLE:
        m_intersectionProgram = "__intersection__rectangle";
        break;
    default:
        break;
    }
    BuildAabb();
}

void SurfaceLight::BuildAabb()
{
    const float eps = 0.00001f;
    const glm::vec3 maxCorner = m_corner + m_v1 + m_v2;
    m_aabb.maxX = std::max(m_corner.x, maxCorner.x) + eps;
    m_aabb.minX = std::min(m_corner.x, maxCorner.x) - eps;
    m_aabb.maxY = std::max(m_corner.y, maxCorner.y) + eps;
    m_aabb.minY = std::min(m_corner.y, maxCorner.y) - eps;
    m_aabb.maxZ = std::max(m_corner.z, maxCorner.z) + eps;
    m_aabb.minZ = std::min(m_corner.z, maxCorner.z) - eps;
}

void SurfaceLight::CopyToDevice(device::HitGroupData& data) const
{
    data.modelMatrix = m_modelMatrix;
}

} // namespace host
} // namespace engine