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
}
} // namespace host
} // namespace engine