#include <PointLight.h>

namespace engine
{
namespace host
{
PointLight::PointLight()
    : m_position()
    , m_color()
    , m_falloff(0.0f)
{
}

PointLight::PointLight(const glm::vec3& position, const glm::vec3& color, const float falloff)
    : m_position(position)
    , m_color(color)
    , m_falloff(falloff)
{
}
} // namespace host
} // namespace engine