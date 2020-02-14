#include <PointLight.h>

namespace engine
{

PointLight::PointLight()
    : m_position()
    , m_color()
{
}

PointLight::PointLight(const glm::vec3& position, const glm::vec3& color)
    : m_position(position)
    , m_color(color)
{
}

} // namespace engine