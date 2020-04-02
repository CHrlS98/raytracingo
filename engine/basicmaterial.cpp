#include <basicmaterial.h>

namespace engine 
{
namespace host 
{
BasicMaterial::BasicMaterial()
    : m_kd(0.0f)
    , m_roughness(0.0f)
{
}

BasicMaterial::BasicMaterial(
    const glm::vec3& kd,
    const float& roughness)
    : m_kd(kd)
    , m_roughness(roughness)
{
}

BasicMaterial::BasicMaterial(const BasicMaterial& other)
    : m_kd(other.m_kd)
    , m_roughness(other.m_roughness)
{
}
} // namespace host
} // namespace engine