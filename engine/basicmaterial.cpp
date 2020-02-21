#include <basicmaterial.h>

namespace engine 
{
namespace host 
{
BasicMaterial::BasicMaterial()
    : m_ka(0.0f)
    , m_kd(0.0f)
    , m_ks(0.0f)
    , m_kr(1.0f)
    , m_alpha(30.0f)
{
}

BasicMaterial::BasicMaterial(
    const glm::vec3& ka,
    const glm::vec3& kd,
    const glm::vec3& ks,
    const glm::vec3& kr,
    const float& alpha)
    : m_ka(ka)
    , m_kd(kd)
    , m_ks(ks)
    , m_kr(kr)
    , m_alpha(alpha)
{
}

BasicMaterial::BasicMaterial(const BasicMaterial& other)
    : m_ka(other.m_ka)
    , m_kd(other.m_kd)
    , m_ks(other.m_ks)
    , m_kr(other.m_kr)
    , m_alpha(other.m_alpha)
{
}
} // namespace host
} // namespace engine