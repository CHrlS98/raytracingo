#include <basicmaterial.h>

namespace engine 
{
namespace host 
{
BasicMaterial::BasicMaterial()
    : m_kd(0.0f)
    , m_specularity(0.0f)
    , m_Le(0.0f)
    , m_kr(0.0f)
{
}

BasicMaterial::BasicMaterial(
    const glm::vec3& kd,
    const glm::vec3& kr,
    const glm::vec3& le,
    const float& specularity)
    : m_kd(kd)
    , m_kr(kr)
    , m_specularity(specularity)
    , m_Le(le)
{
}

BasicMaterial::BasicMaterial(const BasicMaterial& other)
    : m_kd(other.m_kd)
    , m_specularity(other.m_specularity)
    , m_Le(other.m_Le)
    , m_kr(other.m_kr)
{
}
} // namespace host
} // namespace engine