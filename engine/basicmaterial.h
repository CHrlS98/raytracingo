#pragma once

#include <glm/vec3.hpp>

namespace engine
{
namespace host
{
class BasicMaterial
{
public:
    /// Constructeur pour un objet noir 100% mirroir
    BasicMaterial();

    /// Constructeur
    /// \param[in] kd Couleur de la composante diffuse
    /// \param[in] roughness Coefficient de reflexion speculaire
    BasicMaterial(
        const glm::vec3& kd, 
        const float& roughness);

    /// Copy constructeur
    BasicMaterial(const BasicMaterial& material);

    /// Default destructeur
    virtual ~BasicMaterial() = default;

    /// Getters
    inline glm::vec3 GetKd() const { return m_kd; }
    inline float GetRoughness() const { return m_roughness; }

private:
    /// Couleur du materiau
    glm::vec3 m_kd;
    /// Coefficient de rugosite
    float m_roughness;
};
} // namespace host
} // namespace engine