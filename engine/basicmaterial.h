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
    BasicMaterial(
        const glm::vec3& kd,
        const glm::vec3& kr,
        const glm::vec3& le,
        const float& specularity);

    /// Copy constructeur
    BasicMaterial(const BasicMaterial& material);

    /// Default destructeur
    virtual ~BasicMaterial() = default;

    /// Getters
    inline glm::vec3 GetKd() const { return m_kd; }
    inline glm::vec3 GetKr() const { return m_kr; }
    inline glm::vec3 GetLe() const { return m_Le; }
    inline float GetSpecularity() const { return m_specularity; }

private:
    /// Couleur du materiau
    glm::vec3 m_kd;
    /// Proportion de lumiere reflechie
    glm::vec3 m_kr;
    /// Emission
    glm::vec3 m_Le;
    /// Coefficient de specularite
    /// une specularite de 1000 represente un materiel tres reflechissant
    float m_specularity;
};
} // namespace host
} // namespace engine