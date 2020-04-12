#pragma once

#include <glm/vec3.hpp>

namespace engine
{
namespace host
{
/// Classe definissant les proprietes d'un materiau
class BasicMaterial
{
public:
    /// Constructeur
    BasicMaterial();
    
    /// Constructeur
    /// \param[in] kd La couleur diffuse du materiau
    /// \param[in] kr Proportion de couleur reflechie
    /// \param[in] le Lumiere emise du materiau
    /// \param[in] specularity Coefficient de specularite
    /// \remark Un coefficient de specularite eleve cree des reflexions parfaites
    BasicMaterial(
        const glm::vec3& kd,
        const glm::vec3& kr,
        const glm::vec3& le,
        const float& specularity);

    /// Copy constructeur
    BasicMaterial(const BasicMaterial& material);

    /// Destructeur par defaut
    virtual ~BasicMaterial() = default;

    /// Accesseur pour la propriete m_kd
    /// \return Couleur diffuse
    inline glm::vec3 GetKd() const { return m_kd; }

    /// Accesseur pour la propriete m_kr
    /// \return Proportion de couleur reflechie
    inline glm::vec3 GetKr() const { return m_kr; }

    /// Accesseur pour la propriete m_Le
    /// \return Lumiere emise
    inline glm::vec3 GetLe() const { return m_Le; }

    /// Accesseur pour la propriete m_specularity
    /// \return Coefficient de specularite du materiau
    inline float GetSpecularity() const { return m_specularity; }

private:
    /// Couleur du materiau
    glm::vec3 m_kd;
    /// Proportion de lumiere reflechie
    glm::vec3 m_kr;
    /// Emission
    glm::vec3 m_Le;
    /// Coefficient de specularite
    float m_specularity;
};
} // namespace host
} // namespace engine