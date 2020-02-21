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
    /// \param[in] ka Couleur de la composante ambiante
    /// \param[in] kd Couleur de la composante diffuse
    /// \param[in] ks Couleur de la composante speculaire
    /// \param[in] kr Couleur de la composante reflechie
    /// \param[in] alpha Coefficient de reflexion speculaire
    BasicMaterial(
        const glm::vec3& ka, 
        const glm::vec3& kd, 
        const glm::vec3& ks, 
        const glm::vec3& kr, 
        const float& alpha);

    /// Copy constructeur
    BasicMaterial(const BasicMaterial& material);

    /// Default destructeur
    virtual ~BasicMaterial() = default;

    /// Getters
    inline glm::vec3 GetKa() const { return m_ka; }
    inline glm::vec3 GetKd() const { return m_kd; }
    inline glm::vec3 GetKs() const { return m_ks; }
    inline glm::vec3 GetKr() const { return m_kr; }
    inline float GetAlpha() const { return m_alpha; }

private:
    /// Couleur de la composante ambiante
    glm::vec3 m_ka;
    /// Couleur de la composante diffuse
    glm::vec3 m_kd;
    /// Couleur de la composante speculaire
    glm::vec3 m_ks;
    /// Couleur de la composante reflechie
    glm::vec3 m_kr;
    /// Coefficient de reflexion speculaire
    float m_alpha;
};
} // namespace host
} // namespace engine