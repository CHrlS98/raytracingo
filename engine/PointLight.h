#pragma once

#include <glm/vec3.hpp>

namespace engine
{
namespace host
{
class PointLight
{
public: 
    /// Default constructeur
    PointLight();

    /// Constructeur
    /// \param[in] position La position de la lumiere
    /// \param[in] color La couleur de l'eclairage
    PointLight(const glm::vec3& position, const glm::vec3& color, const float falloff);

    /// Default destructeur
    ~PointLight() = default;

    /// Getter pour la propriete m_position
    /// \return La position de l'objet
    inline glm::vec3 GetPosition() const { return m_position; };

    /// Getter pour la propriete m_color
    /// \return La couleur de la lumiere
    inline glm::vec3 GetColor() const { return m_color; };

    /// Getter pour la propriete m_falloff
    /// \return La constante de decroissance de la lumiere
    inline float GetFalloff() const { return m_falloff; };

private:
    /// Position de la lumiere
    glm::vec3 m_position;

    /// Couleur de la lumiere
    glm::vec3 m_color;

    /// Constante de decroissante de la lumiere. Devrait etre < 1
    float m_falloff;
};
} // namespace host
} // namespace engine