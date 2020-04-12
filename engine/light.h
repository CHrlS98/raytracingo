#pragma once

#include <glm/glm.hpp> 
#include <primitive.h> 
#include<shape.h>
#include<memory>

namespace engine
{
namespace host
{
/// Implementation d'une lumiere de surface
class SurfaceLight
{
public:
    /// Constructeur
    /// \param[in] type Le type de primitive qui compose la surface
    /// \param[in] modelMatrix La matrice de transformation de la surface
    /// \param[in] color La couleur de l'eclairage
    /// \param[in] falloff Constante de decroissante de la lumiere. Devrait etre < 1
    SurfaceLight(PRIMITIVE_TYPE type, const sutil::Matrix4x4& modelMatrix, const glm::vec3& color, const float falloff);

    /// Default destructeur
    ~SurfaceLight() = default;

    /// Getter pour la propriete m_position
    /// \return La position de l'objet
    inline glm::vec3 GetCorner() const { return m_corner; };
    /// Getter pour la propriete m_v1
    /// \return Le vecteur v1 de la surface de la lumiere
    inline glm::vec3 GetV1() const { return m_v1; };
    /// Getter pour la propriete m_v2
    /// \return Le vecteur v2 de la surface de la lumiere
    inline glm::vec3 GetV2() const { return m_v2; };
    /// Getter pour la propriete m_normal
    /// \return Le vecteur normale de la surface de la lumiere
    inline glm::vec3 GetNormal() const { return m_normal; };
    /// Getter pour la propriete m_color
    /// \return La couleur de la lumiere
    inline glm::vec3 GetColor() const { return m_color; };
    /// Getter pour la propriete m_falloff
    /// \return La constante de decroissance de la lumiere
    inline float GetFalloff() const { return m_falloff; };

private:
    /// Type de primitive de la surface
    PRIMITIVE_TYPE m_type;
    /// Matrice de transformation de la surface
    sutil::Matrix4x4 m_modelMatrix;
    /// Position du coin de la lumiere
    glm::vec3 m_corner;
    /// Premier vecteur de direction
    glm::vec3 m_v1;
    /// Deuxieme vecteur de direction
    glm::vec3 m_v2;
    /// Normale de la surface
    glm::vec3 m_normal;
    /// Couleur de la lumiere
    glm::vec3 m_color;
    /// Constante de decroissante de la lumiere. Doit etre <= 1
    float m_falloff;
};

} // namespace host
} // namespace engine