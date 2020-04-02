#pragma once

#include <glm/glm.hpp> 
#include <primitive.h> 

namespace engine
{
namespace host
{
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
    /// Getter pour la propriete m_intersectionProgram
    /// \return La c-string du programme d'intersection de la lumière
    inline const char* GetIntersectionProgram() const { return m_intersectionProgram.c_str(); }
    /// Getter pour la propriete m_aabb
    /// \return Le volume englobant de la lumiere
    inline OptixAabb GetAabb() const { return m_aabb; }

    /// Copie la representation de l'objet sur le GPU dans data
    void SurfaceLight::CopyToDevice(device::HitGroupData& data) const;

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
    /// Programme d'intersection de la lumière
    std::string m_intersectionProgram;
    /// Volume englobant de la lumiere
    OptixAabb m_aabb;

    void BuildAabb();
};

} // namespace host
} // namespace engine