#pragma once

#include <basicmaterial.h>
#include <params.h>

#include <optix.h>

#include <sutil/Matrix.h>
#include <string>

namespace engine
{
namespace host
{
/// Enumeration des types de primitives disponibles
enum class PRIMITIVE_TYPE
{
    CYLINDER,
    DISK,
    RECTANGLE,
    SPHERE
};

/// Boite rectangulaire alignee sur les axes utilisee pour creer les AABB
struct CubeBox
{
    /// Constructeur
    CubeBox();

    /// Transforme la boite et aligne ses faces avec les axes
    /// \param[in]model Matrice de transformation
    void TransformAndAlign(const sutil::Matrix4x4& model);

    /// Obtenir la coordonnee minimale en X
    /// \return La coordonnee minimale en X
    inline float GetMinX() { return face0[0]; }

    /// Obtenir la coordonnee minimale en Y
    /// \return La coordonnee minimale en Y
    inline float GetMinY() { return face0[4]; }

    /// Obtenir la coordonnee minimale en Z
    /// \return La coordonnee minimale en Z
    inline float GetMinZ() { return face0[8]; }

    /// Obtenir la coordonnee maximale en X
    /// \return La coordonnee maximale en X
    inline float GetMaxX() { return face1[3]; }

    /// Obtenir la coordonnee maximale en Y
    /// \return La coordonnee maximale en Y
    inline float GetMaxY() { return face1[7]; }

    /// Obtenir la coordonnee maximale en Z
    /// \return La coordonnee maximale en Z
    inline float GetMaxZ() { return face1[11]; }

    /// Matrice contenant les coordonnees des points de la premiere face du cube
    sutil::Matrix4x4 face0;

    /// Matrice contenant les coordonnees des points de la deuxieme face du cube
    sutil::Matrix4x4 face1;
};

/// Classe representant une primitive graphique
class Primitive
{
public:
    /// Constructeur par defaut pas disponible
    Primitive() = delete;

    /// Constructeur
    /// \param[in] type Type de primitive
    /// \param[in] modelMatrix Matrice de transformation de l'objet
    /// \param[in] material Materiau de l'objet
    Primitive(PRIMITIVE_TYPE type, const sutil::Matrix4x4& modelMatrix, const BasicMaterial& material);
    
    /// Destructeur
    ~Primitive() = default;

    /// Copie la representation de l'objet dans data
    void CopyToDevice(device::HitGroupData& data) const;

    /// Appliquer la transformation transform a m_transformMatrix
    void Transform(const sutil::Matrix4x4& transform);

    /// Accesseur pour le volume englobant
    /// \return Le volume englobant l'objet (axis-aligned bounding box)
    OptixAabb GetAabb() const;

    /// Accesseur du nom du program d'intersection
    /// \return Le nom du programme d'intersection
    inline const char* GetIntersectionProgram() const { return m_intersectionProgram.c_str(); }

    /// Accesseur de la matrice du modele
    /// \return La matrice de transformation de l'objet
    inline sutil::Matrix4x4 GetModelMatrix() const { return m_modelMatrix; }

    /// Accesseur du type de primitive
    /// \return Le type de la primitive
    inline PRIMITIVE_TYPE GetType() const { return m_type; }

    /// Accesseur du materiau de l'objet
    /// \return Le materiau de l'objet
    inline BasicMaterial GetMaterial() const { return m_material; }

private:
    /// Type de primitive
    PRIMITIVE_TYPE m_type;

    /// Matrice de transformation
    sutil::Matrix4x4 m_modelMatrix;

    /// Materiau de l'objet
    BasicMaterial m_material;

    /// Nom du programme d'intersection
    std::string m_intersectionProgram;
};
} // namespace host
} // namespace engine