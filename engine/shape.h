#pragma once

#include <params.h>
#include <primitive.h>
#include <sutil/Matrix.h>
#include <optix.h>
#include <vector>

namespace engine
{
namespace host
{
class Shape
{
public:
    /// Constructeur par defaut
    Shape() = default;

    /// Constructeur
    /// \param[in] primitives Vecteurs de primitives composant la Shape
    /// \param[in] modelMatrix Matrice de transformation de l'objet
    Shape(const std::vector<Primitive>& primitives, const sutil::Matrix4x4& modelMatrix);

    /// destructeur par defaut
    virtual ~Shape() = default;

    /// Appliquer la transformation transform a m_transformMatrix
    /// \param[in] transform Matrice de transformation a appliquer
    virtual void Transform(const sutil::Matrix4x4& transform);

    /// Getter des primitives
    /// \return Vecteur de primitives
    inline std::vector<Primitive> GetPrimitives() const { return m_primitives; }

protected:
    std::vector<Primitive> m_primitives;
};
} // namespace host
} // namespace engine
