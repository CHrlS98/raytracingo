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
    Shape() = default;
    Shape(const std::vector<Primitive>& primitives, const sutil::Matrix4x4& modelMatrix);
    virtual ~Shape() = default;

    /// Appliquer la transformation transform a m_transformMatrix
    virtual void Transform(const sutil::Matrix4x4& transform);

    /// Getters
    inline std::vector<Primitive> GetPrimitives() const { return m_primitives; }

protected:
    std::vector<Primitive> m_primitives;
};
} // namespace host
} // namespace engine
