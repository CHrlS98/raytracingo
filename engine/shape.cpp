#include <shape.h>

namespace engine
{
namespace host
{
Shape::Shape(const std::vector<Primitive>& primitives, const sutil::Matrix4x4& modelMatrix)
    : m_primitives(primitives)
{
    Transform(modelMatrix);
}

void Shape::Transform(const sutil::Matrix4x4& transform)
{
    for (Primitive& p : m_primitives)
    {
        p.Transform(transform);
    }
}
} // namespace host
} // namespace engine