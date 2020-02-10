#include <shape.h>

namespace engine
{

Shape::Shape()
{
    Shape(glm::vec3(0.0f));
}

Shape::Shape(const glm::vec3& worldPosition)
    : m_type(ShapeType::None)
    , m_worldPosition(worldPosition)
    , m_boundingBox({})
{
}

} // namespace engine
