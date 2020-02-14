#include <shape.h>

namespace engine
{
IShape::IShape(const ShapeType& type, const glm::vec3& position, const std::string& intersectionProgram)
    : m_worldPosition(position)
    , m_type(type)
    , m_intersectionProgram(intersectionProgram)
{
}

} // namespace engine