#include <shape.h>

namespace engine
{
namespace host
{
IShape::IShape(const ShapeType& type, const glm::vec3& position, const BasicMaterial& material, const std::string& intersectionProgram)
    : m_worldPosition(position)
    , m_type(type)
    , m_material(material)
    , m_intersectionProgram(intersectionProgram)
{
}

} // namespace host
} // namespace engine