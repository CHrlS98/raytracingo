#pragma once
#include <shape.h>
#include <glm/vec3.hpp>

namespace engine
{
class Plane : public IShape
{
public:
    Plane() = default;
    Plane(const glm::vec3& position, const glm::vec3& normal);
    ~Plane() = default;

    inline ShapeType GetShapeType() const override { return m_type; }
    inline glm::vec3 GetWorldPosition() const override { return m_worldPosition; }
    inline OptixAabb GetAabb() const override { return m_boundingBox; }

    inline glm::vec3 GetNormal() const { return m_normal; }

private:
    glm::vec3 m_normal;
    void BuildAabb();
};
}// namespace engine