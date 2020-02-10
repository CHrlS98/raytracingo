#pragma once

#include <optix.h>

#include <glm/vec3.hpp>

namespace engine
{

enum ShapeType
{
    None,
    SphereType
};

class Shape
{
public:
    Shape();
    Shape(const glm::vec3& worldPosition);
    virtual ~Shape() {};

    inline ShapeType GetShapeType() const { return m_type; }
    inline glm::vec3 GetWorldPosition() const { return m_worldPosition; }
    inline OptixAabb GetAabb() const { return m_boundingBox; }

protected:
    ShapeType m_type;
    glm::vec3 m_worldPosition;
    OptixAabb m_boundingBox;
};

} // namespace engine
