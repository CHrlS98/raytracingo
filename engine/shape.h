#pragma once

#include <optix.h>

#include <glm/glm.hpp>

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

    inline ShapeType GetShapeType() { return m_type; }
    inline ShapeType GetShapeType() const { return m_type; }
    inline glm::vec3 GetWorldPosition() { return m_worldPosition; }
    inline glm::vec3 GetWorldPosition() const { return m_worldPosition; }
    inline OptixAabb GetAabb() { return m_boundingBox; }
    inline OptixAabb GetAabb() const { return m_boundingBox; }
    inline void SetWorldPosition(glm::vec3 worldPosition) { m_worldPosition = worldPosition; }

protected:
    ShapeType m_type;
    glm::vec3 m_worldPosition;
    OptixAabb m_boundingBox;
};