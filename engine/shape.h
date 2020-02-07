#pragma once

#include <optix.h>

#include <glm/glm.hpp>

class Shape
{
public:
    Shape();
    Shape(const glm::vec3& worldPosition);

    inline glm::vec3 GetWorldPosition() { return m_worldPosition; }
    inline glm::vec3 GetWorldPosition() const { return m_worldPosition; }
    inline OptixAabb GetAabb() { return m_boundingBox; }
    inline OptixAabb GetAabb() const { return m_boundingBox; }
    inline void SetWorldPosition(glm::vec3 worldPosition) { m_worldPosition = worldPosition; }

protected:
    glm::vec3 m_worldPosition;
    OptixAabb m_boundingBox;
};