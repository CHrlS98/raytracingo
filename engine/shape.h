#pragma once

#include <optix.h>

#include <glm/glm.hpp>

class Shape
{
public:
    Shape();
    Shape(const glm::vec3& worldPosition);

    inline glm::vec3 GetWorldPosition() { return m_worldPosition; }

protected:
    glm::vec3 m_worldPosition;
    OptixAabb m_boundingBox;
};