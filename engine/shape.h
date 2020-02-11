#pragma once

#include <optix.h>

#include <glm/vec3.hpp>

namespace engine
{

enum ShapeType
{
    SphereType
};

class IShape
{
public:
    IShape() = default;
    virtual ~IShape() {};

    virtual ShapeType GetShapeType() const = 0;
    virtual glm::vec3 GetWorldPosition() const = 0;
    virtual OptixAabb GetAabb() const = 0;

protected:
    ShapeType m_type;
    glm::vec3 m_worldPosition;
    OptixAabb m_boundingBox;
};

} // namespace engine
