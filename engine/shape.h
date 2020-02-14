#pragma once

#include <optix.h>

#include <glm/vec3.hpp>
#include <string>

namespace engine
{

enum ShapeType
{
    SphereType,
    PlaneType
};

class IShape
{
public:
    IShape() = default;
    IShape(const ShapeType& type, const glm::vec3& position, const std::string& intersectionProgram);
    virtual ~IShape() {};

    virtual ShapeType GetShapeType() const = 0;
    virtual glm::vec3 GetWorldPosition() const = 0;
    virtual OptixAabb GetAabb() const = 0;
    inline virtual const char* GetIntersectionProgram() const { return m_intersectionProgram.c_str(); }

protected:
    ShapeType m_type;
    glm::vec3 m_worldPosition;
    OptixAabb m_boundingBox;
    std::string m_intersectionProgram;
};

} // namespace engine
