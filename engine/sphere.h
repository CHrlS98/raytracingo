#pragma once

#include <memory>

#include <shape.h>

namespace engine
{

class Sphere : public IShape
{
public:
    Sphere();
    Sphere(const glm::vec3& worldPosition, const float radius);
    ~Sphere() = default;

    inline float GetRadius() const { return m_radius; }
    ShapeType GetShapeType() const override { return m_type; }
    glm::vec3 GetWorldPosition() const override { return m_worldPosition; }
    OptixAabb GetAabb() const override { return m_boundingBox; }

private:
    float m_radius;
};

} // namespace engine
