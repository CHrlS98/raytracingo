#pragma once

#include <memory>

#include <shape.h>

namespace engine
{
namespace host 
{
class Sphere : public IShape
{
public:
    Sphere();
    Sphere(const glm::vec3& worldPosition, const float radius, const BasicMaterial& material);
    ~Sphere() = default;

    virtual void CopyToDevice(device::HitGroupData& data) const override;

    inline float GetRadius() const { return m_radius; }
    ShapeType GetShapeType() const override { return m_type; }
    glm::vec3 GetWorldPosition() const override { return m_worldPosition; }
    OptixAabb GetAabb() const override { return m_boundingBox; }

protected:
    virtual void BuildAabb() override;

private:
    float m_radius;
};
} // namespace host
} // namespace engine
