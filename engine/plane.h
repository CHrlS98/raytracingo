#pragma once
#include <shape.h>
#include <glm/vec3.hpp>

namespace engine
{
namespace host
{
class Plane : public IShape
{
public:
    Plane() = default;
    Plane(const glm::vec3& position, const glm::vec3& normal, const BasicMaterial& material);
    ~Plane() = default;

    virtual void CopyToDevice(device::HitGroupData& data) const override;

    inline ShapeType GetShapeType() const override { return m_type; }
    inline glm::vec3 GetWorldPosition() const override { return m_worldPosition; }
    inline OptixAabb GetAabb() const override { return m_boundingBox; }

    inline glm::vec3 GetNormal() const { return m_normal; }

protected:
    virtual void BuildAabb() override;

private:
    glm::vec3 m_normal;
};
} // namespace host
} // namespace engine