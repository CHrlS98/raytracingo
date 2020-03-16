#pragma once
#include <ishape.h>

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

    /// Getters
    inline float GetRadius() const { return m_radius; }

protected:
    virtual void BuildAabb() override;

private:
    float m_radius;
};
} // namespace host
} // namespace engine
