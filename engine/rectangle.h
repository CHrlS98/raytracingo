#pragma once
#include <ishape.h>

namespace engine
{
namespace host
{
class Rectangle : public IShape
{
public:
    Rectangle() = default;
    Rectangle(const glm::vec3& a, const glm::vec3& b, const glm::vec3& p0, const BasicMaterial& material);
    ~Rectangle() = default;

    virtual void CopyToDevice(device::HitGroupData& data) const override;

protected:
    virtual void BuildAabb() override;

private:
    glm::vec3 m_a;
    glm::vec3 m_b;
};
}// namespace host
}// namespace engine