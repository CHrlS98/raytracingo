#pragma once

#include <memory>

#include <shape.h>

namespace engine
{

class Sphere : public Shape
{
public:
    Sphere();
    Sphere(const glm::vec3& worldPosition, const float radius);
    ~Sphere() = default;

    inline float GetRadius() const { return m_radius; }

private:
    float m_radius;

    typedef Shape super;
};

} // namespace engine