#pragma once

#include <shape.h>

class Sphere : public Shape
{
public:
    Sphere();
    Sphere(const glm::vec3& worldPosition, const float radius);

    inline float GetRadius() { return m_radius; }
    inline float GetRadius() const { return m_radius; }

private:
    float m_radius;

    typedef Shape super;
};