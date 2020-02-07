#pragma once

#include <shape.h>

class Sphere : Shape
{
public:
    Sphere();
    Sphere(const glm::vec3& worldPosition);

private:
    typedef Shape super;
};