#pragma once

#include <vector>
#include <shape.h>

class Scene
{
public:
    Scene();

    inline std::vector<Shape> GetShapes() const { return m_shapes; }
    inline size_t GetShapesCount() const { return m_shapes.size(); }

private:
    std::vector<Shape> m_shapes;
};