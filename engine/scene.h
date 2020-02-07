#pragma once

#include <vector>
#include <shape.h>

class Scene
{
public:
    Scene();

    inline std::vector<Shape> GetShapes() { return m_shapes; }
    inline std::vector<Shape> GetShapes() const { return m_shapes; }
    inline size_t GetShapesCount() { return m_shapes.size(); }
    inline size_t GetShapesCount() const { return m_shapes.size(); }

    void AddShape(const Shape& shape);

private:
    std::vector<Shape> m_shapes;

    //std::vector<Light> m_lights;
};