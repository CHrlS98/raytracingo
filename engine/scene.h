#pragma once

#include <vector>
#include <memory>

namespace engine 
{

constexpr unsigned int NB_OBJ = 3;

class Shape;

class Scene
{
public:
    Scene();
    ~Scene() = default;

    inline std::vector<std::shared_ptr<Shape>> GetShapes() const { return m_shapes; }

private:
    std::vector<std::shared_ptr<Shape>> m_shapes;
};

} // namespace engine
