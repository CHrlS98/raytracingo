#pragma once

#include <vector>
#include <memory>

namespace engine 
{

constexpr unsigned int NB_OBJ = 3;

class IShape;

class Scene
{
public:
    Scene();
    ~Scene() = default;

    inline std::vector<std::shared_ptr<IShape>> GetShapes() const { return m_shapes; }

private:
    std::vector<std::shared_ptr<IShape>> m_shapes;
};

} // namespace engine
