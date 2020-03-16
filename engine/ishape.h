#pragma once

#include <optix.h>
#include <params.h>
#include <basicmaterial.h>

#include <glm/vec3.hpp>
#include <string>

namespace engine
{
namespace host
{
class IShape
{
public:
    IShape() = default;
    IShape(const glm::vec3& position, const BasicMaterial& material, const std::string& intersectionProgram);
    virtual ~IShape() = default;

    /// Copie la representation geometrique de l'objet dans data
    virtual void CopyToDevice(device::HitGroupData& data) const = 0;

    /// Getters
    inline virtual glm::vec3 GetWorldPosition() const { return m_worldPosition; }
    inline virtual OptixAabb GetAabb() const { return m_boundingBox; }
    inline virtual const char* GetIntersectionProgram() const { return m_intersectionProgram.c_str(); }

protected:
    glm::vec3 m_worldPosition;
    OptixAabb m_boundingBox;
    std::string m_intersectionProgram;
    BasicMaterial m_material;

    virtual void BuildAabb() = 0;
};
} // namespace host
} // namespace engine
