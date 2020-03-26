#pragma once

#include <basicmaterial.h>
#include <params.h>

#include <optix.h>

#include <sutil/Matrix.h>
#include <string>

namespace engine
{
namespace host
{
enum class PRIMITIVE_TYPE
{
    CYLINDER,
    DISK,
    RECTANGLE,
    SPHERE
};

class Primitive
{
public:
    /// Constructeurs
    Primitive() = delete;
    Primitive(PRIMITIVE_TYPE type, const sutil::Matrix4x4& modelMatrix, const BasicMaterial& material);
    
    /// Destructeur
    ~Primitive() = default;

    /// Copie la representation de l'objet dans data
    void CopyToDevice(device::HitGroupData& data) const;

    /// Appliquer la transformation transform a m_transformMatrix
    void Transform(const sutil::Matrix4x4& transform);

    // Getter
    inline OptixAabb GetAabb() const { return m_aabb; }
    inline const char* GetIntersectionProgram() const { return m_intersectionProgram.c_str(); }
    inline sutil::Matrix4x4 GetModelMatrix() const { return m_modelMatrix; }

private:
    PRIMITIVE_TYPE m_type;
    sutil::Matrix4x4 m_modelMatrix;
    BasicMaterial m_material;
    OptixAabb m_aabb;
    std::string m_intersectionProgram;

    void BuildAabb();
};
} // namespace host
} // namespace engine