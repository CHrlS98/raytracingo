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

struct CubeBox
{
    CubeBox();
    void TransformAndAlign(const sutil::Matrix4x4& model);
    inline float GetMinX() { return face0[0]; }
    inline float GetMinY() { return face0[4]; }
    inline float GetMinZ() { return face0[8]; }
    inline float GetMaxX() { return face1[3]; }
    inline float GetMaxY() { return face1[7]; }
    inline float GetMaxZ() { return face1[11]; }
    sutil::Matrix4x4 face0;
    sutil::Matrix4x4 face1;
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
    OptixAabb GetAabb() const;
    inline const char* GetIntersectionProgram() const { return m_intersectionProgram.c_str(); }
    inline sutil::Matrix4x4 GetModelMatrix() const { return m_modelMatrix; }
    inline PRIMITIVE_TYPE GetType() const { return m_type; }
    inline BasicMaterial GetMaterial() const { return m_material; }

private:
    PRIMITIVE_TYPE m_type;
    sutil::Matrix4x4 m_modelMatrix;
    BasicMaterial m_material;
    std::string m_intersectionProgram;
};
} // namespace host
} // namespace engine