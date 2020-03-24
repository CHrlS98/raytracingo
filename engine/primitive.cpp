#include <primitive.h>

namespace engine
{
namespace host
{
namespace
{
const std::string CYLINDER_INTERSECTION_PROGRAM = "__intersection__cylinder";
const std::string DISK_INTERSECTION_PROGRAM = "__intersection__disk";
const std::string RECTANGLE_INTERSECTION_PROGRAM = "__intersection__rectangle";
const std::string SPHERE_INTERSECTION_PROGRAM = "__intersection__sphere";
}

Primitive::Primitive(PRIMITIVE_TYPE type, const sutil::Matrix4x4& modelMatrix, const BasicMaterial& material)
    : m_type(type)
    , m_modelMatrix(modelMatrix)
    , m_material(material)
    , m_intersectionProgram()
{
    switch (m_type)
    {
    case engine::host::PRIMITIVE_TYPE::CYLINDER:
        m_intersectionProgram = CYLINDER_INTERSECTION_PROGRAM;
        break;
    case engine::host::PRIMITIVE_TYPE::DISK:
        m_intersectionProgram = DISK_INTERSECTION_PROGRAM;
        break;
    case engine::host::PRIMITIVE_TYPE::RECTANGLE:
        m_intersectionProgram = RECTANGLE_INTERSECTION_PROGRAM;
        break;
    case engine::host::PRIMITIVE_TYPE::SPHERE:
        m_intersectionProgram = SPHERE_INTERSECTION_PROGRAM;
        break;
    }
    BuildAabb();
}

void Primitive::BuildAabb()
{
    m_aabb.maxX = 15.0f;
    m_aabb.minX = -15.0f;
    m_aabb.maxY = 15.0f;
    m_aabb.minY = -15.0f;
    m_aabb.maxZ = 15.0f;
    m_aabb.minZ = -15.0f;
}

void Primitive::Transform(const sutil::Matrix4x4& transform)
{
    m_modelMatrix = transform * m_modelMatrix;
}

void Primitive::CopyToDevice(device::HitGroupData& data) const
{
    // Materiel
    const glm::vec3& ka = m_material.GetKa();
    const glm::vec3& kd = m_material.GetKd();
    const glm::vec3& ks = m_material.GetKs();
    const glm::vec3& kr = m_material.GetKr();
    data.material.basicMaterial.ka = { ka.r, ka.g, ka.b };
    data.material.basicMaterial.kd = { kd.r, kd.g, kd.b };
    data.material.basicMaterial.ks = { ks.r, ks.g, ks.b };
    data.material.basicMaterial.kr = { kr.r, kr.g, kr.b };
    data.material.basicMaterial.alpha = m_material.GetAlpha();

    // Transformations affines
    data.modelMatrix = m_modelMatrix;
}


} // namespace host
} // namespace engine