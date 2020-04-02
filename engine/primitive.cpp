#include <primitive.h>
#include <algorithm>
#include <sutil/vec_math.h>
#include <iostream>

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
const float AABB_EPSILON = 0.001f;
const float SCENE_MAX_BOUND = 50.0f;
}

CubeBox::CubeBox()
    : face0()
    , face1()
{
    // Par defaut le cube box est aligne avec les axes principaux et de dimensions 2x2x2
    // Premiere ligne
    face0[0] = -1.f; face0[1] = -1.f; face0[2] = 1.f; face0[3] = 1.f; face1[0] = -1.f; face1[1] = -1.f; face1[2] = 1.f; face1[3] = 1.f;
    // Deuxieme ligne
    face0[4] = -1.f; face0[5] = -1.f; face0[6] = -1.f; face0[7] = -1.f; face1[4] = 1.f; face1[5] = 1.f; face1[6] = 1.f; face1[7] = 1.f;
    // Troisieme ligne
    face0[8] = -1.f; face0[9] = 1.f; face0[10] = -1.f; face0[11] = 1.f; face1[8] = -1.f; face1[9] = 1.f; face1[10] = -1.f; face1[11] = 1.f;
    // Derniere ligne
    face0[12] = 1.f; face0[13] = 1.f; face0[14] = 1.f; face0[15] = 1.f; face1[12] = 1.f; face1[13] = 1.f; face1[14] = 1.f; face1[15] = 1.f;
}

void CubeBox::TransformAndAlign(const sutil::Matrix4x4& modelMatrix)
{
    // On transforme le cube
    face0 = modelMatrix * face0;
    face1 = modelMatrix * face1;

    // On calcule ses dimensions maximales
    float minX = SCENE_MAX_BOUND;
    float maxX = -SCENE_MAX_BOUND;
    float minY = SCENE_MAX_BOUND;
    float maxY = -SCENE_MAX_BOUND;
    float minZ = SCENE_MAX_BOUND;
    float maxZ = -SCENE_MAX_BOUND;
    for (int i = 0; i < 4; ++i)
    {
        minX = std::min(std::min(minX, face0[i]), face1[i]);
        maxX = std::max(std::max(maxX, face0[i]), face1[i]);

        minY = std::min(std::min(minY, face0[i + 4]), face1[i + 4]);
        maxY = std::max(std::max(maxY, face0[i + 4]), face1[i + 4]);

        minZ = std::min(std::min(minZ, face0[i + 8]), face1[i + 8]);
        maxZ = std::max(std::max(maxZ, face0[i + 8]), face1[i + 8]);
    }

    // On ajoute un petit epsilon pour la tolerance
    minX -= AABB_EPSILON;
    maxX += AABB_EPSILON;
    minY -= AABB_EPSILON;
    maxY += AABB_EPSILON;
    minZ -= AABB_EPSILON;
    maxZ += AABB_EPSILON;

    // On cree un nouveau cube aligne avec les axes principaux qui contient le cube transforme
    // Premiere ligne - coordonnee X
    face0[0] = minX; face0[1] = minX; face0[2] = maxX; face0[3] = maxX; face1[0] = minX; face1[1] = minX; face1[2] = maxX; face1[3] = maxX;
    // Deuxieme ligne - coordonnee Y
    face0[4] = minY; face0[5] = minY; face0[6] = minY; face0[7] = minY; face1[4] = maxY; face1[5] = maxY; face1[6] = maxY; face1[7] = maxY;
    // Troisieme ligne - coordonnee Z
    face0[8] = minZ; face0[9] = maxZ; face0[10] = minZ; face0[11] = maxZ; face1[8] = minZ; face1[9] = maxZ; face1[10] = minZ; face1[11] = maxZ;
    // Derniere ligne - Coordonnees homogenes
    face0[12] = 1.f; face0[13] = 1.f; face0[14] = 1.f; face0[15] = 1.f; face1[12] = 1.f; face1[13] = 1.f; face1[14] = 1.f; face1[15] = 1.f;
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
}

OptixAabb Primitive::GetAabb() const
{
    // On instancie une boite unitaire
    CubeBox cubeBox;
    cubeBox.TransformAndAlign(m_modelMatrix);

    // On remplit les donnees de la struct a retourner
    OptixAabb bb;
    bb.maxX = cubeBox.GetMaxX();
    bb.minX = cubeBox.GetMinX();
    bb.maxY = cubeBox.GetMaxY();
    bb.minY = cubeBox.GetMinY();
    bb.maxZ = cubeBox.GetMaxZ();
    bb.minZ = cubeBox.GetMinZ();

    return bb;
}

void Primitive::Transform(const sutil::Matrix4x4& transform)
{
    m_modelMatrix = transform * m_modelMatrix;
}

void Primitive::CopyToDevice(device::HitGroupData& data) const
{
    // Materiel
    const glm::vec3& kd = m_material.GetKd();
    data.material.basicMaterial.kd = { kd.r, kd.g, kd.b };
    data.material.basicMaterial.roughness = m_material.GetRoughness();

    // Transformations affines
    data.modelMatrix = m_modelMatrix;
}


} // namespace host
} // namespace engine