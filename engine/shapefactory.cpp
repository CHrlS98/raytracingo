#include <shapefactory.h>
#include <primitive.h>
#include <vector>

namespace engine
{
namespace host
{
std::pair<std::shared_ptr<Shape>, int> ShapeFactory::CreateRectangle(const sutil::Matrix4x4& modelMatrix, const BasicMaterial& material) const
{
    std::vector<Primitive> primitives(1, Primitive(PRIMITIVE_TYPE::RECTANGLE, sutil::Matrix4x4::identity(), material));

    return std::make_pair<std::shared_ptr<Shape>, int>(std::make_shared<Shape>(Shape(primitives, modelMatrix)), 1);
}

std::pair<std::shared_ptr<Shape>, int> ShapeFactory::CreateOpenCylinder(const sutil::Matrix4x4& modelMatrix, const BasicMaterial& material) const
{
    std::vector<Primitive> primitives(1, Primitive(PRIMITIVE_TYPE::CYLINDER, sutil::Matrix4x4::identity(), material));

    return std::make_pair<std::shared_ptr<Shape>, int>(std::make_shared<Shape>(Shape(primitives, modelMatrix)), 1);
}

std::pair<std::shared_ptr<Shape>, int> ShapeFactory::CreateClosedCylinder(const sutil::Matrix4x4& modelMatrix, const BasicMaterial& material) const
{
    std::vector<Primitive> primitives;
    primitives.push_back(Primitive(PRIMITIVE_TYPE::CYLINDER, sutil::Matrix4x4::identity(), material));
    primitives.push_back(Primitive(PRIMITIVE_TYPE::DISK, sutil::Matrix4x4::translate(make_float3(0.0f, 1.0f, 0.0f)), material));
    primitives.push_back(Primitive(PRIMITIVE_TYPE::DISK, sutil::Matrix4x4::translate(make_float3(0.0f, -1.0f, 0.0f)), material));

    return std::make_pair<std::shared_ptr<Shape>, int>(std::make_shared<Shape>(Shape(primitives, modelMatrix)), 3);
}

std::pair<std::shared_ptr<Shape>, int> ShapeFactory::CreateDisk(const sutil::Matrix4x4& modelMatrix, const BasicMaterial& material) const
{
    std::vector<Primitive> primitives(1, Primitive(PRIMITIVE_TYPE::DISK, sutil::Matrix4x4::identity(), material));

    return std::make_pair<std::shared_ptr<Shape>, int>(std::make_shared<Shape>(Shape(primitives, modelMatrix)), 1);
}

std::pair<std::shared_ptr<Shape>, int> ShapeFactory::CreateSphere(const sutil::Matrix4x4& modelMatrix, const BasicMaterial& material) const
{
    std::vector<Primitive> primitives(1, Primitive(PRIMITIVE_TYPE::SPHERE, sutil::Matrix4x4::identity(), material));

    return std::make_pair<std::shared_ptr<Shape>, int>(std::make_shared<Shape>(Shape(primitives, modelMatrix)), 1);
}

std::pair<std::shared_ptr<Shape>, int> ShapeFactory::CreateCube(const sutil::Matrix4x4& modelMatrix, const BasicMaterial& material) const
{
    std::vector<Primitive> primitives;
    const float3 xAxis = make_float3(1.0f, 0.0f, 0.0f);
    const float3 zAxis = make_float3(0.0f, 0.0f, 1.0f);

    const sutil::Matrix4x4 modelMatrix0 = sutil::Matrix4x4::translate(make_float3(0.5f, 0.0f, 0.0f)) * sutil::Matrix4x4::rotate(M_PIf / 2.0f, zAxis);
    const sutil::Matrix4x4 modelMatrix1 = sutil::Matrix4x4::translate(make_float3(-0.5f, 0.0f, 0.0f)) * sutil::Matrix4x4::rotate(-M_PIf / 2.0f, zAxis);
    const sutil::Matrix4x4 modelMatrix2 = sutil::Matrix4x4::translate(make_float3(0.0f, 0.5f, 0.0f));
    const sutil::Matrix4x4 modelMatrix3 = sutil::Matrix4x4::translate(make_float3(0.0f, -0.5f,0.0f));
    const sutil::Matrix4x4 modelMatrix4 = sutil::Matrix4x4::translate(make_float3(0.0f, 0.0f, 0.5f)) * sutil::Matrix4x4::rotate(M_PIf / 2.0f, xAxis);
    const sutil::Matrix4x4 modelMatrix5 = sutil::Matrix4x4::translate(make_float3(0.0f, 0.0f, -0.5f)) * sutil::Matrix4x4::rotate(-M_PIf / 2.0f, xAxis);

    primitives.push_back(Primitive(PRIMITIVE_TYPE::RECTANGLE, modelMatrix0, material));
    primitives.push_back(Primitive(PRIMITIVE_TYPE::RECTANGLE, modelMatrix1, material));
    primitives.push_back(Primitive(PRIMITIVE_TYPE::RECTANGLE, modelMatrix2, material));
    primitives.push_back(Primitive(PRIMITIVE_TYPE::RECTANGLE, modelMatrix3, material));
    primitives.push_back(Primitive(PRIMITIVE_TYPE::RECTANGLE, modelMatrix4, material));
    primitives.push_back(Primitive(PRIMITIVE_TYPE::RECTANGLE, modelMatrix5, material));

    return std::make_pair<std::shared_ptr<Shape>, int>(std::make_shared<Shape>(Shape(primitives, modelMatrix)), 6);
}
} // namespace host
} // namespace engine