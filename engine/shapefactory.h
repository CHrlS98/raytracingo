#pragma once
#include <shape.h>
#include <memory>

namespace engine
{
namespace host
{
class ShapeFactory
{
public:
    /// Constructeurs
    ShapeFactory() = default;
    ~ShapeFactory() = default;

    /// Contruire rectangle
    std::pair<std::shared_ptr<Shape>, int> CreateRectangle(const sutil::Matrix4x4& modelMatrix, const BasicMaterial& material) const;

    /// Construire cylindre ouvert
    std::pair<std::shared_ptr<Shape>, int> CreateOpenCylinder(const sutil::Matrix4x4& modelMatrix, const BasicMaterial& material) const;

    /// Construire cylindre ferme
    std::pair<std::shared_ptr<Shape>, int> CreateClosedCylinder(const sutil::Matrix4x4& modelMatrix, const BasicMaterial& material) const;

    /// Contruire un disque
    std::pair<std::shared_ptr<Shape>, int> CreateDisk(const sutil::Matrix4x4& modelMatrix, const BasicMaterial& material) const;

    /// Construire une sphere
    std::pair<std::shared_ptr<Shape>, int> CreateSphere(const sutil::Matrix4x4& modelMatrix, const BasicMaterial& material) const;

    /// Construire un cube centre a l'origine
    std::pair<std::shared_ptr<Shape>, int> CreateCube(const sutil::Matrix4x4& modelMatrix, const BasicMaterial& material) const;

    /// Construire une forme personnalisee a partir d'un ensemble de primitives
    std::pair<std::shared_ptr<Shape>, int> CreateCustom(const std::vector<Primitive>& primitives, const sutil::Matrix4x4& modelMatrix) const;
};
}
}