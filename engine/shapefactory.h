#pragma once
#include <shape.h>
#include <memory>

namespace engine
{
namespace host
{
/// Classe d'utilitaire pour creer des formes
class ShapeFactory
{
public:
    /// Constructeur par defaut
    ShapeFactory() = default;

    /// Destructeur par defaut
    ~ShapeFactory() = default;

    /// Contruire un rectangle
    /// \param[in] modelMatrix Matrice de transformation de l'objet
    /// \param[in] material Materiel de l'objet
    /// \return Paire contenant un pointeur vers l'objet instancie et un entier representant le nombre de primitives le composant
    std::pair<std::shared_ptr<Shape>, int> CreateRectangle(const sutil::Matrix4x4& modelMatrix, const BasicMaterial& material) const;

    /// Construire un cylindre ouvert
    /// \param[in] modelMatrix Matrice de transformation de l'objet
    /// \param[in] material Materiel de l'objet
    /// \return Paire contenant un pointeur vers l'objet instancie et un entier representant le nombre de primitives le composant
    std::pair<std::shared_ptr<Shape>, int> CreateOpenCylinder(const sutil::Matrix4x4& modelMatrix, const BasicMaterial& material) const;

    /// Construire un cylindre ferme
    /// \param[in] modelMatrix Matrice de transformation de l'objet
    /// \param[in] material Materiel de l'objet
    /// \return Paire contenant un pointeur vers l'objet instancie et un entier representant le nombre de primitives le composant
    std::pair<std::shared_ptr<Shape>, int> CreateClosedCylinder(const sutil::Matrix4x4& modelMatrix, const BasicMaterial& material) const;

    /// Contruire un disque
    /// \param[in] modelMatrix Matrice de transformation de l'objet
    /// \param[in] material Materiel de l'objet
    /// \return Paire contenant un pointeur vers l'objet instancie et un entier representant le nombre de primitives le composant
    std::pair<std::shared_ptr<Shape>, int> CreateDisk(const sutil::Matrix4x4& modelMatrix, const BasicMaterial& material) const;

    /// Construire une sphere
    /// \param[in] modelMatrix Matrice de transformation de l'objet
    /// \param[in] material Materiel de l'objet
    /// \return Paire contenant un pointeur vers l'objet instancie et un entier representant le nombre de primitives le composant
    std::pair<std::shared_ptr<Shape>, int> CreateSphere(const sutil::Matrix4x4& modelMatrix, const BasicMaterial& material) const;

    /// Construire un cube centre a l'origine
    /// \param[in] modelMatrix Matrice de transformation de l'objet
    /// \param[in] material Materiel de l'objet
    /// \return Paire contenant un pointeur vers l'objet instancie et un entier representant le nombre de primitives le composant
    std::pair<std::shared_ptr<Shape>, int> CreateCube(const sutil::Matrix4x4& modelMatrix, const BasicMaterial& material) const;

    /// Construire une forme personnalisee a partir d'un ensemble de primitives
    /// \param[in] modelMatrix Matrice de transformation de l'objet
    /// \param[in] material Materiel de l'objet
    /// \return Paire contenant un pointeur vers l'objet instancie et un entier representant le nombre de primitives le composant
    std::pair<std::shared_ptr<Shape>, int> CreateCustom(const std::vector<Primitive>& primitives, const sutil::Matrix4x4& modelMatrix) const;
};
}
}