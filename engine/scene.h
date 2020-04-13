#pragma once

#include <shapefactory.h>

#include <vector>
#include <memory>

#include <glm/vec3.hpp>
#include <sutil/Camera.h>

namespace engine 
{
namespace host
{
class Shape;
class SurfaceLight;

/// Enumeration des scenes disponibles
enum class SceneModel
{
    SLIDE,
    CORNELL,
    PLATE,
    MIRROR_SPHERES
};

/// Classe gerant la scene dont on veut faire un rendu
class Scene
{
public:
    /// Constructeur
    /// \param[in] sceneModel La scene a construire
    /// \param[in] camWidth La largeur de la fenetre en pixels
    /// \param[in] camHeight La hauteur de la fenetre en pixels
    Scene(SceneModel sceneModel, const unsigned int& camWidth, const unsigned int& camHeight);

    /// Destructeur par defaut
    ~Scene() = default;

    /// Accesseur pour la propriete m_shapes
    /// \return Vecteur de pointeurs de Shape
    inline std::vector<std::shared_ptr<Shape>> GetShapes() const { return m_shapes; }

    /// Accesseur pour les lumieres
    /// \return Vecteur de lumieres m_surfaceLights
    inline std::vector<SurfaceLight> GetSurfaceLights() const { return m_surfaceLights; }

    /// Accesseur pour la camera
    /// \return Pointeur vers la camera de la scene
    inline std::shared_ptr<sutil::Camera> GetCamera() const { return m_camera; }

    /// Accesseur pour la largeur de la camera
    /// \return La largeur de la camera
    inline unsigned int GetCameraWidth() const { return m_cameraWidth; }

    /// Accesseur pour la hauteur de la camera
    /// \return La hauteur de la camera
    inline unsigned int GetCameraHeight() const { return m_cameraHeight; }

    /// Accesseur pour la couleur d'arriere-plan
    /// \return La couleur de l'arriere-plan
    inline glm::vec3 GetBackgroundColor() const { return m_backgroundColor; }

    /// Accesseur du nombre de primitives dans la scene
    /// \return Le nombre de primitives dans la scene
    inline int GetNbObjects() const { return m_nbObjects; }

private:
    /// Vecteur de pointeurs de Shape composant la scene
    std::vector<std::shared_ptr<Shape>> m_shapes;

    /// Vecteur de lumieres
    std::vector<SurfaceLight> m_surfaceLights;

    /// Pointeur vers la camera de la scene
    std::shared_ptr<sutil::Camera> m_camera;

    /// Couleur de l'arriere-plan
    glm::vec3 m_backgroundColor;

    /// Instance de ShapeFactory utilisee pour instancier des objets
    ShapeFactory m_factory;

    /// Nombre d'objets
    int m_nbObjects;

    /// Largeur de la camera (fenetre)
    unsigned int m_cameraWidth;

    /// Hauteur de la camera (fenetre)
    unsigned int m_cameraHeight;

    /// Scene choisie
    SceneModel m_sceneModel;

    /// Initialiser la camera
    void SetupCamera();

    /// Initialiser les objets geometriques
    void SetupObjects();

    /// Creer la scene PLATE
    void CreateFunPlate();

    /// Creer la scene SLIDE
    void CreateSlide();

    /// Creer la scene FILIP
    void CreateFilip();

    /// Creer la scene CHECKERED
    void CreateCheckeredFloor();

    /// Creer la scene BALLS
    void CreateBalls();

    /// Creer la scene CORNELL
    void CreateCornellBox();

    /// Creer la scene MIRROR_SPHERES
    void CreateMirrorSpheres();

    /// Ajouter un objet dans la scene lors de l'initialisation
    /// \param[in] object Paire contenant la Shape et le nombre de primitives qu'elle contient
    void AddObject(const std::pair<std::shared_ptr<Shape>, int>& object);

    /// Fonction utilitaire pour generer une matrice de mise a l'echelle
    /// \param[in] sx Facteur d'echelle en x
    /// \param[in] sy Facteur d'echelle en y
    /// \param[in] sz Facteur d'echelle en z
    /// \return Une matrice 4x4 de mise a l'echelle
    sutil::Matrix4x4 GetScale(float sx, float sy, float sz) const;

    /// Fonction utilitaire pour generer une matrice de translation
    /// \param[in] tx Translation en x
    /// \param[in] ty Translation en y
    /// \param[in] tz Translation en z
    /// \return Une matrice 4x4 de translation
    sutil::Matrix4x4 GetTranslate(float tx, float ty, float tz) const;

    /// Fonction utilitaire pour generer une matrice de rotation
    /// \param[in] angleRad L'angle en radian de la rotation
    /// \param[in] vx Composante en x de l'axe de rotation
    /// \param[in] vy Composante en y de l'axe de rotation
    /// \param[in] vz Composante en z de l'axe de rotation
    /// \return Une matrice 4x4 de rotation
    sutil::Matrix4x4 GetRotate(float angleRad, float vx, float vy, float vz) const;
};
} // namespace host
} // namespace engine
