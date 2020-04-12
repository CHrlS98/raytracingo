#pragma once

#include <engine/light.h>
#include <params.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Trackball.h>
#include <optix.h>

#include <vector>
#include <memory>

#include <scene.h>

namespace engine
{
namespace host
{

/// Etat du renderer apres l'execution d'un frame
struct RendererState
{
    /// Pointeur vers les parametres globaux de l'execution
    device::Params* params;

    /// Pointeur vers la 'trackball' utilisee pour la manipulation de la camera
    std::shared_ptr<sutil::Trackball> trackball;

    /// Vrai si l'etat de la camera est change
    bool cameraChangedFlag;

    /// Vrai quand la fenetre est redimensionnee
    bool windowResizeFlag;

    /// Vrai quand une demande de sauvegarde d'image est faite
    bool sendSaveRequest;

    /// Identifiant du bouton de souris avec lequel il y a interaction
    int mouseButton;
};

/// Niveau de severite du log
enum LogCallbackLevel
{
    Disable = 0,    // Setting the callback level will disable all messages.The callback function will not be called in this case
    Fatal = 1,      // A non-recoverable error. The context and/or OptiX itself might no longer be in a usable state
    Error = 2,      // A recoverable error, e.g., when passing invalid call parameters
    Warning = 3,    // Hints that OptiX might not behave exactly as requested by the user or may perform slower than expected
    Print = 4       // Status or progress messages
};

/// Modes de rendu disponibles
enum class RenderMode
{
    DISTRIBUTED_RAY_TRACING,
    PATH_TRACING
};

/// Classe gerant le rendu des objets a l'ecran
class Renderer
{
public:
    /// Constructeur
    /// \param[in] scene Scene a rendre
    /// \param[in] renderMode Type de rendu a effectuer
    /// \param[in] sqrtSamplePerPixel Nombre de sous-echantillons par pixel
    /// \param[in] useAmbientCoeff Vrai pour utiliser un coefficient d'illumination ambiante
    Renderer(std::shared_ptr<Scene> scene, RenderMode renderMode, int sqrtSamplePerPixel, bool useAmbientCoeff);

    /// Destructeur
    ~Renderer();

    /// Demarrer l'affichage de la scene
    void Display();

private:
    /// Pointeur vers le contexte OptiX
    OptixDeviceContext m_optixContext;

    /// Pointeur vers le module OptiX
    OptixModule m_module;

    /// Options de compilation du module OptiX
    OptixModuleCompileOptions m_moduleCompileOptions;

    /// Pointeur vers le pipeline OptiX
    OptixPipeline m_pipeline;

    /// Options de compilation du pipeline m_pipeline
    OptixPipelineCompileOptions m_pipelineCompileOptions;

    /// Options de "linking" du pipeline
    OptixPipelineLinkOptions m_pipelineLinkOptions;

    /// Pointeur vers le contexte cuda utilise
    CUcontext m_cudaContext;

    /// Pointeur vers le CUDAStream utilise
    CUstream  m_cudaStream;

    /// Proprietes du systeme CUDA utilise
    cudaDeviceProp m_cudaDeviceProperties;

    /// Vecteur des programmes OptiX utilises pour le rendu
    /// \note De maniere generale, on a un programme par primitive graphique,
    ///       un programme pour le lancer de rayons initial, un programme pour le
    ///       calcul des ombres et un programme pour les rayons sans intersection
    std::vector<OptixProgramGroup> m_programs;

    /// Le SBT fait le lien entre les variables definissant les objets 
    /// et les programmes sur le GPU
    OptixShaderBindingTable m_shaderBindingTable;

    /// Pointeur vers la scene a afficher
    std::shared_ptr<Scene> m_scene;

    /// Structure accelerante utilisee sur le GPU par OptiX
    CUdeviceptr m_deviceGasOutputBuffer;

    /// Handle pour la structure accelerante utilisee par OptiX
    OptixTraversableHandle m_traversableHandle;

    /// Mode de rendu utilise
    RenderMode m_renderMode;

    /// Etat de l'affichage
    RendererState m_state;

    /// Vrai pour utiliser un coefficient ambiant pour l'eclairage ddes objets dans l'obscurite
    bool m_useAmbientCoefficient;

    /// Nombre d'echantillons par pixel
    int m_sqrtSamplePerPixel;

    /// Initialiser l'ensemble des contextes et variables
    /// necessaires a OptiX pour le rendu d'une scene
    void Initialize();

    /// Initialiser Optix, charger tous les points d'entree de l'API
    void InitOptix();

    /// Creer le contexte OptiX sur le GPU
    void CreateContext();

    /// Creer le module Optix, c-a-d en compilant le code cuda
    void CreateModule();

    /// Definit le comportement d'OptiX pour le lancer des rayons originant de la camera
    void CreateRayGen();

    /// Definit le comportement d'OptiX lorsqu'aucun rayon n'intercepte d'objet
    void CreateMiss();

    /// Cree les formes a afficher (incluant volumes englobants et SBT)
    void CreateShapes();

    /// Creer le programme pour generer les rayons a partir de la camera
    /// \return Le programme de generation de rayons
    OptixProgramGroup CreateRayGenPrograms() const;

    /// Creer le programme OptiX pour generer les rayons sans intersections
    /// \return Le programme appele par les rayons sans intersection
    OptixProgramGroup CreateMissPrograms() const;

    /// Creer les programmes OptiX des objets intersectables
    /// \param[in] intersectionProgram Nom du programme d'intersection
    /// \param[in] type Type du rayon pour lequel on cree le programme
    /// \return Le programme d'intersection cree
    OptixProgramGroup CreateHitGroupProgram(const char* intersectionProgram, device::RayType type) const;

    /// Copie les SBT records du programme de generation de rayons sur le GPU
    /// \param[in] records Pointeur vers les CameraSbtRecords a pousser sur le GPU
    /// \param[in] recordsCount Le nombre de CameraSbtRecords a pousser sur le GPU
    void BuildRayGenRecords(CameraSbtRecord* records, const size_t& recordsCount);

    /// Copie les SBT records du programme de generation de rayons sur le GPU
    /// \param[in] records Pointeur vers les MissSbtRecords a pousser sur le GPU
    /// \param[in] recordsCount Le nombre de records a pousser sur le GPU
    void BuildMissRecords(MissSbtRecord* records, const size_t& recordsCount);

    /// Copie les SBT des hit groups sur le GPU
    /// \param[in] records Pointeur vers les HitGroupRecords a pousser sur le GPU
    /// \param[in] recordsCount Le nombre de records a pousser sur le GPU
    void BuildHitGroupRecords(HitGroupSbtRecord* records, const size_t& recordsCount);

    /// Construire la structure accelerante de la scene
    /// \param[in] aabb Pointeur vers la structure accelerante
    /// \param[in] aabbInputFlags Tableau de 'flags' pour chacun des volumes englobants
    /// \param[in] sbtIndex Tableau d'index mettant en correspondance les aabb avec les
    /// \param[in] nbObjects Le nombre total de primitives dans la scene
    void BuildAccelerationStructure(OptixAabb* aabb, uint32_t* aabbInputFlags, uint32_t* sbtIndex, const size_t& nbObjects);

    /// Copier les informations des lumieres
    /// \param[out] params Les parametres globaux pour un frame
    void WriteLights(device::Params& params);

    /// Creer le pipeline OptiX (c'est ici qu'est specifie la profondeur maximum de recursion)
    void CreatePipeline();

    /// Mettre a jour les variables apres un frame
    /// \param[out] outputBuffer Pointeur vers le buffer contenant l'image generee au dernier frame
    /// \param[out] params Reference vers les parametres passees au programme
    /// \param[in] firstLaunch Vrai lors de la premiere execution d'OptiX
    void Update(sutil::CUDAOutputBuffer<uchar4>* outputBuffer, device::Params& params, bool firstLaunch);

    /// Mettre a jour la camera en fonction des entrees clavier/souris
    void UpdateCamera();

    /// Synchroniser la camera mise a jour avec le SBT sur le GPU
    /// \param[in] data Les valeurs de camera a mettre a jour
    void SyncCameraToSbt(device::CameraData& data);

    /// Redimensionne les images apres un redimensionnement de la fenetre
    /// \param[in] outputBuffer Pointeur vers l'image generee
    /// \param[in] params Reference vers les params globaux de l'execution
    void ResizeCUDABuffer(sutil::CUDAOutputBuffer<uchar4>* outputBuffer, device::Params& params);

    /// Lance une execution d'OptiX et genere une image
    /// \param[out] outputBuffer Image generee
    /// \param[in] params Reference vers les parametres globaux de l'execution
    /// \param[in] d_params Pointeur vers les parametres tels qu'ils seront pousses sur le GPU
    void LaunchFrame(sutil::CUDAOutputBuffer<uchar4>* outputBuffer, device::Params& params, device::Params* d_params);

    /// Initialiser les callbacks de GLFW
    /// \param[in] window Pointeur vers la fenetre GLFW
    /// \param[in] state Le RendererState qui sera passe entre les appels de GLFW 
    /// (contenant des informations sur l'etat de l'execution)
    void InitGLFWCallbacks(GLFWwindow* window, RendererState* state);

    /// Detruire les variables a la destruction de l'objet Renderer
    void CleanUp();
};
} // namespace host
} // namespace engine