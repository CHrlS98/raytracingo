#include <renderer.h>

#include <sutil/Exception.h>
#include <sutil/Camera.h>
#include <sutil/GLDisplay.h>
#include <optix_stubs.h>
#include <sutil/sutil.h>

#include <cuda_runtime.h>

#include <iostream>
#include <iomanip>
#include <memory>

#include <GLFW/glfw3.h>
#include <glad/glad.h>
#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>

namespace engine
{
namespace host
{
namespace
{
const char* RAY_GEN_PROGRAM = "__raygen__rg";
const char* MISS_PROGRAM = "__miss__ms";
const char* CLOSEST_HIT_RADIANCE_PROGRAM = "__closesthit__ch";
const char* CLOSEST_HIT_LIGHT_PROGRAM = "__closesthit__light";
const char* CLOSEST_HIT_OCCLUSION_PROGRAM = "__closesthit__full_occlusion";
const char* PARAMS_STRUCT_NAME = "params";
const char* KERNEL_CUDA_NAME = "kernel.cu";
const float MOVEMENT_SPEED = 0.2f;

static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    RendererState* state = static_cast<RendererState*>(glfwGetWindowUserPointer(window));
    if (state)
    {
        double xpos;
        double ypos;
        glfwGetCursorPos(window, &xpos, &ypos);
        if (action == GLFW_PRESS)
        {
            state->mouseButton = button;
            state->trackball->startTracking(static_cast<int>(xpos), static_cast<int>(ypos));
        }
        else
        {
            state->mouseButton = -1;
        }
    }
}

static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
    RendererState* state = static_cast<RendererState*>(glfwGetWindowUserPointer(window));
    if (state)
    {
        if (state->mouseButton == GLFW_MOUSE_BUTTON_RIGHT)
        {
            state->trackball->setViewMode(sutil::Trackball::LookAtFixed);
            state->trackball->updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), state->params->image_width, state->params->image_height);
            state->cameraChangedFlag = true;
        }
        else if (state->mouseButton == GLFW_MOUSE_BUTTON_LEFT)
        {
            state->trackball->setViewMode(sutil::Trackball::EyeFixed);
            state->trackball->updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), state->params->image_width, state->params->image_height);
            state->cameraChangedFlag = true;
        }
        else if (state->mouseButton == GLFW_MOUSE_BUTTON_MIDDLE)
        {
            if (static_cast<double>(state->trackball->GetPrevPosY()) < ypos)
            {
                state->trackball->moveUp(MOVEMENT_SPEED);
                state->cameraChangedFlag = true;
            }
            else if (static_cast<double>(state->trackball->GetPrevPosY()) > ypos)
            {
                state->trackball->moveDown(MOVEMENT_SPEED);
                state->cameraChangedFlag = true;
            }
            state->trackball->startTracking(static_cast<int>(xpos), static_cast<int>(ypos));
        }
    }
}

static void windowSizeCallback(GLFWwindow* window, int32_t res_x, int32_t res_y)
{
    RendererState* state = static_cast<RendererState*>(glfwGetWindowUserPointer(window));
    if (state)
    {
        if (res_x > 0 && res_y > 0 && res_x != state->params->image_width && res_y != state->params->image_height)
        {
            state->params->image_width = res_x;
            state->params->image_height = res_y;
            state->cameraChangedFlag = true;
            state->windowResizeFlag = true;
        }
    }
}

static void keyCallback(GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t mods)
{
    RendererState* state = static_cast<RendererState*>(glfwGetWindowUserPointer(window));
    if (state)
    {
        if ((key == GLFW_KEY_RIGHT || key == GLFW_KEY_D) && (action == GLFW_REPEAT || action == GLFW_PRESS))
        {
            state->trackball->moveRight(MOVEMENT_SPEED);
            state->cameraChangedFlag = true;
        }
        if ((key == GLFW_KEY_LEFT || key == GLFW_KEY_A) && (action == GLFW_REPEAT || action == GLFW_PRESS))
        {
            state->trackball->moveLeft(MOVEMENT_SPEED);
            state->cameraChangedFlag = true;
        }
        if ((key == GLFW_KEY_DOWN || key == GLFW_KEY_S) && (action == GLFW_REPEAT || action == GLFW_PRESS) && mods != GLFW_MOD_CONTROL)
        {
            state->trackball->moveBackward(MOVEMENT_SPEED);
            state->cameraChangedFlag = true;
        }
        if ((key == GLFW_KEY_UP|| key == GLFW_KEY_W) && (action == GLFW_REPEAT || action == GLFW_PRESS))
        {
            state->trackball->moveForward(MOVEMENT_SPEED);
            state->cameraChangedFlag = true;
        }
        if (key == GLFW_KEY_S && action == GLFW_PRESS && mods == GLFW_MOD_CONTROL && !state->sendSaveRequest)
        {
            state->sendSaveRequest = true;
        }
    }
}

static void scrollCallback(GLFWwindow* window, double xscroll, double yscroll)
{
    RendererState* state = static_cast<RendererState*>(glfwGetWindowUserPointer(window));
    if (state)
    {
        state->trackball->wheelEvent(static_cast<int>(yscroll));
        state->cameraChangedFlag = true;
    }
}
} // namespace

static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*callbackdata */)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
}

Renderer::Renderer(std::shared_ptr<Scene> scene, RenderMode renderMode, int sqrtSamplePerPixel, bool useAmbiantCoeff)
    : m_optixContext(nullptr)
    , m_module(nullptr)
    , m_moduleCompileOptions({})
    , m_pipeline(nullptr)
    , m_pipelineCompileOptions({})
    , m_pipelineLinkOptions({})
    , m_cudaContext(nullptr)
    , m_cudaStream(nullptr)
    , m_cudaDeviceProperties({})
    , m_programs()
    , m_shaderBindingTable({})
    , m_deviceGasOutputBuffer(0)
    , m_traversableHandle(0)
    , m_scene(scene)
    , m_state()
    , m_renderMode(renderMode)
    , m_sqrtSamplePerPixel(sqrtSamplePerPixel)
    , m_useAmbientCoefficient(useAmbiantCoeff)
{
    Initialize();
}

Renderer::~Renderer()
{
    CleanUp();
}

void Renderer::Initialize()
{
    InitOptix();
    CreateContext();
    CreateModule();
    
    CreateRayGen();
    CreateMiss();
    CreateShapes();

    CreatePipeline();
}

void Renderer::InitOptix()
{
    // Initialisation de CUDA
    CUDA_CHECK(cudaFree(0));

    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0)
    {
        throw std::runtime_error("RayTracinGO: no CUDA capable devices found!");
    }
    std::cout << "RayTracinGO: found " << numDevices << " CUDA devices" << std::endl;

    // Initialisation de OptiX, charge tous les points d'entree
    OPTIX_CHECK(optixInit());
}

void Renderer::CreateContext()
{
    const int deviceID = 0;
    CUDA_CHECK(cudaSetDevice(deviceID));
    CUDA_CHECK(cudaStreamCreate(&m_cudaStream));

    cudaGetDeviceProperties(&m_cudaDeviceProperties, deviceID);
    std::cout << "RayTracinGO: running on device: " << m_cudaDeviceProperties.name << std::endl;

    // Associe un contexte CUDA et le GPU sur lequel il existe avec le contexte OptiX
    m_cudaContext = nullptr; // null means take the current context

    // Specifier les options pour ce contexte
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;

    // Creer le contexte OptiX sur le GPU
    OPTIX_CHECK(optixDeviceContextCreate(m_cudaContext, &options, &m_optixContext));
}

void Renderer::CreateModule()
{
    m_moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;     // Set to 0 for no explicit limit
    m_moduleCompileOptions.optLevel = OptixCompileOptimizationLevel::OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    m_moduleCompileOptions.debugLevel = OptixCompileDebugLevel::OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

    m_pipelineCompileOptions.usesMotionBlur = false;

    // nividia doc: "This option is important to ensure we compile code 
    // which is optimal for our scene hierarchy. We use a single GAS no 
    // instancing or multi-level hierarchies"
    m_pipelineCompileOptions.traversableGraphFlags = OptixTraversableGraphFlags::OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;

    // Notre application sur le GPU utilise 5 valeurs de payload et 3 valeurs d'attributs
    // Les payload servent a passer des donnees en retour a OptiXTrace
    // tandis que les attributes servent a passer des donnees entre les programmes d'intersection
    // et les programmes de type closest_hit/any_hit
    m_pipelineCompileOptions.numPayloadValues = 5;      // How much storage, in 32b words, to make available for the payload, [0..8]
    m_pipelineCompileOptions.numAttributeValues = 3;    // How much storage, in 32b words, to make available for the attributes , [2..8]
    m_pipelineCompileOptions.exceptionFlags = OptixExceptionFlags::OPTIX_EXCEPTION_FLAG_NONE;  // TODO: should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;

    // Specifie le nom de la structure de parametres globaux du programme
    m_pipelineCompileOptions.pipelineLaunchParamsVariableName = PARAMS_STRUCT_NAME;

    // Specifie le nom du fichier contenant le code CUDA a compiler
    const std::string ptx = sutil::getPtxString(OPTIX_SAMPLE_NAME, KERNEL_CUDA_NAME);

    char log[2048];
    size_t sizeof_log = sizeof(log);

    // Creer le module
    OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
        m_optixContext,
        &m_moduleCompileOptions,
        &m_pipelineCompileOptions,
        ptx.c_str(),
        ptx.size(),
        log,
        &sizeof_log,
        &m_module
    ));
}

OptixProgramGroup Renderer::CreateRayGenPrograms() const
{
    // Creer le program group pour la generation de rayons initiale
    OptixProgramGroup rayGenerationProgram;
    OptixProgramGroupOptions rayGenerationOptions = {};
    OptixProgramGroupDesc rayGenerationDesc = {};

    rayGenerationDesc.kind = OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    rayGenerationDesc.raygen.module = m_module;
    rayGenerationDesc.raygen.entryFunctionName = RAY_GEN_PROGRAM;

    char log[2048];
    size_t sizeof_log = sizeof(log);

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        m_optixContext,
        &rayGenerationDesc,
        1,   // nombre de program groups
        &rayGenerationOptions,
        log,
        &sizeof_log,
        &rayGenerationProgram
    ));

    return rayGenerationProgram;
}

void Renderer::BuildRayGenRecords(CameraSbtRecord* cameraSbt, const size_t& recordsCount)
{
    // Alloue la memoire necessaire pour le record de la camera
    CUdeviceptr  deviceCameraRecord;
    const size_t cameraRecordSize = sizeof(CameraSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceCameraRecord), cameraRecordSize));

    // Copie le record depuis le CPU vers le GPU
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(deviceCameraRecord),
        cameraSbt,
        cameraRecordSize,
        cudaMemcpyKind::cudaMemcpyHostToDevice
    ));

    // Ajoute le record a la shader binding table
    m_shaderBindingTable.raygenRecord = deviceCameraRecord;
}

void Renderer::CreateRayGen()
{
    OptixProgramGroup rayGenerationProgram = CreateRayGenPrograms();
    m_programs.push_back(rayGenerationProgram);

    // Remplir la copie sur l'hote du record avec le header et les donnees
    std::shared_ptr<sutil::Camera> camera = m_scene->GetCamera();
    CameraSbtRecord cameraSbt;
    cameraSbt.data = {};
    cameraSbt.data.cam_eye = camera->eye();
    camera->UVWFrame(cameraSbt.data.camera_u, cameraSbt.data.camera_v, cameraSbt.data.camera_w);

    OPTIX_CHECK(optixSbtRecordPackHeader(rayGenerationProgram, &cameraSbt));

    BuildRayGenRecords(&cameraSbt, 1);
}

OptixProgramGroup Renderer::CreateMissPrograms() const
{
    char log[2048];
    size_t sizeof_log = sizeof(log);

    // Creer le 'miss program group'
    OptixProgramGroup missProgram;
    OptixProgramGroupOptions missOptions = {};

    OptixProgramGroupDesc missDesc = {};
    missDesc.kind = OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_MISS;
    missDesc.miss.module = m_module;
    missDesc.miss.entryFunctionName = MISS_PROGRAM;

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        m_optixContext,
        &missDesc,
        1,
        &missOptions,
        log,
        &sizeof_log,
        &missProgram
    ));

    return missProgram;
}

void Renderer::BuildMissRecords(MissSbtRecord* missRecordSbt, const size_t& recordsCount)
{
    // Alloue l'espace necessaire pour le 'miss record' sur le GPU
    CUdeviceptr deviceMissRecord;
    size_t missRecordSize = recordsCount *  sizeof(MissSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceMissRecord), missRecordSize));

    // Copie le contenu du record du CPU vers le GPU
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(deviceMissRecord),
        missRecordSbt,
        missRecordSize,
        cudaMemcpyKind::cudaMemcpyHostToDevice
    ));

    // Specifier combien de 'record' sont sur le GPU et comment ils sont organises en memoire
    m_shaderBindingTable.missRecordBase = deviceMissRecord;
    m_shaderBindingTable.missRecordStrideInBytes = static_cast<unsigned int>(missRecordSize);
    m_shaderBindingTable.missRecordCount = static_cast<unsigned int>(recordsCount);
}

void Renderer::CreateMiss()
{
    OptixProgramGroup missProgram = CreateMissPrograms();
    m_programs.push_back(missProgram);

    // Remplir la copie sur l'hote du record avec le header et les donnees
    MissSbtRecord missRecordSbt;
    const glm::vec3& bg = m_scene->GetBackgroundColor();
    missRecordSbt.data = { bg.r, bg.g, bg.b };
    OPTIX_CHECK(optixSbtRecordPackHeader(missProgram, &missRecordSbt));

    BuildMissRecords(&missRecordSbt, 1);
}

void Renderer::CreateShapes()
{
    const std::vector<std::shared_ptr<Shape>> shapes = m_scene->GetShapes();
    const size_t& nbObjects = static_cast<size_t>(m_scene->GetNbObjects());

    // Initialisation du tableau des volumes englobants (axis-aligned bounding box)
    OptixAabb* aabb = new OptixAabb[nbObjects];
    uint32_t* aabbInputFlags = new uint32_t[nbObjects];
    uint32_t* sbtIndex = new uint32_t[nbObjects];

    // Initialisation des HitGroupRecords
    const size_t recordsCount = device::RayType::RAY_TYPE_COUNT * nbObjects;
    HitGroupSbtRecord* hitgroupRecords = new HitGroupSbtRecord[recordsCount];

    int i = 0;
    for (std::shared_ptr<Shape> shape : shapes)
    {
        for (Primitive primitive : shape->GetPrimitives())
        {
            // Creer l'AABB
            aabb[i] = primitive.GetAabb();
            aabbInputFlags[i] = OptixGeometryFlags::OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
            sbtIndex[i] = i;

            // Creer le program group associe au rayons calculant la radiance
            OptixProgramGroup radianceProgram = CreateHitGroupProgram(primitive.GetIntersectionProgram(), device::RAY_TYPE_RADIANCE);
            m_programs.push_back(radianceProgram);

            // Creer les sbt records associes a la radiance
            primitive.CopyToDevice(hitgroupRecords[device::RAY_TYPE_COUNT * i + device::RAY_TYPE_RADIANCE].data);
            OPTIX_CHECK(optixSbtRecordPackHeader(radianceProgram, &hitgroupRecords[device::RAY_TYPE_COUNT * i + device::RAY_TYPE_RADIANCE]));

            // Creer le program group associe au rayons d'ombre
            OptixProgramGroup occlusionProgram = CreateHitGroupProgram(primitive.GetIntersectionProgram(), device::RAY_TYPE_OCCLUSION);
            m_programs.push_back(occlusionProgram);

            // Creer les sbt records associes a l'occlusion
            primitive.CopyToDevice(hitgroupRecords[device::RAY_TYPE_COUNT * i + device::RAY_TYPE_OCCLUSION].data);
            OPTIX_CHECK(optixSbtRecordPackHeader(occlusionProgram, &hitgroupRecords[device::RAY_TYPE_COUNT * i + device::RAY_TYPE_OCCLUSION]));
            ++i;
        }
    }

    // Construit la structure accelerante sur le GPU
    BuildAccelerationStructure(aabb, aabbInputFlags, sbtIndex, nbObjects);

    // Contruit les 'hit group records' sur le GPU
    BuildHitGroupRecords(hitgroupRecords, recordsCount);

    delete[] aabb;
    delete[] aabbInputFlags;
    delete[] sbtIndex;
    delete[] hitgroupRecords;
}

OptixProgramGroup Renderer::CreateHitGroupProgram(const char* intersectionProgram, device::RayType type) const
{
    OptixProgramGroup program;
    OptixProgramGroupOptions options = {}; // No options yet
    OptixProgramGroupDesc desc = {};

    char log[2048];
    size_t sizeof_log = sizeof(log);

    switch (type)
    {
    case device::RAY_TYPE_RADIANCE:
        desc.kind = OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

        // Parametres du programme closest_hit
        desc.hitgroup.moduleCH = m_module;
        desc.hitgroup.entryFunctionNameCH = CLOSEST_HIT_RADIANCE_PROGRAM;

        // Parametres du programme any_hit
        desc.hitgroup.moduleAH = nullptr;
        desc.hitgroup.entryFunctionNameAH = nullptr;

        // Parametres du programme d'intersection
        desc.hitgroup.moduleIS = m_module;
        desc.hitgroup.entryFunctionNameIS = intersectionProgram;
        break;
    case device::RAY_TYPE_OCCLUSION:
        desc.kind = OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

        // Parametres du programme closest_hit
        desc.hitgroup.moduleCH = m_module;
        desc.hitgroup.entryFunctionNameCH = CLOSEST_HIT_OCCLUSION_PROGRAM;

        // Parametres du programme any_hit
        desc.hitgroup.moduleAH = nullptr;
        desc.hitgroup.entryFunctionNameAH = nullptr;

        // Parametres du programme d'intersection
        desc.hitgroup.moduleIS = m_module;
        desc.hitgroup.entryFunctionNameIS = intersectionProgram;
        break;
    default:
        throw std::runtime_error("RayTracinGO: Invalid ray type!");
        break;
    }

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        m_optixContext,
        &desc,
        1,
        &options,
        log,
        &sizeof_log,
        &program
    ));

    return program;
}

void Renderer::BuildAccelerationStructure(
    OptixAabb* aabb, 
    uint32_t* aabbInputFlags, 
    uint32_t* sbtIndex,
    const size_t& nbObjects)
{
    // Ce code provient en majeure partie de optixSphere tutorial de OptiX 7 SDK
    // Specifier les options du build
    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OptixBuildFlags::OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelOptions.operation = OptixBuildOperation::OPTIX_BUILD_OPERATION_BUILD;

    CUdeviceptr deviceAabbBuffer;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceAabbBuffer), nbObjects * sizeof(OptixAabb)));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(deviceAabbBuffer),
        aabb,
        nbObjects * sizeof(OptixAabb),
        cudaMemcpyKind::cudaMemcpyHostToDevice
    ));

    CUdeviceptr deviceSbtIndex;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceSbtIndex), nbObjects * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(deviceSbtIndex),
        sbtIndex,
        nbObjects * sizeof(uint32_t),
        cudaMemcpyKind::cudaMemcpyHostToDevice));
    
    // Remplir l'OptixBuildInput avec les aabb et les informations sur la taille et le type des donnees
    OptixBuildInput aabbInput = {};
    aabbInput.type = OptixBuildInputType::OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    aabbInput.aabbArray.aabbBuffers = &deviceAabbBuffer;
    aabbInput.aabbArray.numPrimitives = static_cast<int>(nbObjects);
    aabbInput.aabbArray.flags = aabbInputFlags;
    aabbInput.aabbArray.numSbtRecords = static_cast<int>(nbObjects);
    aabbInput.aabbArray.sbtIndexOffsetBuffer = deviceSbtIndex;
    aabbInput.aabbArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
    aabbInput.aabbArray.primitiveIndexOffset = 0;

    // Requete a OptiX pour la memoire requise pour notre structure accelerante
    OptixAccelBufferSizes gasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(m_optixContext, &accelOptions, &aabbInput, 1, &gasBufferSizes));

    // Alloue la memoire necessaire sur le GPU
    CUdeviceptr deviceTempBufferGas;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceTempBufferGas), gasBufferSizes.tempSizeInBytes));

    // Sortie non-compressee
    CUdeviceptr deviceBufferTempOutputGasAndCompactedSize;
    size_t compactedSizeOffset = roundUp<size_t>(gasBufferSizes.outputSizeInBytes, 8ull);
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&deviceBufferTempOutputGasAndCompactedSize),
        compactedSizeOffset + 8 // c'est ainsi que c'est fait dans l'exemple
    ));

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OptixAccelPropertyType::OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = (CUdeviceptr)((char*)deviceBufferTempOutputGasAndCompactedSize + compactedSizeOffset);

    // Contruire le GAS ( geometry acceleration structure)
    OPTIX_CHECK(optixAccelBuild(m_optixContext,
        0,                  // CUDA stream
        &accelOptions,
        &aabbInput,
        1,
        deviceTempBufferGas,
        gasBufferSizes.tempSizeInBytes,
        deviceBufferTempOutputGasAndCompactedSize,
        gasBufferSizes.outputSizeInBytes,
        &m_traversableHandle,
        &emitProperty,
        1
    ));

    // Libere l'espace additionnel utilise pour le build
    CUDA_CHECK(cudaFree((void*)deviceTempBufferGas));
    CUDA_CHECK(cudaFree((void*)deviceAabbBuffer));

    // Optimization (compression de la structure)
    size_t compactedGasSize;
    CUDA_CHECK(cudaMemcpy(&compactedGasSize,
        (void*)emitProperty.result,
        sizeof(size_t),
        cudaMemcpyKind::cudaMemcpyDeviceToHost));

    if (compactedGasSize < gasBufferSizes.outputSizeInBytes)
    {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_deviceGasOutputBuffer), compactedGasSize));
        OPTIX_CHECK(optixAccelCompact(m_optixContext, 0, m_traversableHandle, m_deviceGasOutputBuffer, compactedGasSize, &m_traversableHandle));
        CUDA_CHECK(cudaFree((void*)deviceBufferTempOutputGasAndCompactedSize));
    }
    else
    {
        m_deviceGasOutputBuffer = deviceBufferTempOutputGasAndCompactedSize;
    }
}

void Renderer::CreatePipeline()
{
    m_pipelineLinkOptions = {};
    m_pipelineLinkOptions.maxTraceDepth = 3; // Niveau maximum de recursivite. Le maximum que supporte OptiX est 31.
    m_pipelineLinkOptions.debugLevel = OptixCompileDebugLevel::OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    m_pipelineLinkOptions.overrideUsesMotionBlur = false;

    char log[2048];
    size_t sizeof_log = sizeof(log);

    // Cree le pipeline OptiX pour nos program groups
    OPTIX_CHECK_LOG(optixPipelineCreate(
        m_optixContext,
        &m_pipelineCompileOptions,
        &m_pipelineLinkOptions,
        m_programs.data(),
        static_cast<unsigned int>(m_programs.size()),
        log,
        &sizeof_log,
        &m_pipeline
    ));
}

void Renderer::BuildHitGroupRecords(HitGroupSbtRecord* hitgroupRecords, const size_t& recordsCount)
{
    // Copie le record du CPU vers le GPU
    CUdeviceptr deviceHitGroupRecord;
    size_t      hitGroupRecordSize = sizeof(HitGroupSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceHitGroupRecord), recordsCount * hitGroupRecordSize));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(deviceHitGroupRecord),
        hitgroupRecords,
        recordsCount * hitGroupRecordSize,
        cudaMemcpyKind::cudaMemcpyHostToDevice
    ));

    // Specifie combien on a de records et comment ils sont organises en memoire
    m_shaderBindingTable.hitgroupRecordBase = deviceHitGroupRecord;
    m_shaderBindingTable.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    m_shaderBindingTable.hitgroupRecordCount = static_cast<unsigned int>(recordsCount);
}

void Renderer::WriteLights(device::Params& params)
{
    const std::vector<SurfaceLight>& surfaceLights = m_scene->GetSurfaceLights();
    const size_t& nbSurfaceLights = surfaceLights.size();

    // Lumiere de surface
    for (size_t i = 0; i < nbSurfaceLights && i < device::Params::MAX_LIGHTS; ++i)
    {
        const glm::vec3& lightCorner = surfaceLights[i].GetCorner();
        const glm::vec3& v1 = surfaceLights[i].GetV1();
        const glm::vec3& v2 = surfaceLights[i].GetV2();
        const glm::vec3& normal = surfaceLights[i].GetNormal();
        const glm::vec3& color = surfaceLights[i].GetColor();
        params.surfaceLights[i].corner = { lightCorner.x, lightCorner.y, lightCorner.z };
        params.surfaceLights[i].v1 = { v1.x, v1.y, v1.z };
        params.surfaceLights[i].v2 = { v2.x, v2.y, v2.z };
        params.surfaceLights[i].normal = { normal.x, normal.y, normal.z };
        params.surfaceLights[i].color = { color.x, color.y, color.z };
        params.surfaceLights[i].falloff = surfaceLights[i].GetFalloff();
    }

    params.nbSurfaceLights = static_cast<int>(nbSurfaceLights);
}

void Renderer::Update(sutil::CUDAOutputBuffer<uchar4>* outputBuffer, device::Params& params, bool firstLaunch)
{
    // Reinitialiser le compteur de frame
    params.frameCount = m_state.cameraChangedFlag || m_state.windowResizeFlag || firstLaunch ? 0 : params.frameCount + 1;

    if (m_state.sendSaveRequest)
    {
        std::cout << "Saving to file output.ppm" << std::endl;
        sutil::ImageBuffer buffer;
        buffer.data = outputBuffer->getHostPointer();
        buffer.width = outputBuffer->width();
        buffer.height = outputBuffer->height();
        buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
        sutil::displayBufferFile("output.ppm", buffer, false);
        m_state.sendSaveRequest = false;
        std::cout << "Save complete" << std::endl;
    }

    // Mettre a jour la camera
    UpdateCamera();
    // Mettre a jour les dimensions de l'image
    ResizeCUDABuffer(outputBuffer, params);
}

void Renderer::UpdateCamera()
{
    if (!m_state.cameraChangedFlag)
    {
        return;
    }
    m_state.cameraChangedFlag = false;
    std::shared_ptr<sutil::Camera> camera = m_scene->GetCamera();
    camera->setAspectRatio(static_cast<float>(m_state.params->image_width) / static_cast<float>(m_state.params->image_height));

    device::CameraData cameraData;
    cameraData.cam_eye = m_scene->GetCamera()->eye();
    camera->UVWFrame(cameraData.camera_u, cameraData.camera_v, cameraData.camera_w);
    SyncCameraToSbt(cameraData);
}

void Renderer::SyncCameraToSbt(device::CameraData& data)
{
    CameraSbtRecord cameraSbt;
    optixSbtRecordPackHeader(m_programs[0], &cameraSbt);
    cameraSbt.data = data;

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(m_shaderBindingTable.raygenRecord),
        &cameraSbt,
        sizeof(CameraSbtRecord),
        cudaMemcpyHostToDevice
    ));
}

void Renderer::ResizeCUDABuffer(sutil::CUDAOutputBuffer<uchar4>* outputBuffer, device::Params& params)
{
    if (!m_state.windowResizeFlag)
    {
        return;
    }
    m_state.windowResizeFlag = false;
    outputBuffer->resize(m_state.params->image_width, m_state.params->image_height);
    // Realloue le buffer d'accumulation
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(params.accum_buffer)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&params.accum_buffer),
        params.image_width* params.image_height * sizeof(float4)
    ));
}

void Renderer::LaunchFrame(sutil::CUDAOutputBuffer<uchar4>* outputBuffer, device::Params& params, device::Params* d_params)
{
    uchar4* result_buffer_data = outputBuffer->map();
    params.image = result_buffer_data;
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_params),
        &params,
        sizeof(device::Params),
        cudaMemcpyHostToDevice,
        m_cudaStream
    ));

    // Lancer une execution sur le GPU maintenant, en specifiant le pipeline, les parametres globaux et la SBT
    OPTIX_CHECK(optixLaunch(
        m_pipeline,
        m_cudaStream,
        reinterpret_cast<CUdeviceptr>(d_params),
        sizeof(device::Params),
        &m_shaderBindingTable,
        params.image_width,
        params.image_height,
        1 /*depth=*/));

    // Unmapper l'image rendue
    outputBuffer->unmap();
    CUDA_SYNC_CHECK();
}

void Renderer::InitGLFWCallbacks(GLFWwindow* window, RendererState* state)
{
    glfwSetInputMode(window, GLFW_STICKY_KEYS, false);

    // Donner acces aux callback au params
    glfwSetWindowUserPointer(window, state);
    glfwSetWindowSizeCallback(window, windowSizeCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetScrollCallback(window, scrollCallback);
}

void Renderer::Display()
{
    unsigned int width = m_scene->GetCameraWidth();
    unsigned int height = m_scene->GetCameraHeight();

    // Initialiser les parametres
    device::Params params;
    params.image = nullptr;
    params.image_width = width;
    params.image_height = height;
    params.sqrtSamplePerPixel = m_sqrtSamplePerPixel;
    params.useAmbientLight = m_useAmbientCoefficient;
    params.handle = m_traversableHandle;
    params.maxTraceDepth = m_pipelineLinkOptions.maxTraceDepth;
    params.enablePathTracing = m_renderMode == RenderMode::PATH_TRACING;
    params.frameCount = 0;
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&params.accum_buffer),
        params.image_width* params.image_height * sizeof(float4)
    ));
    WriteLights(params);

    // Transfere les parametres au GPU
    device::Params* d_params = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(device::Params)));

    // Initialise la fenetre et l'UI
    GLFWwindow* window = sutil::initGLFW("RayTracinGO", width, height);
    sutil::initGL();
    sutil::initImGui(window);

    sutil::CUDAOutputBuffer<uchar4>* outputBuffer = new sutil::CUDAOutputBuffer<uchar4>(sutil::CUDAOutputBufferType::CUDA_DEVICE, width, height);
    outputBuffer->setStream(m_cudaStream);
    
    sutil::GLDisplay display;
    int framebuf_res_x = 0;
    int framebuf_res_y = 0;
    unsigned int frameCount = 0;

    m_state.params = &params;
    m_state.trackball.reset(new sutil::Trackball);
    m_state.trackball->setCamera(m_scene->GetCamera().get());
    m_state.trackball->setMoveSpeed(10.0f);
    m_state.trackball->setReferenceFrame(make_float3(1.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 1.0f), make_float3(0.0f, 1.0f, 0.0f));
    m_state.trackball->setGimbalLock(true);
    m_state.cameraChangedFlag = false;
    m_state.windowResizeFlag = false;
    m_state.sendSaveRequest = false;
    m_state.mouseButton = -1;
    InitGLFWCallbacks(window, &m_state);

    bool firstLaunch = true;
    do
    {
        glfwPollEvents();

        Update(outputBuffer, params, firstLaunch);
        
        // Lancer les rayons pour un frame
        LaunchFrame(outputBuffer, params, d_params);
        firstLaunch = false;

        // Affichage du frame genere
        glfwGetFramebufferSize(window, &framebuf_res_x, &framebuf_res_y);
        display.display(outputBuffer->width(), outputBuffer->height(), framebuf_res_x, framebuf_res_y, outputBuffer->getPBO());

        // Affichage du framerate
        sutil::beginFrameImGui();
        sutil::displayFPS(frameCount++);
        sutil::endFrameImGui();

        glfwSwapBuffers(window);

    } while (!glfwWindowShouldClose(window));

    delete outputBuffer;
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(params.accum_buffer)));
    glfwDestroyWindow(window);
    glfwTerminate();
}

void Renderer::CleanUp()
{
    // Libere la memoire utilisee par le programme CUDA
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_shaderBindingTable.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_shaderBindingTable.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_shaderBindingTable.hitgroupRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_deviceGasOutputBuffer)));

    OPTIX_CHECK(optixPipelineDestroy(m_pipeline));
    for (OptixProgramGroup program : m_programs)
    {
        OPTIX_CHECK(optixProgramGroupDestroy(program));
    }
    OPTIX_CHECK(optixModuleDestroy(m_module));
    OPTIX_CHECK(optixDeviceContextDestroy(m_optixContext));
}

} // namespace host
} // namespace engine
