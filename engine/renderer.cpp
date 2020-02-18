#include <renderer.h>

#include <sutil/Exception.h>
#include <sutil/Camera.h>
#include <optix_stubs.h>
#include <sutil/sutil.h>

#include <cuda_runtime.h>

#include <sphere.h>
#include <plane.h>

#include <iostream>
#include <iomanip>
#include <memory>

namespace engine
{

static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*callbackdata */)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
}

Renderer::Renderer(std::shared_ptr<Scene> scene, const int width, const int height)
    : m_windowWidth(width)
    , m_windowHeight(height)
    , m_optixContext(nullptr)
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
    , m_outputBuffer(nullptr)
    , m_deviceGasOutputBuffer(0)
    , m_traversableHandle(0)
    , m_scene(scene)
{
    m_outputBuffer.reset(
        new sutil::CUDAOutputBuffer<uchar4>(
            sutil::CUDAOutputBufferType::CUDA_DEVICE, 
            m_windowWidth, 
            m_windowHeight
        )
    );

    InitOptix();
    CreateContext();
    CreateModule();
    BuildAccelerationStructure();
    CreateRayGenerationPrograms();
    CreateMissPrograms();
    CreateHitGroupPrograms();
    CreatePipeline();
    BuildShaderBindingTable();
}

Renderer::~Renderer()
{
    CleanUp();
}

void Renderer::InitOptix()
{
    std::cout << "RayTracinGO: initializing OptiX ..." << std::endl;
    // Initialize CUDA with a no-op call to the CUDA runtime API
    CUDA_CHECK(cudaFree(0));

    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0)
    {
        throw std::runtime_error("RayTracinGO: no CUDA capable devices found!");
    }
    std::cout << "RayTracinGO: found " << numDevices << " CUDA devices" << std::endl;
    
    // Initialize the OptiX API, loading all API entry points
    OPTIX_CHECK(optixInit());

    std::cout << "RayTracinGO: successfully initialized OptiX" << std::endl;
}

void Renderer::CreateContext()
{
    std::cout << "RayTracinGO: creating OptiX context ..." << std::endl;
    const int deviceID = 0;
    CUDA_CHECK(cudaSetDevice(deviceID));
    CUDA_CHECK(cudaStreamCreate(&m_cudaStream));

    cudaGetDeviceProperties(&m_cudaDeviceProperties, deviceID);
    std::cout << "RayTracinGO: running on device: " << m_cudaDeviceProperties.name << std::endl;

    // Associate a CUDA context (and therefore a specific GPU) with this
    // device context
    m_cudaContext = nullptr; // null means take the current context

    // Specify options for this context. A good practice is to zero-
    // initialize all OptiX input struct to mark all fields as default, 
    // then to selectively override the fields to be used
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;

    // Create the optix context on the GPU
    OPTIX_CHECK(optixDeviceContextCreate(m_cudaContext, &options, &m_optixContext));
    std::cout << "RayTracinGO: successfully created OptiX context" << std::endl;
}

void Renderer::CreateModule()
{
    std::cout << "RayTracinGO: creating OptiX module ..." << std::endl;

    m_moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;     // Set to 0 for no explicit limit
    m_moduleCompileOptions.optLevel = OptixCompileOptimizationLevel::OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    m_moduleCompileOptions.debugLevel = OptixCompileDebugLevel::OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

    m_pipelineCompileOptions.usesMotionBlur = false;

    // This option is important to ensure we compile code which is optimal
    // for our scene hierarchy. We use a single GAS no instancing or
    // multi-level hierarchies
    m_pipelineCompileOptions.traversableGraphFlags = OptixTraversableGraphFlags::OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;

    // Our device code uses 3 payload registers (r,g,b output value)
    m_pipelineCompileOptions.numPayloadValues = 3;      // How much storage, in 32b words, to make available for the payload, [0..8]
    m_pipelineCompileOptions.numAttributeValues = 3;    // How much storage, in 32b words, to make available for the attributes , [2..8]
    m_pipelineCompileOptions.exceptionFlags = OptixExceptionFlags::OPTIX_EXCEPTION_FLAG_NONE;  // TODO: should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;

    // This is the name of the param struct variable in our device code
    m_pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

    char log[2048];
    size_t sizeof_log = sizeof(log);
    const std::string ptx = sutil::getPtxString(OPTIX_SAMPLE_NAME, "kernel.cu");    // 1 PTX = 1 module ??

    // Create the module from PTX file
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

void Renderer::CreateRayGenerationPrograms()
{
    std::cout << "RayTracinGO: creating the Ray Generation programs ..." << std::endl;

    // Create ray generation group
    OptixProgramGroup rayGenerationProgram;
    OptixProgramGroupOptions rayGenerationOptions = {}; // No options yet
    OptixProgramGroupDesc rayGenerationDesc = {};

    rayGenerationDesc.kind = OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_RAYGEN;

    // Ray generation device settings
    rayGenerationDesc.raygen.module = m_module;
    rayGenerationDesc.raygen.entryFunctionName = "__raygen__rg";

    char log[2048];
    size_t sizeof_log = sizeof(log);

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        m_optixContext,
        &rayGenerationDesc,
        1,   // num program groups
        &rayGenerationOptions,
        log,
        &sizeof_log,
        &rayGenerationProgram
    ));
    m_programs.push_back(rayGenerationProgram);
}

void Renderer::CreateMissPrograms()
{
    std::cout << "RayTracinGO: creating the Miss programs ..." << std::endl;

    // Create miss program group
    OptixProgramGroup radianceMissProgram;
    OptixProgramGroupOptions missOptions = {}; // No options yet
    OptixProgramGroupDesc missDesc = {};
    missDesc.kind = OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_MISS;

    // Miss group device settings
    missDesc.miss.module = m_module;
    missDesc.miss.entryFunctionName = "__miss__ms";

    char log[2048];
    size_t sizeof_log = sizeof(log);

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        m_optixContext,
        &missDesc,
        1,   // num program groups
        &missOptions,
        log,
        &sizeof_log,
        &radianceMissProgram
    ));
    m_programs.push_back(radianceMissProgram);

    OptixProgramGroup occlusionMissProgram;
    missDesc.miss.module = nullptr;
    missDesc.miss.entryFunctionName = nullptr;

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        m_optixContext,
        &missDesc,
        1,   // num program groups
        &missOptions,
        log,
        &sizeof_log,
        &occlusionMissProgram
    ));
    m_programs.push_back(occlusionMissProgram);

}

void Renderer::CreateHitGroupPrograms()
{
    std::cout << "RayTracinGO: creating HitGroup programs ..." << std::endl;

    // Create hit group
    std::vector<std::shared_ptr<IShape>>& shapes = m_scene->GetShapes();
    for (std::shared_ptr<IShape> shape : shapes)
    {
        OptixProgramGroup radianceProgram;
        OptixProgramGroupOptions radianceOptions = {}; // No options yet
        OptixProgramGroupDesc radianceDesc = {};
        radianceDesc.kind = OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

        // Closest hit device settings
        radianceDesc.hitgroup.moduleCH = m_module;
        radianceDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";

        // Any hit device settings
        radianceDesc.hitgroup.moduleAH = nullptr;
        radianceDesc.hitgroup.entryFunctionNameAH = nullptr;

        // Intersection device settings
        radianceDesc.hitgroup.moduleIS = m_module;
        radianceDesc.hitgroup.entryFunctionNameIS = shape->GetIntersectionProgram();

        char log[2048];
        size_t sizeof_log = sizeof(log);

        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            m_optixContext,
            &radianceDesc,
            1,   // num program groups
            &radianceOptions,
            log,
            &sizeof_log,
            &radianceProgram
        ));
        m_programs.push_back(radianceProgram);


        OptixProgramGroup occlusionProgram;
        OptixProgramGroupOptions occlusionOptions = {}; // No options yet
        OptixProgramGroupDesc occlusionDesc = {};
        occlusionDesc.kind = OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

        // Closest hit device settings
        occlusionDesc.hitgroup.moduleCH = m_module;
        occlusionDesc.hitgroup.entryFunctionNameCH = "__closesthit__full_occlusion";

        // Any hit device settings
        occlusionDesc.hitgroup.moduleAH = nullptr;
        occlusionDesc.hitgroup.entryFunctionNameAH = nullptr;

        // Intersection device settings
        occlusionDesc.hitgroup.moduleIS = m_module;
        occlusionDesc.hitgroup.entryFunctionNameIS = shape->GetIntersectionProgram();

        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            m_optixContext,
            &occlusionDesc,
            1,   // num program groups
            &occlusionOptions,
            log,
            &sizeof_log,
            &occlusionProgram
        ));
        m_programs.push_back(occlusionProgram);
    }
}

void Renderer::BuildAccelerationStructure()
{
    // Specify options for the build
    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OptixBuildFlags::OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelOptions.operation = OptixBuildOperation::OPTIX_BUILD_OPERATION_BUILD;

    // AABB build input
    // OptixAabb(minX, minY, minZ, maxX, maxY, maxZ)
    // axis-aligned bounding box
    const size_t nbObjects = m_scene->GetShapes().size();
    OptixAabb* aabb = new OptixAabb[nbObjects];
    CUdeviceptr deviceAaabbBuffer;

    int i = 0;
    std::vector<std::shared_ptr<IShape>> shapes = m_scene->GetShapes();
    for (std::shared_ptr<IShape> shape : shapes)
    {
        switch (shape->GetShapeType())
        {
        case ShapeType::SphereType:
        {
            std::shared_ptr<Sphere> sphere = std::dynamic_pointer_cast<Sphere>(shape);
            if (sphere)
            {
                aabb[i++] = sphere->GetAabb();
            }
            break;
        }
        case ShapeType::PlaneType:
        {
            std::shared_ptr<Plane> plane = std::dynamic_pointer_cast<Plane>(shape);
            if (plane)
            {
                aabb[i++] = plane->GetAabb();
            }
            break;
        }
        default:
            break;
        }
    }

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceAaabbBuffer), nbObjects * sizeof(OptixAabb)));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(deviceAaabbBuffer),
        aabb,
        nbObjects * sizeof(OptixAabb),
        cudaMemcpyKind::cudaMemcpyHostToDevice
    ));


    // Populate build input struct with our aabb (axis-aligned bounding box) as well as 
    // information about the sizes and type of our data
    OptixBuildInput aabbInput = {};

    aabbInput.type = OptixBuildInputType::OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    aabbInput.aabbArray.aabbBuffers = &deviceAaabbBuffer;
    aabbInput.aabbArray.numPrimitives = static_cast<int>(nbObjects);

    uint32_t* aabbInputFlags = new uint32_t[nbObjects];//{ OptixGeometryFlags::OPTIX_GEOMETRY_FLAG_NONE
    for (int i = 0; i < nbObjects; ++i)
    {
        aabbInputFlags[i] = OptixGeometryFlags::OPTIX_GEOMETRY_FLAG_NONE;
    }

    aabbInput.aabbArray.flags = aabbInputFlags;
    aabbInput.aabbArray.numSbtRecords = static_cast<int>(nbObjects); // plus tard nbObjects * RayCount

    uint32_t* sbtIndex = new uint32_t[nbObjects];
    for (int i = 0; i < nbObjects; ++i)
    {
        sbtIndex[i] = i;
    }

    CUdeviceptr deviceSbtIndex;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceSbtIndex), nbObjects * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(deviceSbtIndex),
        sbtIndex,
        nbObjects * sizeof(uint32_t),
        cudaMemcpyKind::cudaMemcpyHostToDevice));
    aabbInput.aabbArray.sbtIndexOffsetBuffer = deviceSbtIndex;
    aabbInput.aabbArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
    aabbInput.aabbArray.primitiveIndexOffset = 0;

    // Query OptiX for the memory requirements for our GAS
    OptixAccelBufferSizes gasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(m_optixContext, &accelOptions, &aabbInput, 1, &gasBufferSizes));

    // Allocate device memory for the scratch space buffer as well
    // as the GAS itself
    CUdeviceptr deviceTempBufferGas;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceTempBufferGas), gasBufferSizes.tempSizeInBytes));

    // non-compacted output
    CUdeviceptr deviceBufferTempOutputGasAndCompactedSize;
    size_t compactedSizeOffset = roundUp<size_t>(gasBufferSizes.outputSizeInBytes, 8ull);
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&deviceBufferTempOutputGasAndCompactedSize),
        compactedSizeOffset + 8
    ));

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OptixAccelPropertyType::OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = (CUdeviceptr)((char*)deviceBufferTempOutputGasAndCompactedSize + compactedSizeOffset);

    // Build the GAS ( geometry acceleration structure)
    OPTIX_CHECK(optixAccelBuild(m_optixContext,
        0,                  // CUDA stream
        &accelOptions,
        &aabbInput,
        1,                  // num build inputs
        deviceTempBufferGas,
        gasBufferSizes.tempSizeInBytes,
        deviceBufferTempOutputGasAndCompactedSize,
        gasBufferSizes.outputSizeInBytes,
        &m_traversableHandle,
        &emitProperty,      // emitted property list
        1                   // num emitted properties
    ));

    // Free scratch space used during the build
    CUDA_CHECK(cudaFree((void*)deviceTempBufferGas));
    CUDA_CHECK(cudaFree((void*)deviceAaabbBuffer));

    // Additionnal compaction steps
    size_t compactedGasSize;
    CUDA_CHECK(cudaMemcpy(&compactedGasSize,
        (void*)emitProperty.result,
        sizeof(size_t),
        cudaMemcpyKind::cudaMemcpyDeviceToHost));

    if (compactedGasSize < gasBufferSizes.outputSizeInBytes)
    {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_deviceGasOutputBuffer), compactedGasSize));

        // use handle as input and output
        OPTIX_CHECK(optixAccelCompact(m_optixContext, 0, m_traversableHandle, m_deviceGasOutputBuffer, compactedGasSize, &m_traversableHandle));

        CUDA_CHECK(cudaFree((void*)deviceBufferTempOutputGasAndCompactedSize));
    }
    else
    {
        m_deviceGasOutputBuffer = deviceBufferTempOutputGasAndCompactedSize;
    }

    delete[] aabb;
    delete[] aabbInputFlags;
    delete[] sbtIndex;
}

void Renderer::CreatePipeline()
{
    std::cout << "RayTracinGO: creating the pipeline ..." << std::endl;

    m_pipelineLinkOptions = {};
    m_pipelineLinkOptions.maxTraceDepth = 5; // Maximum trace recursion depth. The maximum is 31
    m_pipelineLinkOptions.debugLevel = OptixCompileDebugLevel::OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    m_pipelineLinkOptions.overrideUsesMotionBlur = false;

    char log[2048];
    size_t sizeof_log = sizeof(log);

    // Create OptiX pipeline for our program groups
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

void Renderer::BuildShaderBindingTable()
{
    std::cout << "RayTracinGO: building the shader binding table ..." << std::endl;
    int sbtIndex = 0;
    BuildRayGenerationRecords(sbtIndex);
    BuildMissRecords(sbtIndex);
    BuildHitGroupRecords(sbtIndex);
}

void Renderer::BuildRayGenerationRecords(int& sbtIndex)
{
    // Allocate the raygen record on the device
    CUdeviceptr  deviceCameraRecord;
    const size_t cameraRecordSize = sizeof(CameraSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceCameraRecord), cameraRecordSize));

    /// \todo Configurer la camera ailleurs
    sutil::Camera camera;
    camera.ConfigureCamera(m_windowWidth, m_windowHeight);

    // Populate host side copy of the record with header and data
    CameraSbtRecord cameraSbt;
    cameraSbt.data = {};
    cameraSbt.data.cam_eye = camera.eye();   // cam_eye name use in cuda file
    camera.UVWFrame(cameraSbt.data.camera_u, cameraSbt.data.camera_v, cameraSbt.data.camera_w);

    OPTIX_CHECK(optixSbtRecordPackHeader(m_programs[sbtIndex], &cameraSbt));
    sbtIndex++;

    // Now copy our host record to the device
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(deviceCameraRecord),
        &cameraSbt,
        cameraRecordSize,
        cudaMemcpyKind::cudaMemcpyHostToDevice
    ));

    // Specify how raygen record are packed in memory
    m_shaderBindingTable.raygenRecord = deviceCameraRecord;
}

void Renderer::BuildMissRecords(int& sbtIndex)
{
    // Allocate our miss record on the device
    CUdeviceptr deviceMissRecord;
    size_t      missRecordSize = sizeof(MissSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceMissRecord), RayType::RAY_TYPE_COUNT * missRecordSize));

    // Populate host-side copy of the record with header and data
    MissSbtRecord* missRecordSbt = new MissSbtRecord[RayType::RAY_TYPE_COUNT];
    for (int i = 0; i < RayType::RAY_TYPE_COUNT; ++i)
    {
        missRecordSbt[i].data = { 0.8f, 0.97f, 1.0f };
        OPTIX_CHECK(optixSbtRecordPackHeader(m_programs[sbtIndex], &missRecordSbt[i]));
        sbtIndex++;
    }

    // Now copy our host record to the device
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(deviceMissRecord),
        missRecordSbt,
        RayType::RAY_TYPE_COUNT * missRecordSize,
        cudaMemcpyKind::cudaMemcpyHostToDevice
    ));

    // Finally we specify how many records and how they are packed in memory
    m_shaderBindingTable.missRecordBase = deviceMissRecord;
    m_shaderBindingTable.missRecordStrideInBytes = sizeof(MissSbtRecord);
    m_shaderBindingTable.missRecordCount = RayType::RAY_TYPE_COUNT;

    delete[] missRecordSbt;
}

void Renderer::BuildHitGroupRecords(int& sbtIndex)
{
    const size_t recordsCount = RayType::RAY_TYPE_COUNT * m_scene->GetShapes().size();
    HitGroupSbtRecord* hitgroupRecords = new HitGroupSbtRecord[recordsCount];
    int i = 0;
    std::vector<std::shared_ptr<IShape>> shapes = m_scene->GetShapes();
    for (std::shared_ptr<IShape> shape : shapes)
    {
        switch (shape->GetShapeType())
        {
        case ShapeType::SphereType:
        {
            std::shared_ptr<Sphere> sphere = std::dynamic_pointer_cast<Sphere>(shape);
            if (sphere)
            {
                const glm::vec3& pos = sphere->GetWorldPosition();
                hitgroupRecords[i].data.geometry.sphere.radius = sphere->GetRadius();
                hitgroupRecords[i].data.geometry.sphere.position = { pos.x, pos.y, pos.z };
                hitgroupRecords[i].data.material.basicMaterial.ka = { 1.0f, 0.0f, 1.0f };
                hitgroupRecords[i].data.material.basicMaterial.kd = { 1.0f, 0.0f, 1.0f };
                hitgroupRecords[i].data.material.basicMaterial.ks = { 1.0f, 0.0f, 1.0f };
                hitgroupRecords[i].data.material.basicMaterial.alpha = 30.0f;
                OPTIX_CHECK(optixSbtRecordPackHeader(m_programs[sbtIndex], &hitgroupRecords[i]));
                ++sbtIndex;
                ++i;
                hitgroupRecords[i] = hitgroupRecords[i - 1];
                OPTIX_CHECK(optixSbtRecordPackHeader(m_programs[sbtIndex], &hitgroupRecords[i]));
                ++sbtIndex;
                ++i;
            }
            break;
        }
        case ShapeType::PlaneType:
        {
            std::shared_ptr<Plane> plane = std::dynamic_pointer_cast<Plane>(shape);
            if (plane)
            {
                const glm::vec3& pos = plane->GetWorldPosition();
                const glm::vec3& normal = plane->GetNormal();
                hitgroupRecords[i].data.geometry.plane.normal = { normal.x, normal.y, normal.z };
                hitgroupRecords[i].data.geometry.plane.position = { pos.x, pos.y, pos.z };
                hitgroupRecords[i].data.material.basicMaterial.ka = { 1.0f, 0.0f, 0.0f };
                hitgroupRecords[i].data.material.basicMaterial.kd = { 1.0f, 0.0f, 0.0f };
                hitgroupRecords[i].data.material.basicMaterial.ks = { 1.0f, 1.0f, 1.0f };
                hitgroupRecords[i].data.material.basicMaterial.alpha = 30.0f;
                OPTIX_CHECK(optixSbtRecordPackHeader(m_programs[sbtIndex], &hitgroupRecords[i]));
                ++sbtIndex;
                ++i;
                hitgroupRecords[i] = hitgroupRecords[i - 1];
                OPTIX_CHECK(optixSbtRecordPackHeader(m_programs[sbtIndex], &hitgroupRecords[i]));
                ++i;
            }
            break;
        }
        default:
            break;
        }
    }

    // Now copy our host record to the device
    CUdeviceptr deviceHitGroupRecord;
    size_t      hitGroupRecordSize = sizeof(HitGroupSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceHitGroupRecord), recordsCount * hitGroupRecordSize));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(deviceHitGroupRecord),
        hitgroupRecords,
        recordsCount * hitGroupRecordSize,
        cudaMemcpyKind::cudaMemcpyHostToDevice
    ));

    // Finally we specify how many records and how they are packed in memory
    m_shaderBindingTable.hitgroupRecordBase = deviceHitGroupRecord;
    m_shaderBindingTable.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    m_shaderBindingTable.hitgroupRecordCount = static_cast<unsigned int>(recordsCount);

    delete[] hitgroupRecords;
}

void Renderer::WriteLights(Params& params)
{
    const std::vector<PointLight>& lights = m_scene->GetLights();
    const size_t& nbLights = lights.size();

    for (size_t i = 0; i < nbLights && i < Params::MAX_LIGHTS; ++i)
    {
        const glm::vec3& lightPos = lights[i].GetPosition();
        const glm::vec3& lightColor = lights[i].GetColor();
        params.lights[i].position = { lightPos.x, lightPos.y, lightPos.z };
        params.lights[i].color = { lightColor.x, lightColor.y, lightColor.z };
    }

    const glm::vec3& ambientLight = m_scene->GetAmbientLight();
    params.ambientLight = { ambientLight.r, ambientLight.g, ambientLight.b };

    params.nbLights = static_cast<int>(nbLights);
}

void Renderer::Launch()
{
    std::cout << "RayTracinGO: launching OptiX ..." << std::endl;
    // Populate the per-launch params
    Params params;
    params.image = m_outputBuffer->map();
    params.image_width = m_windowWidth;
    params.image_height = m_windowHeight;
    params.origin_x = m_windowWidth / 2;
    params.origin_y = m_windowHeight / 2;
    params.handle = m_traversableHandle;

    WriteLights(params);

    // Transfer params to the device
    CUdeviceptr d_param;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_param), sizeof(Params)));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_param),
        &params, sizeof(params),
        cudaMemcpyKind::cudaMemcpyHostToDevice
    ));

    // Launch now, passing our pipeline, lauch params and SBT
    // (for a high performance application you want to use streams and double-buffering)
    OPTIX_CHECK(optixLaunch(m_pipeline, 
        m_cudaStream,
        d_param,
        sizeof(Params),
        &m_shaderBindingTable,
        m_windowWidth,
        m_windowHeight,
        1 /*depth=*/));
    CUDA_SYNC_CHECK();

    // Rendered results are now in params.image
    m_outputBuffer->unmap();
}

void Renderer::Display(std::string outfile)
{
    sutil::ImageBuffer buffer;
    buffer.data = m_outputBuffer->getHostPointer();
    buffer.width = m_windowWidth;
    buffer.height = m_windowHeight;
    buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
    if (outfile.empty())
    {
        sutil::displayBufferWindow("RayTracinGO", buffer);
    }
    else
    {
        sutil::displayBufferFile(outfile.c_str(), buffer, false);
    }
}

void Renderer::CleanUp()
{
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
} // namespace engine
