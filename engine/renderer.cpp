#include "renderer.h"

#include <sutil/Exception.h>
#include <sutil/Camera.h>
#include <optix_stubs.h>
#include <sutil/sutil.h>

#include <cuda_runtime.h>

#include <iostream>
#include <iomanip>

static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*callbackdata */)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
}

Renderer::Renderer(const int width, const int height)
    : m_windowWidth(width)
    , m_windowHeight(height)
    , m_rayGenerationPrograms()
    , m_missPrograms()
    , m_hitGroupPrograms()
    , m_outputBuffer(new sutil::CUDAOutputBuffer<uchar4>(sutil::CUDAOutputBufferType::CUDA_DEVICE, m_windowWidth, m_windowHeight))
{
    InitOptix();
    CreateContext();
    CreateModule();
    CreateRayGenerationPrograms();
    CreateMissPrograms();
    CreateHitGroupPrograms();

    BuildAccelerationStructure();


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
    // Initialize CUDA
    CUDA_CHECK(cudaFree(0));

    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0)
    {
        throw std::runtime_error("RayTracinGO: no CUDA capable devices found!");
    }
    std::cout << "RayTracinGO: found " << numDevices << " CUDA devices" << std::endl;

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

    m_cudaContext = 0; // zero means take the current context

    OPTIX_CHECK(optixDeviceContextCreate(m_cudaContext, 0, &m_optixContext));
    OPTIX_CHECK(optixDeviceContextSetLogCallback(m_optixContext, context_log_cb, nullptr, LogCallbackLevel::Print));
    std::cout << "RayTracinGO: successfully created OptiX context" << std::endl;
}

void Renderer::CreateModule()
{
    std::cout << "RayTracinGO: creating OptiX module ..." << std::endl;
    m_moduleCompileOptions = {};
    m_moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;     // Set to 0 for no explicit limit
    m_moduleCompileOptions.optLevel = OptixCompileOptimizationLevel::OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    m_moduleCompileOptions.debugLevel = OptixCompileDebugLevel::OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

    m_pipelineCompileOptions = {};
    m_pipelineCompileOptions.usesMotionBlur = false;
    m_pipelineCompileOptions.traversableGraphFlags = OptixTraversableGraphFlags::OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    m_pipelineCompileOptions.numPayloadValues = 3;      // How much storage, in 32b words, to make available for the payload, [0..8]
    m_pipelineCompileOptions.numAttributeValues = 3;    // How much storage, in 32b words, to make available for the attributes , [2..8]
    m_pipelineCompileOptions.exceptionFlags = OptixExceptionFlags::OPTIX_EXCEPTION_FLAG_NONE;  // TODO: should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    m_pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";                      // "optixLaunchParams" ??

    char log[2048];
    size_t sizeof_log = sizeof(log);
    const std::string ptx = sutil::getPtxString(OPTIX_SAMPLE_NAME, "kernel.cu");    // 1 PTX = 1 module ??

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
    m_rayGenerationPrograms.resize(1); // or reserve ?    
    OptixProgramGroupOptions rayGenerationOptions = {}; // No options yet
    OptixProgramGroupDesc rayGenerationDesc = {};

    rayGenerationDesc.kind = OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
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
        &m_rayGenerationPrograms[0]
    ));
}

void Renderer::CreateMissPrograms()
{
    std::cout << "RayTracinGO: creating the Miss programs ..." << std::endl;
    m_missPrograms.resize(1); // or reserve ?    
    OptixProgramGroupOptions missOptions = {}; // No options yet
    OptixProgramGroupDesc missDesc = {};

    missDesc.kind = OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_MISS;
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
        &m_missPrograms[0]
    ));
}

void Renderer::CreateHitGroupPrograms()
{
    std::cout << "RayTracinGO: creating HitGroup programs ..." << std::endl;
    m_hitGroupPrograms.resize(1); // or reserve ?    
    OptixProgramGroupOptions hitGroupOptions = {}; // No options yet
    OptixProgramGroupDesc hitGroupDesc = {};

    hitGroupDesc.kind = OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitGroupDesc.hitgroup.moduleCH = m_module;
    hitGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    hitGroupDesc.hitgroup.moduleAH = nullptr;
    hitGroupDesc.hitgroup.entryFunctionNameAH = nullptr;
    hitGroupDesc.hitgroup.moduleIS = m_module;
    hitGroupDesc.hitgroup.entryFunctionNameIS = "__intersection__is";

    char log[2048];
    size_t sizeof_log = sizeof(log);
    
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        m_optixContext,
        &hitGroupDesc,
        1,   // num program groups
        &hitGroupOptions,
        log,
        &sizeof_log,
        &m_hitGroupPrograms[0]
    ));
}

void Renderer::BuildAccelerationStructure()
{
    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OptixBuildFlags::OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelOptions.operation = OptixBuildOperation::OPTIX_BUILD_OPERATION_BUILD;

    // AABB build input
    OptixAabb   aabb = { -1.5f, -1.5f, -1.5f, 1.5f, 1.5f, 1.5f };
    CUdeviceptr deviceAaabbBuffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceAaabbBuffer), sizeof(OptixAabb)));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(deviceAaabbBuffer),
        &aabb,
        sizeof(OptixAabb),
        cudaMemcpyKind::cudaMemcpyHostToDevice
    ));

    OptixBuildInput aabbInput = {};
    aabbInput.type = OptixBuildInputType::OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    aabbInput.aabbArray.aabbBuffers = &deviceAaabbBuffer;
    aabbInput.aabbArray.numPrimitives = 1;

    uint32_t aabbInputFlags[1] = { OptixGeometryFlags::OPTIX_GEOMETRY_FLAG_NONE };
    aabbInput.aabbArray.flags = aabbInputFlags;
    aabbInput.aabbArray.numSbtRecords = 1;

    OptixAccelBufferSizes gasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(m_optixContext, &accelOptions, &aabbInput, 1, &gasBufferSizes));
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

    CUDA_CHECK(cudaFree((void*)deviceTempBufferGas));
    CUDA_CHECK(cudaFree((void*)deviceAaabbBuffer));

    size_t compactedGasSize;
    CUDA_CHECK(cudaMemcpy(&compactedGasSize, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

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
}

void Renderer::CreatePipeline()
{
    std::cout << "RayTracinGO: creating the pipeline ..." << std::endl;
    std::vector<OptixProgramGroup> programGroups;
    programGroups.reserve(m_rayGenerationPrograms.size() + m_missPrograms.size() + m_hitGroupPrograms.size());
    for (OptixProgramGroup rayGenerationProgram : m_rayGenerationPrograms)
    {
        programGroups.push_back(rayGenerationProgram);
    }
    for (OptixProgramGroup missProgram : m_missPrograms)
    {
        programGroups.push_back(missProgram);
    }
    for (OptixProgramGroup hitGroupProgram : m_hitGroupPrograms)
    {
        programGroups.push_back(hitGroupProgram);
    }

    m_pipelineLinkOptions = {};
    m_pipelineLinkOptions.maxTraceDepth = 5; // Maximum trace recursion depth. The maximum is 31
    m_pipelineLinkOptions.debugLevel = OptixCompileDebugLevel::OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    m_pipelineLinkOptions.overrideUsesMotionBlur = false;

    char log[2048];
    size_t sizeof_log = sizeof(log);

    OPTIX_CHECK_LOG(optixPipelineCreate(
        m_optixContext,
        &m_pipelineCompileOptions,
        &m_pipelineLinkOptions,
        programGroups.data(),
        (int)programGroups.size(),
        log,
        &sizeof_log,
        &m_pipeline
    ));
}

void Renderer::BuildShaderBindingTable()
{
    std::cout << "RayTracinGO: building the shader binding table ..." << std::endl;
    m_shaderBindingTable = {};
    BuildRayGenerationRecords();
    BuildMissRecords();
    BuildHitGroupRecords();
}

void Renderer::BuildRayGenerationRecords()
{
    CUdeviceptr  rayGenerationRecord;
    const size_t rayGenerationRecordSize = sizeof(RayGenerationSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&rayGenerationRecord), rayGenerationRecordSize));

    sutil::Camera camera;
    camera.ConfigureCamera(m_windowWidth, m_windowHeight);

    RayGenerationSbtRecord rayGenerationSbt;
    rayGenerationSbt.data = {};
    rayGenerationSbt.data.cam_eye = camera.eye();   // cam_eye name use in cuda file
    camera.UVWFrame(rayGenerationSbt.data.camera_u, rayGenerationSbt.data.camera_v, rayGenerationSbt.data.camera_w);

    OPTIX_CHECK(optixSbtRecordPackHeader(m_rayGenerationPrograms[0], &rayGenerationSbt));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(rayGenerationRecord),
        &rayGenerationSbt,
        rayGenerationRecordSize,
        cudaMemcpyKind::cudaMemcpyHostToDevice
    ));

    m_shaderBindingTable.raygenRecord = rayGenerationRecord;
}

void Renderer::BuildMissRecords()
{
    CUdeviceptr missRecord;
    size_t      missRecordSize = sizeof(MissSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&missRecord), missRecordSize));

    MissSbtRecord missRecordSbt;
    missRecordSbt.data = { 0.f, 0.f, 0.f };

    OPTIX_CHECK(optixSbtRecordPackHeader(m_missPrograms[0], &missRecordSbt));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(missRecord),
        &missRecordSbt,
        missRecordSize,
        cudaMemcpyKind::cudaMemcpyHostToDevice
    ));

    m_shaderBindingTable.missRecordBase = missRecord;
    m_shaderBindingTable.missRecordStrideInBytes = sizeof(MissSbtRecord);
    m_shaderBindingTable.missRecordCount = 1;
}

void Renderer::BuildHitGroupRecords()
{
    CUdeviceptr hitGroupRecord;
    size_t      hitGroupRecordSize = sizeof(HitGroupSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitGroupRecord), hitGroupRecordSize));

    HitGroupSbtRecord hitGroupRecordSbt;
    hitGroupRecordSbt.data = { 1.5f };

    OPTIX_CHECK(optixSbtRecordPackHeader(m_hitGroupPrograms[0], &hitGroupRecordSbt));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(hitGroupRecord),
        &hitGroupRecordSbt,
        hitGroupRecordSize,
        cudaMemcpyKind::cudaMemcpyHostToDevice
    ));

    m_shaderBindingTable.hitgroupRecordBase = hitGroupRecord;
    m_shaderBindingTable.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    m_shaderBindingTable.hitgroupRecordCount = 1;
}

void Renderer::Launch()
{
    CUstream stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    Params params;
    params.image = m_outputBuffer->map();
    params.image_width = m_windowWidth;
    params.image_height = m_windowHeight;
    params.origin_x = m_windowWidth / 2;
    params.origin_y = m_windowHeight / 2;
    params.handle = m_traversableHandle;

    CUdeviceptr d_param;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_param), sizeof(Params)));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_param),
        &params, sizeof(params),
        cudaMemcpyKind::cudaMemcpyHostToDevice
    ));

    // for a high performance application you
    // want to use streams and double-buffering,
    OPTIX_CHECK(optixLaunch(m_pipeline, 
        stream,
        d_param,
        sizeof(Params),
        &m_shaderBindingTable,
        m_windowWidth,
        m_windowHeight,
        1 /*depth=*/));
    CUDA_SYNC_CHECK();

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
    OPTIX_CHECK(optixProgramGroupDestroy(m_hitGroupPrograms[0]));
    OPTIX_CHECK(optixProgramGroupDestroy(m_missPrograms[0]));
    OPTIX_CHECK(optixProgramGroupDestroy(m_rayGenerationPrograms[0]));
    OPTIX_CHECK(optixModuleDestroy(m_module));
    OPTIX_CHECK(optixDeviceContextDestroy(m_optixContext));
}
