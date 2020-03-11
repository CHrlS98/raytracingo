#pragma once

#include <PointLight.h>
#include <params.h>

#include <sutil/CUDAOutputBuffer.h>
#include <optix.h>

#include <vector>
#include <memory>

#include <scene.h>

namespace engine
{
namespace host
{
enum LogCallbackLevel
{
    Disable = 0,    // Setting the callback level will disable all messages.The callback function will not be called in this case
    Fatal = 1,      // A non-recoverable error. The context and/or OptiX itself might no longer be in a usable state
    Error = 2,      // A recoverable error, e.g., when passing invalid call parameters
    Warning = 3,    // Hints that OptiX might not behave exactly as requested by the user or may perform slower than expected
    Print = 4       // Status or progress messages
};

class Renderer
{
public:
    Renderer(std::shared_ptr<Scene> scene);
    ~Renderer();

    void Display();

private:
    OptixDeviceContext m_optixContext;
    OptixModule m_module;
    OptixModuleCompileOptions m_moduleCompileOptions;

    OptixPipeline m_pipeline;
    OptixPipelineCompileOptions m_pipelineCompileOptions;
    OptixPipelineLinkOptions m_pipelineLinkOptions;

    CUcontext m_cudaContext;
    CUstream  m_cudaStream;
    cudaDeviceProp m_cudaDeviceProperties;

    std::vector<OptixProgramGroup> m_programs;
    OptixShaderBindingTable m_shaderBindingTable;

    std::shared_ptr<Scene> m_scene;

    CUdeviceptr m_deviceGasOutputBuffer;
    OptixTraversableHandle m_traversableHandle;

    void Initialize();

    void InitOptix();
    void CreateContext();
    void CreateModule();

    void CreateRayGen();
    void CreateMiss();
    void CreateShapes();

    OptixProgramGroup CreateRayGenPrograms() const;
    OptixProgramGroup CreateMissPrograms();
    OptixProgramGroup CreateHitGroupProgram(const std::shared_ptr<IShape> shape, device::RayType type);

    void BuildRayGenRecords(CameraSbtRecord* records, const size_t& recordsCount);
    void BuildMissRecords(MissSbtRecord* records, const size_t& recordsCount);
    void BuildHitGroupRecords(HitGroupSbtRecord* records, const size_t& recordsCount);

    void BuildAccelerationStructure(OptixAabb* aabb, uint32_t* aabbInputFlags, uint32_t* sbtIndex, const size_t& nbObjects);

    void WriteLights(device::Params& params);
    void CreatePipeline();

    void LaunchFrame(sutil::CUDAOutputBuffer<uchar4>& outputBuffer, device::Params& params, device::Params* d_params);

    void CleanUp();
};
} // namespace host
} // namespace engine