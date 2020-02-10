#pragma once

#include <optix.h>
#include <vector>
#include <driver_types.h>
#include <memory>

#include <sutil/CUDAOutputBuffer.h>

#include <scene.h>

#include "params.h"

namespace engine
{
enum LogCallbackLevel
{
    Disable = 0,    // Setting the callback level will disable all messages.The callback function will not be called in this case
    Fatal = 1,      // A non-recoverable error. The context and/or OptiX itself might no longer be in a usable state
    Error = 2,      // A recoverable error, e.g., when passing invalid call parameters
    Warning = 3,    // Hints that OptiX might not behave exactly as requested by the user or may perform slower than expected
    Print = 4       // Status or progress messages
};

template <typename T>
struct ShaderBindingTableRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

// These Data are defined in the params.h
typedef ShaderBindingTableRecord<CameraData> CameraSbtRecord;
typedef ShaderBindingTableRecord<MissData> MissSbtRecord;
typedef ShaderBindingTableRecord<HitGroupData> HitGroupSbtRecord;

class Renderer
{
public:
    Renderer(const Scene& scene, const int width, const int height);
    ~Renderer();

    void Launch();
    void Display(std::string outfile);

private:
    int m_windowWidth;
    int m_windowHeight;

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

    std::shared_ptr<sutil::CUDAOutputBuffer<uchar4>> m_outputBuffer;
    CUdeviceptr m_deviceGasOutputBuffer;
    OptixTraversableHandle m_traversableHandle;

    void InitOptix();
    void CreateContext();
    void CreateModule();
    void CreateRayGenerationPrograms();
    void CreateMissPrograms();
    void CreateHitGroupPrograms();

    void BuildAccelerationStructure(const Scene& scene);
    void CreatePipeline();
    void BuildShaderBindingTable(const Scene& scene);

    void BuildRayGenerationRecords(int& sbtIndex);
    void BuildMissRecords(int& sbtIndex);
    void BuildHitGroupRecords(int& sbtIndex, const Scene& scene);

    void CleanUp();
};
} // namespace engine