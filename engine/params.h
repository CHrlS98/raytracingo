//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <stdint.h>
#include <device_types.h>
#include <vector_types.h>
#include <optix.h>

namespace engine
{

enum RayType
{
    RAY_TYPE_RADIANCE  = 0,
    RAY_TYPE_OCCLUSION = 1,
    RAY_TYPE_COUNT
};

struct BasicMaterial
{
    /// Couleur ambiante
    float3 ka;

    /// Couleur diffuse
    float3 kd;

    /// Couleur speculaire
    float3 ks;

    /// Coefficient de reflexion speculaire
    float alpha;
};

struct SphereData
{
    float radius;
    float3 position;
};

struct PlaneData
{
    float3 normal;
    float3 position;
};

struct BasicLight
{
    /// Position
    float3 position;

    /// Couleur de l'eclairage
    float3 color;
};

struct CameraData
{
    float3 cam_eye;
    float3 camera_u, camera_v, camera_w;
};

struct MissData
{
    float r, g, b;
};

struct HitGroupData
{
    /// Representation geometrique de l'objet
    union
    {
        PlaneData plane;
        SphereData sphere;
    } geometry;

    /// Materiel de l'objet a representer
    union
    {
        BasicMaterial basicMaterial;
    } material;
};

struct Params
{
    static const int MAX_LIGHTS = 10;

    uchar4* image;
    uint32_t image_width;
    uint32_t image_height;
    int32_t origin_x;
    int32_t origin_y;

    int nbLights;

    BasicLight lights[MAX_LIGHTS];
    float3 ambientLight;
    OptixTraversableHandle handle;
};
} // namespace engine
