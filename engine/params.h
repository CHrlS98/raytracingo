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
#pragma once

#include <stdint.h>
#include <device_types.h>
#include <vector_types.h>
#include <optix.h>

namespace engine
{
namespace device
{
enum RayType
{
    /// Rayon pour le calcul d'illumination
    RAY_TYPE_RADIANCE  = 0,
    /// Rayon pour le calcul d'ombre
    RAY_TYPE_OCCLUSION = 1,
    /// Nombre de rayons differents
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
    /// Couleur reflexion
    float3 kr;
    /// Coefficient de reflexion speculaire
    float alpha;
};

struct SphereData
{
    /// Rayon de la sphere
    float radius;
    /// Position dans le repere monde
    float3 position;
};

struct PlaneData
{
    /// Normale du plan en coordonnees monde
    float3 normal;
    /// Position du plan dans le repere monde
    float3 position;
};

struct RectangleData
{
    float3 a;
    float3 b;
    float3 p0;
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
    /// Position de l'oeil en coordonnees monde
    float3 cam_eye;
    /// Axes de la camera dans le repere monde
    float3 camera_u, camera_v, camera_w;
};

struct MissData
{
    /// Couleur de l'arriere-plan
    float r, g, b;
};

struct HitGroupData
{
    /// Representation geometrique de l'objet
    union
    {
        PlaneData plane;
        SphereData sphere;
        RectangleData rectangle;
    } geometry;
    /// Materiel de l'objet a representer
    union
    {
        BasicMaterial basicMaterial;
    } material;
};

struct Params
{
    /// Nombre maximum de lumieres dans une scene
    static const int MAX_LIGHTS = 10;

    /// Tableau contenant l'image rendue apres une execution d'OptiX
    uchar4* image;
    /// Largeur de l'image a generer
    uint32_t image_width;
    /// hauteur de l'image a generer
    uint32_t image_height;
    /// Cette variable au carre correspond au nombre d'echantillons par pixels
    int32_t sqrtSamplePerPixel;
    /// Nombre de lumieres dans la scene
    int nbLights; 
    /// Nombre maximum de recursions pour le calcul des reflexions
    int maxTraceDepth;
    /// Tableau de lumieres
    BasicLight lights[MAX_LIGHTS];
    /// Couleur de l'eclairage ambiant
    float3 ambientLight;
    /// Handle vers la geometrie de la scene
    OptixTraversableHandle handle;
};
} // namespace device

namespace host
{
template <typename T>
struct ShaderBindingTableRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef ShaderBindingTableRecord<device::CameraData> CameraSbtRecord;
typedef ShaderBindingTableRecord<device::MissData> MissSbtRecord;
typedef ShaderBindingTableRecord<device::HitGroupData> HitGroupSbtRecord;
} // namespace host
} // namespace engine
