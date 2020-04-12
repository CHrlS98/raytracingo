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
#include <sutil/Matrix.h>
#include <optix.h>

namespace engine
{
namespace device
{
/// Rayon de la sphere generique
const __device__ float GENERIC_SPHERE_RADIUS = 1.0f;
/// Largeur du rectangle generique
const __device__ float GENERIC_RECTANGLE_WIDTH = 1.0f;
/// Rayon du cylindre generique
const __device__ float GENERIC_CYLINDER_RADIUS = 1.0f;
/// Hauteur du cylindre generique
const __device__ float GENERIC_CYLINDER_HEIGHT = 2.0f;
/// Rayon du disque generique
const __device__ float GENERIC_DISK_RADIUS = 1.0f;

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
    /// Couleur diffuse
    float3 kd;
    /// Proportion couleur reflechie
    float3 kr;
    /// Coefficient de reflexion speculaire
    float specularity;
    // Emission
    float3 Le;
};

struct SurfaceLight
{
    /// centre de la surface
    float3 corner;
    /// Vecteur v1
    float3 v1;
    /// Vecteur v2
    float3 v2;
    /// Normale de la surface
    float3 normal;
    /// Couleur de l'eclairage
    float3 color;
    /// Constante de decroissance
    float falloff;
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
    /// Matrice de transformation en coordonnees homogenes
    sutil::Matrix4x4 modelMatrix;

    /// Materiel de l'objet a representer
    BasicMaterial material;
};

struct Params
{
    /// Nombre maximum de lumieres dans une scene
    static const int MAX_LIGHTS = 10;
    /// Tableau contenant l'image rendue apres une execution d'OptiX
    uchar4* image;
    /// Buffer dans lequel on accumule l'image
    float4* accum_buffer;
    /// Largeur de l'image a generer
    uint32_t image_width;
    /// hauteur de l'image a generer
    uint32_t image_height;
    /// Cette variable au carre correspond au nombre d'echantillons par pixels
    int32_t sqrtSamplePerPixel;
    /// Nombre de lumieres de surface dans la scene
    int nbSurfaceLights;
    /// Nombre maximum de recursions pour le calcul des reflexions
    int maxTraceDepth;
    /// Tableau des lumieres de surface
    SurfaceLight surfaceLights[MAX_LIGHTS];
    /// Handle vers la geometrie de la scene
    OptixTraversableHandle handle;
    /// Nombre total de frame ecoules depuis le debut de l'affichage
    unsigned int frameCount;
    /// Vrai pour activer le pur path tracing
    bool enablePathTracing;
    /// Vrai pour utiliser un coefficient ambiant pour les zones dans l'obscurite
    bool useAmbientLight;
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
