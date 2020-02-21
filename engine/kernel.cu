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

#include <optix.h>

#include "params.h"
#include "random.h"

#include <sutil/vec_math.h>
#include <stdio.h>

namespace engine
{
namespace device
{
extern "C" {
    __constant__ Params params;
}


static __forceinline__ __device__ void trace(
    OptixTraversableHandle handle,
    float3 ray_origin,
    float3 ray_direction,
    float tmin,
    float tmax,
    float3* prd,
    int* depth
)
{
    uint32_t p0, p1, p2, p3;
    p0 = float_as_int(prd->x);
    p1 = float_as_int(prd->y);
    p2 = float_as_int(prd->z);
    p3 = *depth;
    optixTrace(
        handle,
        ray_origin,
        ray_direction,
        tmin,
        tmax,
        0.0f,                // rayTime
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_NONE,
        RAY_TYPE_RADIANCE,   // SBT offset
        RAY_TYPE_COUNT,      // SBT stride
        RAY_TYPE_RADIANCE,   // missSBTIndex
        p0, p1, p2, p3);
    prd->x = int_as_float(p0);
    prd->y = int_as_float(p1);
    prd->z = int_as_float(p2);
    *depth = p3 + 1;
}


static __forceinline__ __device__ void setPayload(float3 p)
{
    optixSetPayload_0(float_as_int(p.x));
    optixSetPayload_1(float_as_int(p.y));
    optixSetPayload_2(float_as_int(p.z));
}

__forceinline__ __device__ uchar4 make_color(const float3&  c)
{
    return make_uchar4(
        static_cast<uint8_t>(clamp(c.x, 0.0f, 1.0f) *255.0f),
        static_cast<uint8_t>(clamp(c.y, 0.0f, 1.0f) *255.0f),
        static_cast<uint8_t>(clamp(c.z, 0.0f, 1.0f) *255.0f),
        255u
    );
}

static __device__ float3 traceOcclusion(const float3& lightPos)
{
    const float3 origin = optixGetWorldRayOrigin();
    const float3 dir = optixGetWorldRayDirection();
    const float  t = optixGetRayTmax();

    const float3 hitPoint = origin + t * dir;

    const float lightDistance = length(lightPos - hitPoint);
    const float3 directionToLight = normalize(lightPos - hitPoint);

    float3 attenuation = make_float3(1.0f);

    optixTrace(
        params.handle,
        hitPoint,
        directionToLight,
        0.01f,
        lightDistance - 0.01f,
        0.0f,
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_NONE,
        RAY_TYPE_OCCLUSION,
        RAY_TYPE_COUNT,
        RAY_TYPE_OCCLUSION,
        reinterpret_cast<uint32_t&>(attenuation.x),
        reinterpret_cast<uint32_t&>(attenuation.y),
        reinterpret_cast<uint32_t&>(attenuation.z));
    return attenuation;
}

extern "C" __global__ void __raygen__rg()
{
    // lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const float x = static_cast<float>(idx.x);
    const float y = static_cast<float>(idx.y);
    const float dimX = static_cast<float>(dim.x);
    const float dimY = static_cast<float>(dim.y);

    // Get the generic memory space pointer to the data region of the currently
    // active SBT (shader binding table) record corresponding to the current program
    const CameraData* rtData = (CameraData*)optixGetSbtDataPointer();
    const float3      U = rtData->camera_u;
    const float3      V = rtData->camera_v;
    const float3      W = rtData->camera_w;

    unsigned int seed = idx.y;
    float3 color = { 0.0f, 0.0f, 0.0f };
    for (int i = 0; i < params.samplePerPixel; i++)
    {
        const float2 d = 2.0f * make_float2((x + rnd(seed)) / dimX, (y + rnd(seed)) / dimY) - 1.0f;
        const float3 origin = rtData->cam_eye;
        const float3 direction = normalize(d.x * U + d.y * V + W);
        float3       payload_rgb = make_float3(0.5f, 0.5f, 0.5f);
        int depth = 0;
        trace(params.handle,
            origin,
            direction,
            0.00f,  // tmin
            1e16f,  // tmax
            &payload_rgb,
            &depth);

        color += payload_rgb;
    }
    params.image[idx.y * params.image_width + idx.x] = make_color(color / static_cast<float>(params.samplePerPixel));
}


extern "C" __global__ void __miss__ms()
{
    MissData* rt_data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    setPayload(make_float3(rt_data->r, rt_data->g, rt_data->b));
}

/// Teste si un rayon intersecte avec la sphere
extern "C" __global__ void __intersection__sphere()
{
    const HitGroupData* hg_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    const SphereData& sphere = hg_data->geometry.sphere;
    const float3 o = optixGetWorldRayOrigin(); // optixGetObjectRayOrigin() peu etre moins couteux ?
    const float3 dir = optixGetWorldRayDirection();

    const float3 center = sphere.position;
    const float  radius = sphere.radius;
    const float3 l = normalize(dir);

    // -b +/- sqrt(b^2 -c)
    const float b = dot(l, (o - center));
    const float c = dot(o - center, o - center) - radius * radius;
    const float discr = b * b - c;
    if (discr > 0.0f)
    {
        const float sdiscr = sqrtf(discr);
        const float t = (-b - sdiscr); // car sdiscr toujours positif

        const float3 normale = normalize(o + t * l - center);

        unsigned int p0, p1, p2;
        p0 = float_as_int(normale.x);
        p1 = float_as_int(normale.y);
        p2 = float_as_int(normale.z);

        optixReportIntersection(
            t,      // t hit
            0,          // user hit kind
            p0, p1, p2
        );
    }
}

/// Teste si le rayon intersecte un plan
extern "C" __global__ void __intersection__plane()
{
    const HitGroupData* hg_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    const PlaneData& plane = hg_data->geometry.plane;
    const float3 o = optixGetWorldRayOrigin();
    const float3 dir = optixGetWorldRayDirection();
    const float3 l = normalize(dir);

    const float3 position = plane.position;
    const float3 n = normalize(plane.normal);

    float divisor = dot(l, n);
    if (divisor != 0.0f)
    {
        const float t = dot(position - o, n) / divisor;
        unsigned int p0, p1, p2;
        p0 = float_as_int(n.x);
        p1 = float_as_int(n.y);
        p2 = float_as_int(n.z);

        if (t > 0.0f)
        {
            optixReportIntersection(
                t,      // t hit
                0,          // user hit kind
                p0, p1, p2
            );
        }
    }
}


extern "C" __global__ void __closesthit__ch()
{
    const float3 normale =
        make_float3(
            int_as_float(optixGetAttribute_0()),
            int_as_float(optixGetAttribute_1()),
            int_as_float(optixGetAttribute_2())
        );
    const float3 N = normalize(normale);

    const HitGroupData* hgData = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    const BasicMaterial& material = hgData->material.basicMaterial;

    const float t = optixGetRayTmax();
    const float3 origin = optixGetWorldRayOrigin();
    const float3 direction = optixGetWorldRayDirection();
    const float3 x = origin + t * normalize(direction);
    const float3 V = normalize(origin - x);

    const float3 lumiereAmbiante = params.ambientLight;

    const float& alpha = material.alpha;
    const float3& couleurAmbiante = material.ka;
    const float3& couleurDiffuse = material.kd;
    const float3& couleurSpeculaire = material.ks;
    const float3& couleurReflexion = material.kr;

    const float3 omega = -normalize(direction);

    float3 color = {0.0f, 0.0f, 0.0f};

    // vecteur reflechi
    const float3 r = -omega + 2 * dot(N, omega) * N;
    float3 prd = make_float3(
        int_as_float(optixGetPayload_0()),
        int_as_float(optixGetPayload_1()),
        int_as_float(optixGetPayload_2())
    );
    int depth = optixGetPayload_3() + 1;

    if (depth < params.maxTraceDepth)
    {
        trace(params.handle, x, r, 0.01f, 1e6f, &prd, &depth);
        color += prd * couleurReflexion;
    }

    const int& nbLights = params.nbLights;
    for (int i = 0; i < nbLights; ++i)
    {
        const float3 lightPos = params.lights[i].position;
        const float3 attenuation = traceOcclusion(lightPos); 
        const float3 lightColor = params.lights[i].color * attenuation;

        // Modele d'illumination de Blinn
        const float3 Lm = normalize(lightPos - x);
        const float3 H = normalize(Lm + V);

        const float3 compAmbiante = lumiereAmbiante * couleurAmbiante;
        const float3 compDiffuse = max(dot(Lm, N), 0.0f) * lightColor * couleurDiffuse;
        const float3 compSpeculaire = max(powf(dot(N, H), alpha), 0.0f) * lightColor * couleurSpeculaire;

        color += compAmbiante + compDiffuse + compSpeculaire;
    }

    setPayload(color);
}

extern "C" __global__ void __closesthit__reflection()
{
    const float3 normale =
        make_float3(
            int_as_float(optixGetAttribute_0()),
            int_as_float(optixGetAttribute_1()),
            int_as_float(optixGetAttribute_2())
        );
    const float3 N = normalize(normale);

    const HitGroupData* hgData = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    const BasicMaterial& material = hgData->material.basicMaterial;

    const float t = optixGetRayTmax();
    const float3 origin = optixGetWorldRayOrigin();
    const float3 direction = optixGetWorldRayDirection();
    const float3 x = origin + t * normalize(direction);
    const float3 V = normalize(origin - x);

    const float3 lumiereAmbiante = params.ambientLight;

    const float& alpha = material.alpha;
    const float3& couleurAmbiante = material.ka;
    const float3& couleurDiffuse = material.kd;
    const float3& couleurSpeculaire = material.ks;

    float3 color = { 0.0f, 0.0f, 0.0f };

    const int& nbLights = params.nbLights;
    for (int i = 0; i < nbLights; ++i)
    {
        const float3 lightPos = params.lights[i].position;
        const float3 attenuation = traceOcclusion(lightPos);
        const float3 lightColor = params.lights[i].color * attenuation;

        // Modele d'illumination de Blinn
        const float3 Lm = normalize(lightPos - x);
        const float3 H = normalize(Lm + V);

        const float3 compAmbiante = lumiereAmbiante * couleurAmbiante;
        const float3 compDiffuse = max(dot(Lm, N), 0.0f) * lightColor * couleurDiffuse;
        const float3 compSpeculaire = max(powf(dot(N, H), alpha), 0.0f) * lightColor * couleurSpeculaire;

        color += compAmbiante + compDiffuse + compSpeculaire;
    }

    setPayload(color);
}

extern "C" __global__ void __closesthit__full_occlusion()
{
    // materiel 100% opaque
    setPayload({ 0.0f, 0.0f, 0.0f });
}
} // namespace host
} // namespace engine
