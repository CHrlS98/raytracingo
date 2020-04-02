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
    RayType rayType,
    float tmin,
    float tmax,
    float3* prd,
    int* depth,
    unsigned int seed
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
        rayType,   // SBT offset
        RAY_TYPE_COUNT,      // SBT stride
        rayType,   // missSBTIndex
        p0, p1, p2, p3, seed);
    prd->x = int_as_float(p0);
    prd->y = int_as_float(p1);
    prd->z = int_as_float(p2);
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

    float3 color = { 0.0f, 0.0f, 0.0f };
    const uint32_t sqrtSamplePerPixel = params.sqrtSamplePerPixel;
    const uint32_t image_index = params.image_width*idx.y + idx.x;
    unsigned int seed = tea<16>(image_index, params.frameCount);
    const float ratioNewImage = 1.0f / static_cast<float>(params.frameCount + 1);
    
    for (unsigned int i = 0; i < sqrtSamplePerPixel; ++i)
    {
        for (unsigned int j = 0; j < sqrtSamplePerPixel; ++j)
        {
            const float offsetIncrement = 1.0f / static_cast<float>(sqrtSamplePerPixel);
            const float fi = static_cast<float>(i);
            const float fj = static_cast<float>(j);

            const float2 d = 2.0f * make_float2(
                (x + (fi + rnd(seed))* offsetIncrement) / dimX, 
                (y + (fj + rnd(seed)) * offsetIncrement) / dimY
            ) - 1.0f;

            const float3 origin = rtData->cam_eye;
            const float3 direction = normalize(d.x * U + d.y * V + W);
            float3       payload_rgb = make_float3(0.5f, 0.5f, 0.5f);
            int depth = 0;
            trace(params.handle,
                origin,
                direction,
                RAY_TYPE_RADIANCE,
                0.00f,  // tmin
                1e16f,  // tmax
                &payload_rgb,
                &depth,
                seed);
            color += payload_rgb;
        }
    }

    float3 currentColor = color / static_cast<float>(sqrtSamplePerPixel * sqrtSamplePerPixel);
    int pxlIndex = idx.y * params.image_width + idx.x;

    if (params.frameCount > 0)
    {
        float3 previousColor = make_float3(params.accum_buffer[pxlIndex]);
        currentColor = lerp(previousColor, currentColor, ratioNewImage);
    }
    params.accum_buffer[pxlIndex] = make_float4(currentColor, 1.0f);
    params.image[pxlIndex] = make_color(currentColor);
}


extern "C" __global__ void __miss__ms()
{
    MissData* rt_data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    setPayload(make_float3(rt_data->r, rt_data->g, rt_data->b));
}

static __forceinline__ __device__ float3 GetRayOnHemisphere(const float3& normal, unsigned int& seed)
{
    float3 ray;
    do
    {
        // Cree un rayon sur l'hemisphere centree sur l'axe des y
        ray = make_float3(
            rnd(seed) * 2.0f - 1.0f,
            rnd(seed),
            rnd(seed) * 2.0f - 1.0f
        );

    // Pour une distribution uniforme sur la surface de l'hemisphere,
    // on ignore les rayons qui sont a l'exterieur de son rayon et les
    // rayons nuls
    } while (length(ray) > 1.0f || length(ray) < 0.01f);
    ray = normalize(ray);

    // On cree un repere autour de la normale
    const float3 Yaxis = normalize(normal);
    const float3 Xaxis = normalize(make_float3(Yaxis.y - Yaxis.z, -Yaxis.x, Yaxis.x)); // x est orthogonal a y
    const float3 Zaxis = cross(Xaxis, Yaxis); // z est orthogonal a x et y et est normalise

    // On reexprime le rayon emis dans ce nouveau repere
    ray = ray.x * Xaxis + ray.y * Yaxis + ray.z * Zaxis;
    return ray;
}

static __forceinline__ __device__ void TransformRay(const sutil::Matrix4x4& invModelMatrix, float3& origin, float3& direction)
{
    const float4 homogenousOrigin = make_float4(origin, 1.0f);
    const float4 homogenousDir = make_float4(direction, 0.0f);

    const float4 invTransformedDir = invModelMatrix * homogenousDir;
    const float4 invTransformedOrigin = invModelMatrix * homogenousOrigin;

    direction = make_float3(invTransformedDir.x, invTransformedDir.y, invTransformedDir.z);
    origin = make_float3(invTransformedOrigin.x, invTransformedOrigin.y, invTransformedOrigin.z);
}

static __forceinline__ __device__ void TransformNormal(const sutil::Matrix4x4& invModelMatrix, float3& normal)
{
    const float4 homogenousN = make_float4(normal, 0.0f);
    normal = make_float3(invModelMatrix.transpose() * homogenousN);
}

static __forceinline__ __device__ bool equalFloat(const float a, const float b, const float tolerance)
{
    return a > b - tolerance && a < b + tolerance;
}

static __forceinline__ __device__ bool GetTMinCylinder(const float3& origin, const float3& direction, const float& t0, const float& t1, float& out_t)
{
    float t = 1e16f;
    const float t_epsilon = 0.001f;
    bool valid = false;
    const float halfHeight = GENERIC_CYLINDER_HEIGHT / 2.0f;
    if (t0 > t_epsilon)
    {
        // t0 est devant la camera et est la premiere intersection avec le rayon
        const float3 p = origin + t0 * direction;
        if (p.y > -halfHeight && p.y < halfHeight)
        {
            // t0 est une intersection valide
            t = t0;
            valid = true;
        }
    }
    if (t1 > t_epsilon && t1 < t)
    {
        const float3 p = origin + t1 * direction;
        if (p.y > -halfHeight && p.y < halfHeight)
        {
            // t0 est une intersection valide
            t = t1;
            valid = true;
        }
    }
    out_t = t;
    return valid;
}

/// Teste si un rayon intersecte avec la sphere
extern "C" __global__ void __intersection__sphere()
{
    const HitGroupData* hg_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    const sutil::Matrix4x4& modelMatrix = hg_data->modelMatrix;
    const sutil::Matrix4x4 inverseMM = modelMatrix.inverse();

    float3 origin = optixGetWorldRayOrigin();
    float3 direction = optixGetWorldRayDirection();
    TransformRay(inverseMM, origin, direction);

    // -b +/- sqrt(b^2 -c)
    const float a = dot(direction, direction);
    const float b = 2.0f * dot(direction, origin);
    const float c = dot(origin, origin) - GENERIC_SPHERE_RADIUS;
    const float discr = b * b - 4.0f * a * c;
    if (discr > 0.0f)
    {
        const float sdiscr = sqrtf(discr);
        const float t = (-b - sdiscr)/(2.0f * a);

        float3 n = normalize(origin + t * direction);
        TransformNormal(inverseMM, n);

        const float t_epsilon = 0.0001f;
        if (t > t_epsilon)
        {
            unsigned int nx, ny, nz;
            nx = float_as_int(n.x);
            ny = float_as_int(n.y);
            nz = float_as_int(n.z);
            optixReportIntersection(
                t,      // t hit
                0,          // user hit kind
                nx, ny, nz
            );
        }
    }
}

/// Teste si un rayon intersecte un cylindre
extern "C" __global__ void __intersection__cylinder()
{
    static const float discrEpsilon = 0.001f;
    const HitGroupData* hg_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    const sutil::Matrix4x4& modelMatrix = hg_data->modelMatrix;
    const sutil::Matrix4x4& inverseMM = modelMatrix.inverse();

    float3 o = optixGetWorldRayOrigin();
    float3 dir = optixGetWorldRayDirection();
    TransformRay(inverseMM, o, dir);
    
    const float a = dir.x * dir.x + dir.z * dir.z;
    const float b = 2.0f * (o.x * dir.x + o.z * dir.z);
    const float c = o.x * o.x + o.z * o.z - GENERIC_CYLINDER_RADIUS;
    const float discr = b * b - 4.0f * a * c;
    if (discr > discrEpsilon)
    {
        const float sdiscr = sqrt(discr);
        const float t0 = (-b + sdiscr) / (2.0f * a);
        const float t1 = (-b - sdiscr) / (2.0f * a);
        
        float t;
        if (GetTMinCylinder(o, dir, t0, t1, t))
        {
            const float3 p = o + t * dir;
            float3 n = make_float3(p.x, 0.0f, p.z);

            TransformNormal(inverseMM, n);

            unsigned int nx, ny, nz;
            nx = float_as_int(n.x);
            ny = float_as_int(n.y);
            nz = float_as_int(n.z);

            optixReportIntersection(
                t,      // t hit
                0,      // user hit kind
                nx, ny, nz
            );
        }
    }
}

/// Teste si un rayon intersecte un disque
extern "C" __global__ void __intersection__disk()
{
    static const float diskEpsilon = 0.0001f;
    const HitGroupData* hg_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    const sutil::Matrix4x4& modelMatrix = hg_data->modelMatrix;
    const sutil::Matrix4x4& inverseMM = modelMatrix.inverse();

    float3 o = optixGetWorldRayOrigin();
    float3 dir = optixGetWorldRayDirection();
    TransformRay(inverseMM, o, dir);

    float3 n = make_float3(0.0f, 1.0f, 0.0f);
    const float divisor = dot(dir, n);
    if (!equalFloat(divisor, 0.0f, 0.01f))
    {
        const float t = dot(-o, n) / divisor;
        if (t > diskEpsilon)
        {
            const float3 p = o + t * dir;
            if (dot(p, p) < GENERIC_DISK_RADIUS)
            {
                TransformNormal(inverseMM, n);
                unsigned int nx, ny, nz;
                nx = float_as_int(n.x);
                ny = float_as_int(n.y);
                nz = float_as_int(n.z);

                optixReportIntersection(
                    t,      // t hit
                    0,      // user hit kind
                    nx, ny, nz
                );
            }
        }
    }
}

/// Teste si un rayon intersecte un rectangle
extern "C" __global__ void __intersection__rectangle()
{
    static const float rectangleEpsilon = 0.0001f;

    const HitGroupData* hg_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    const sutil::Matrix4x4& modelMatrix = hg_data->modelMatrix;
    const sutil::Matrix4x4& inverseMM = modelMatrix.inverse();

    float3 o = optixGetWorldRayOrigin();
    float3 dir = optixGetWorldRayDirection();
    TransformRay(inverseMM, o, dir);

    const float& width = GENERIC_RECTANGLE_WIDTH;
    const float3 p0 = make_float3(-width/2.0f, 0.0f, width/2.0f);
    const float3 a = make_float3(width, 0.0f, 0.0f);
    const float3 b = make_float3(0.0f, 0.0f, -width);
    float3 n = make_float3(0.0f, 1.0f, 0.0f);

    float divisor = dot(dir, n);
    if (divisor != 0.0f)
    {
        const float t = dot(p0 - o, n) / divisor;
        if (t > rectangleEpsilon)
        {
            const float3 p = o + t * dir;

            if (0.0f < dot(p - p0, a) && dot(p - p0, a) < width && 0.0f < dot(p - p0, b) && dot(p - p0, b) < width)
            {
                TransformNormal(inverseMM, n);
                unsigned int nx, ny, nz;
                nx = float_as_int(n.x);
                ny = float_as_int(n.y);
                nz = float_as_int(n.z);
            
                optixReportIntersection(
                    t,      // t hit
                    0,      // user hit kind
                    nx, ny, nz
                );
            }
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
    float3 N = normalize(normale);

    unsigned int seed = optixGetPayload_4();

    const HitGroupData* hgData = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    const BasicMaterial& material = hgData->material.basicMaterial;

    const float t = optixGetRayTmax();
    const float rayEpsilon = 1e-6f * max(t*t, 1.0f);
    const float3 origin = optixGetWorldRayOrigin();
    const float3 direction = optixGetWorldRayDirection();
    const float3 x = origin + t * normalize(direction);
    const float3 V = normalize(origin - x);

    const float3& couleurDiffuse = material.kd;

    const float3 omega = -normalize(direction);

    float3 color = {0.0f, 0.0f, 0.0f};

    // On flip la normale si elle n'est pas du cote de l'observateur
    if (dot(N, V) < 0.0f)
    {
        N *= -1.0f;
    }

    // vecteur reflechi
    const float3 Rr = -omega + 2 * dot(N, omega) * N;
    const float3 Ra = GetRayOnHemisphere(N, seed);
    const float3 r = lerp(Rr, Ra, material.roughness);

    float3 prd = { 0.0f, 0.0f, 0.0f };
    int depth = optixGetPayload_3();
    const float reflectionContribution = 0.5f;

    if (depth < params.maxTraceDepth)
    {
        ++depth;
        trace(params.handle, x, r, RAY_TYPE_RADIANCE, rayEpsilon, 1e6f, &prd, &depth, seed);
        color += prd * reflectionContribution;
    }

    // Loop qui traite les lumieres de surface
    const int& nbSurfaceLights = params.nbSurfaceLights;
    for (int l = 0; l < nbSurfaceLights; ++l)
    {
        const float3 lightNormal = params.surfaceLights[l].normal;
        const float3 v1 = params.surfaceLights[l].v1;
        const float3 v2 = params.surfaceLights[l].v2;
        const float3 corner = params.surfaceLights[l].corner;

        const float3 samplingPos = corner + rnd(seed) * v1 + rnd(seed) * v2;
        const float3 Lm = normalize(samplingPos - x);
        const float lightDistance = length(samplingPos - x);
        const float3 H = normalize(Lm + V);

        float3 attenuation = { 1.0f, 1.0f, 1.0f };
        trace(params.handle, x, Lm, RAY_TYPE_OCCLUSION, rayEpsilon, lightDistance - rayEpsilon, &attenuation, &depth, seed);
        if (attenuation.x < 0.001f && attenuation.y < 0.001f && attenuation.z < 0.001f)
        {
            continue;
        }
        const float3 lightColor = params.surfaceLights[l].color * attenuation;
        const float cosTheta = dot(N, Lm);

        const float falloff = 1.0f / (1.0f + params.surfaceLights[l].falloff * lightDistance);
        const float3 compDiffuse = dot(N, V) < 0.f ? make_float3(0.0f, 0.0f, 0.0f) : max(cosTheta, 0.0f) * lightColor * couleurDiffuse;
        color += falloff * compDiffuse;
    }

    setPayload(color);
}

extern "C" __global__ void __closesthit__light()
{
    setPayload({ 1.0f, 1.0f, 1.0f });
}

extern "C" __global__ void __closesthit__full_occlusion()
{
    // materiel 100% opaque
    setPayload({ 0.0f, 0.0f, 0.0f });
}
} // namespace host
} // namespace engine
