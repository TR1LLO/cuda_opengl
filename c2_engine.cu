
#include "c2_engine.cuh"
#include <device_launch_parameters.h>

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>

#include <math.h>
#include <vector>
#include <fstream>
#include <iostream>
using namespace std;
using namespace float4_math;

//---------------------------------------UTILS-------------------------------------

template<class t> __device__ __inline__ void tswap(t& a, t& b)
{
    t x = a; a = b; b = x;
}

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

#define maxwarps 16
#define farr(vec) ((float*)&(vec))
#define lerp(v1, v2, a) __fmaf_rn((v2) - (v1), a, v1)

//-----------------------------float4--------------------------------
__host__ __device__ __inline__ float4   float4_math::sum(const float4 v0, const float4 v1)
{
    return { v0.x + v1.x, v0.y + v1.y, v0.z + v1.z, v0.w + v1.w };
}
__host__ __device__ __inline__ float4   float4_math::sub(const float4 v0, const float4 v1)
{
    return { v0.x - v1.x, v0.y - v1.y, v0.z - v1.z, v0.w - v1.w };
}
__host__ __device__ __inline__ float4   float4_math::mul(const float s, const float4 v)
{
    return { v.x * s, v.y * s, v.z * s, v.w };
}
__host__ __device__ __inline__ float4   float4_math::div(const float s, const float4 v)
{
    return { v.x / s, v.y / s, v.z / s, v.w };
}

__host__ __device__ __inline__ float3   float4_math::sum(const float3 v0, const float3 v1)
{
    return { v0.x + v1.x, v0.y + v1.y, v0.z + v1.z };
}
__host__ __device__ __inline__ float3   float4_math::sub(const float3 v0, const float3 v1)
{
    return { v0.x - v1.x, v0.y - v1.y, v0.z - v1.z };
}
__host__ __device__ __inline__ float3   float4_math::mul(const float s, const float3 v)
{
    return { v.x * s, v.y * s, v.z * s };
}
__host__ __device__ __inline__ float3   float4_math::div(const float s, const float3 v)
{
    return { v.x / s, v.y / s, v.z / s };
}

__host__ __device__ __inline__ float    float4_math::dot(const float4 v0, const float4 v1)
{
    return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
}
__host__ __device__ __inline__ float    float4_math::dot(const float3 v0, const float3 v1)
{
    return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
}
__host__ __device__ __inline__ float    float4_math::len(const float4 v)
{
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}
__host__ __device__ __inline__ float    float4_math::len(const float3 v)
{
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}
__host__ __device__ __inline__ float4   float4_math::nor(const float4 v)
{
    return div(len(v), v);
}
__host__ __device__ __inline__ float3   float4_math::nor(const float3 v)
{
    return div(len(v), v);
}

__host__ __device__ __inline__ float4   float4_math::wconv(const float4 v)
{
    float s = fabsf(v.w);
    return { v.x / s, v.y / s, v.z / s, v.w / s };
}
__host__ __device__ __inline__ float4   float4_math::refl(const float4 n, const float4 v)
{
    return sub(v, mul(2 * dot(v, n), n));
}

__host__ __device__ float4  float4_math::mul(const float4 v, const mat4 m)
{
    float4 r;
    r.x = v.x * m.v[0x0] + v.y * m.v[0x4] + v.z * m.v[0x8] + v.w * m.v[0xc];
    r.y = v.x * m.v[0x1] + v.y * m.v[0x5] + v.z * m.v[0x9] + v.w * m.v[0xd];
    r.z = v.x * m.v[0x2] + v.y * m.v[0x6] + v.z * m.v[0xa] + v.w * m.v[0xe];
    r.w = v.x * m.v[0x3] + v.y * m.v[0x7] + v.z * m.v[0xb] + v.w * m.v[0xf];
    return r;
}
__host__ __device__ float3  float4_math::mul(const float3 v, const mat4 m)
{
    float3 r;
    r.x = v.x * m.v[0x0] + v.y * m.v[0x4] + v.z * m.v[0x8];
    r.y = v.x * m.v[0x1] + v.y * m.v[0x5] + v.z * m.v[0x9];
    r.z = v.x * m.v[0x2] + v.y * m.v[0x6] + v.z * m.v[0xa];
    return r;
}
__host__ __device__ mat4    float4_math::mul(const mat4 m1, const mat4 m2)
{
    mat4 res;
    res.v[0x0] = m1.v[0x0] * m2.v[0x0] + m1.v[0x1] * m2.v[0x4] + m1.v[0x2] * m2.v[0x8] + m1.v[0x3] * m2.v[0xc];
    res.v[0x1] = m1.v[0x0] * m2.v[0x1] + m1.v[0x1] * m2.v[0x5] + m1.v[0x2] * m2.v[0x9] + m1.v[0x3] * m2.v[0xd];
    res.v[0x2] = m1.v[0x0] * m2.v[0x2] + m1.v[0x1] * m2.v[0x6] + m1.v[0x2] * m2.v[0xa] + m1.v[0x3] * m2.v[0xe];
    res.v[0x3] = m1.v[0x0] * m2.v[0x3] + m1.v[0x1] * m2.v[0x7] + m1.v[0x2] * m2.v[0xb] + m1.v[0x3] * m2.v[0xf];

    res.v[0x4] = m1.v[0x4] * m2.v[0x0] + m1.v[0x5] * m2.v[0x4] + m1.v[0x6] * m2.v[0x8] + m1.v[0x7] * m2.v[0xc];
    res.v[0x5] = m1.v[0x4] * m2.v[0x1] + m1.v[0x5] * m2.v[0x5] + m1.v[0x6] * m2.v[0x9] + m1.v[0x7] * m2.v[0xd];
    res.v[0x6] = m1.v[0x4] * m2.v[0x2] + m1.v[0x5] * m2.v[0x6] + m1.v[0x6] * m2.v[0xa] + m1.v[0x7] * m2.v[0xe];
    res.v[0x7] = m1.v[0x4] * m2.v[0x3] + m1.v[0x5] * m2.v[0x7] + m1.v[0x6] * m2.v[0xb] + m1.v[0x7] * m2.v[0xf];

    res.v[0x8] = m1.v[0x8] * m2.v[0x0] + m1.v[0x9] * m2.v[0x4] + m1.v[0xa] * m2.v[0x8] + m1.v[0xb] * m2.v[0xc];
    res.v[0x9] = m1.v[0x8] * m2.v[0x1] + m1.v[0x9] * m2.v[0x5] + m1.v[0xa] * m2.v[0x9] + m1.v[0xb] * m2.v[0xd];
    res.v[0xa] = m1.v[0x8] * m2.v[0x2] + m1.v[0x9] * m2.v[0x6] + m1.v[0xa] * m2.v[0xa] + m1.v[0xb] * m2.v[0xe];
    res.v[0xb] = m1.v[0x8] * m2.v[0x3] + m1.v[0x9] * m2.v[0x7] + m1.v[0xa] * m2.v[0xb] + m1.v[0xb] * m2.v[0xf];

    res.v[0xc] = m1.v[0xc] * m2.v[0x0] + m1.v[0xd] * m2.v[0x4] + m1.v[0xe] * m2.v[0x8] + m1.v[0xf] * m2.v[0xc];
    res.v[0xd] = m1.v[0xc] * m2.v[0x1] + m1.v[0xd] * m2.v[0x5] + m1.v[0xe] * m2.v[0x9] + m1.v[0xf] * m2.v[0xd];
    res.v[0xe] = m1.v[0xc] * m2.v[0x2] + m1.v[0xd] * m2.v[0x6] + m1.v[0xe] * m2.v[0xa] + m1.v[0xf] * m2.v[0xe];
    res.v[0xf] = m1.v[0xc] * m2.v[0x3] + m1.v[0xd] * m2.v[0x7] + m1.v[0xe] * m2.v[0xb] + m1.v[0xf] * m2.v[0xf];
    return res;
}

//-----------------------------mat4--------------------------------
__host__ __device__ float   float4_math::det(mat4 m, int x, int y)
{
    int x1 = 0 + (x <= 0); int y1 = (0 + (y <= 0)) * 4;
    int x2 = 1 + (x <= 1); int y2 = (1 + (y <= 1)) * 4;
    int x3 = 2 + (x <= 2); int y3 = (2 + (y <= 2)) * 4;
    return
        m.v[x1 + y1] * (m.v[x2 + y2] * m.v[x3 + y3] - m.v[x3 + y2] * m.v[x2 + y3]) -
        m.v[x2 + y1] * (m.v[x1 + y2] * m.v[x3 + y3] - m.v[x3 + y2] * m.v[x1 + y3]) +
        m.v[x3 + y1] * (m.v[x1 + y2] * m.v[x2 + y3] - m.v[x2 + y2] * m.v[x1 + y3]);
}
__host__ __device__ float   float4_math::det(mat4 m)
{
    float _8d_9c = m.v[0x8] * m.v[0xd] - m.v[0x9] * m.v[0xc];
    float _8e_ac = m.v[0x8] * m.v[0xe] - m.v[0xa] * m.v[0xc];
    float _8f_bc = m.v[0x8] * m.v[0xf] - m.v[0xb] * m.v[0xc];
    float _9e_ad = m.v[0x9] * m.v[0xe] - m.v[0xa] * m.v[0xd];
    float _9f_bd = m.v[0x9] * m.v[0xf] - m.v[0xb] * m.v[0xd];
    float _af_be = m.v[0xa] * m.v[0xf] - m.v[0xb] * m.v[0xe];
    return
        m.v[0x0] * (m.v[0x5] * _af_be - m.v[0x6] * _9f_bd + m.v[0x7] * _9e_ad) -
        m.v[0x1] * (m.v[0x4] * _af_be - m.v[0x6] * _8f_bc + m.v[0x7] * _8e_ac) +
        m.v[0x2] * (m.v[0x4] * _9f_bd - m.v[0x5] * _8f_bc + m.v[0x7] * _8d_9c) -
        m.v[0x3] * (m.v[0x4] * _9e_ad - m.v[0x5] * _8e_ac + m.v[0x6] * _8d_9c);
}
__host__ __device__ mat4    float4_math::inv(mat4 m)
{
    float n = 1 / det(m);
    return {
        +det(m, 0, 0) * n, -det(m, 0, 1) * n, +det(m, 0, 2) * n, -det(m, 0, 3) * n,
        -det(m, 1, 0) * n, +det(m, 1, 1) * n, -det(m, 1, 2) * n, +det(m, 1, 3) * n,
        +det(m, 2, 0) * n, -det(m, 2, 1) * n, +det(m, 2, 2) * n, -det(m, 2, 3) * n,
        -det(m, 3, 0) * n, +det(m, 3, 1) * n, -det(m, 3, 2) * n, +det(m, 3, 3) * n,
    };
}

__host__ __device__ mat4    float4_math::rotx(mat4 mat, float angle)
{
    float c = cosf(angle), s = sinf(angle);
    return mul(mat, {
        1,  0,  0,  0,
        0, +c, +s,  0,
        0, -s, +c,  0,
        0,  0,  0,  1
        });
}
__host__ __device__ mat4    float4_math::roty(mat4 mat, float angle)
{
    float c = cosf(angle), s = sinf(angle);
    return mul(mat, {
        +c,  0, +s,  0,
         0,  1,  0,  0,
        -s,  0, +c,  0,
         0,  0,  0,  1
        });
}
__host__ __device__ mat4    float4_math::rotz(mat4 mat, float angle)
{
    float c = cosf(angle), s = sinf(angle);
    return mul(mat, {
        +c, +s,  0,  0,
        -s, +c,  0,  0,
         0,  0,  1,  0,
         0,  0,  0,  1
        });
}
__host__ __device__ mat4    float4_math::move(mat4 mat, float x, float y, float z)
{
    return mul(mat, {
        1,  0,  0,  0,
        0,  1,  0,  0,
        0,  0,  1,  0,
        x,  y,  z,  1
        });
}
__host__ __device__ mat4    float4_math::scale(mat4 mat, float x, float y, float z)
{
    return mul(mat, {
        x,  0,  0,  0,
        0,  y,  0,  0,
        0,  0,  z,  0,
        0,  0,  0,  1
        });
}

__host__ __device__ mat4    float4_math::perspective(float a, float z, float r)
{
    float c = -tanf(a / 2);
    return {
        c * r,  0,      0,      0,
        0,      c,      0,      0,
        0,      0,     -1,      1,
        0,      0,      1,      0
    };
}

//---------------------------------------MODULE-------------------------------------
namespace graphics
{
    __constant__ ushort2 screensize;
    __device__ __inline__ short xconv(float x)
    {
        return (short)(x * screensize.x + screensize.x) / 2;
    }
    __device__ __inline__ short yconv(float y)
    {
        return (short)(y * screensize.y + screensize.y) / 2;
    }


    #define mtxlock(mtx) while (atomicCAS(mtx, 0, 1) != 0)
    #define mtxunlock(mtx) atomicExch(mtx, 0)
    __constant__ uchar4* display;
    __constant__ float* zbuffer;
    __constant__ int* mtxbuffer;
    __device__ uchar4 cconv(const float4 c)
    {
        float v = max(max(0.0f, c.x), max(c.y, c.z));
        v = (v <= 1) ? 0 : (1 - expf(1 - v));

        uchar4 r; int t = c.w * 0xff; r.w = max(0, min(t, 0xff));
        t = (int)((v + c.x) * 0xff); r.x = max(0, min(t, 0xff));
        t = (int)((v + c.y) * 0xff); r.y = max(0, min(t, 0xff));
        t = (int)((v + c.z) * 0xff); r.z = max(0, min(t, 0xff));
        return r;
    }
    
    
    __global__ void clear_display(float4 col)
    {
        display[threadIdx.x + blockIdx.x * blockDim.x] = cconv(col);
    }
    __global__ void clear_zbuffer(float zmax)
    {
        zbuffer[threadIdx.x + blockIdx.x * blockDim.x] = zmax;
    }

    __device__ uchar4 mix(uchar4 c1, uchar4 c2, float a1, float a2)
    {
        return { (uchar)(c1.x * (1 - a2) + c2.x * a2), (uchar)(c1.y * (1 - a2) + c2.y * a2), (uchar)(c1.z * (1 - a2) + c2.z * a2), (uchar)max(0, min(1, a1 * (1 - a2) + a2) * 0xff) };
    }
    __device__ bool axyz_test(const short2 px, const float4 ver)
    {
        if (px.x < 0 || px.x >= screensize.x) return false;
        if (px.y < 0 || px.y >= screensize.y) return false;
        uint i = px.x + screensize.x * px.y;
        return atomicMin((int*)&zbuffer[i], *(int*)&ver.z) > *(int*)&ver.z;
    }
    __device__ void setpixel(const short2 px, const float4 ver, const float4 col)
    {
        if (px.x < 0 || px.x >= screensize.x) return;
        if (px.y < 0 || px.y >= screensize.y) return;
        uint i = px.x + screensize.x * px.y;
        uchar4 c = cconv(col);

        bool free = false;
        do {
            if (free = atomicCAS(mtxbuffer + i, 0, 1) == 1)
                continue;
            
            atomicMin((int*)&zbuffer[i], *(int*)&ver.z);
            uchar4& p = display[i];
            p = (zbuffer[i] >= ver.z) ? mix(p, c, (float)p.w / 0xff, col.w) : mix(c, p, col.w, (float)p.w / 0xff);
            mtxbuffer[i] = 0;
        } while (free);
    }


    __device__  void drawline(float4 v1, float4 v2, float4 c)
    {
        //------clamp--z------------
        if (v1.z > v2.z) tswap(v1, v2);
        if (v1.z < -1)
        {
            if (v2.z < -1) return; float a = (-1 - v1.z) / (v2.z - v1.z);
            v1.x += a * (v2.x - v1.x); v1.y += a * (v2.y - v1.y); v1.z += a * (v2.z - v1.z);
        }
        if (v2.z > +1)
        {
            if (v1.z > +1) return; float a = (v2.z - 1) / (v2.z - v1.z);
            v2.x -= a * (v2.x - v1.x); v2.y -= a * (v2.y - v1.y); v2.z -= a * (v2.z - v1.z);
        }
        //------clamp--y------------
        if (v1.y > v2.y) tswap(v1, v2);
        if (v1.y < -1)
        {
            if (v2.y < -1) return; float a = (-1 - v1.y) / (v2.y - v1.y);
            v1.x += a * (v2.x - v1.x); v1.y += a * (v2.y - v1.y); v1.z += a * (v2.z - v1.z);
        }
        if (v2.y > +1)
        {
            if (v1.y > +1) return; float a = (v2.y - 1) / (v2.y - v1.y);
            v2.x -= a * (v2.x - v1.x); v2.y -= a * (v2.y - v1.y); v2.z -= a * (v2.z - v1.z);
        }
        //------clamp--x------------
        if (v1.x > v2.x) tswap(v1, v2);
        if (v1.x < -1)
        {
            if (v2.x < -1) return; float a = (-1 - v1.x) / (v2.x - v1.x);
            v1.x += a * (v2.x - v1.x); v1.y += a * (v2.y - v1.y); v1.z += a * (v2.z - v1.z);
        }
        if (v2.x > +1)
        {
            if (v1.x > +1) return; float a = (v2.x - 1) / (v2.x - v1.x);
            v2.x -= a * (v2.x - v1.x); v2.y -= a * (v2.y - v1.y); v2.z -= a * (v2.z - v1.z);
        }


        int n = max(fabsf(xconv(v1.x) - xconv(v2.x)), fabsf(yconv(v1.y) - yconv(v2.y)));
        v2.x = (v2.x - v1.x) / n;
        v2.y = (v2.y - v1.y) / n;
        v2.z = (v2.z - v1.z) / n;
        while (n--)
        {
            setpixel({ xconv(v1.x), yconv(v1.y) }, v1, c);
            v1 = { v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, 0 };
        }
    }


    __global__ void transformer(float4* ver, mat4 m)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        ver[idx] = mul(ver[idx], m);
    }
    __global__ void transformer(float3* nor, mat4 m)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        nor[idx] = mul(nor[idx], m);
    }


    texture<float4, 2> tex0, tex1;
    texture<float4, 2> tex2, tex3;
    

    #define rst_cores 64
    #define rst_fragcache 64
    #define rst_cachesize 2048
    __constant__ int fsize;
    __constant__ int mdata_size; __constant__ void* mdata; __constant__ MShader mshader;
    __constant__ int bdata_size; __constant__ void* bdata; __constant__ BShader bshader;
    __constant__ int vdata_size; __constant__ void* vdata; __constant__ VShader vshader;
    __constant__ int fdata_size; __constant__ void* fdata; __constant__ FShader fshader;

    #define asf4(fragm)             ((float4*)&(fragm))
    #define fraglerp(f, f0, f1, k)  for(int i=0; i<fsize/sizeof(float); i++) farr(f)[i] = lerp(farr(f0)[i], farr(f1)[i], k);
    #define fragswap(f0, f1)        for(int i=0; i<fsize/sizeof(float); i++) tswap(farr(f0)[i], farr(f1)[i]);
    __global__ void vtxkernel(void* data, char* v4)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        extern __shared__ char v43[]; v4 += blockIdx.x * blockDim.x * 4 * fsize;
        int i0 = (0 + threadIdx.x * 4) * fsize;
        int i1 = (1 + threadIdx.x * 4) * fsize;
        int i2 = (2 + threadIdx.x * 4) * fsize;
        int i3 = (3 + threadIdx.x * 4) * fsize;
        mshader(i * 3 + 0, v4 + i0, mdata);
        mshader(i * 3 + 1, v4 + i1, mdata);
        mshader(i * 3 + 2, v4 + i2, mdata);

        bshader(v4 + i0, bdata);
        bshader(v4 + i1, bdata);
        bshader(v4 + i2, bdata);
        
        vshader(v4 + i0, vdata);
        vshader(v4 + i1, vdata);
        vshader(v4 + i2, vdata);

        if (asf4(v4[i1])->y > asf4(v4[i0])->y) fragswap(v4[i1], v4[i0]);
        if (asf4(v4[i2])->y > asf4(v4[i0])->y) fragswap(v4[i2], v4[i0]);
        if (asf4(v4[i1])->y > asf4(v4[i2])->y) fragswap(v4[i1], v4[i2]);

        float k = (asf4(v4[i2])->y - asf4(v4[i1])->y) / (asf4(v4[i0])->y - asf4(v4[i1])->y);
        fraglerp(v4[i3], v4[i1], v4[i0], k);
        if (asf4(v4[i3])->x < asf4(v4[i2])->x)
            fragswap(v4[i3], v4[i2]);

        __syncthreads();
        int* dst = (int*)((char*)data + blockIdx.x * blockDim.x * 4 * fsize);
        for (int i = threadIdx.x; i < blockDim.x * fsize; i += blockDim.x)
            dst[i] = ((int*)&v4[0])[i];
    }
    __global__ void rasterizer2(void* data)
    {
        __shared__ char v4[rst_fragcache * 4];
        int* src = (int*)((char*)data + blockIdx.x * 4 * fsize);
        for (int i = threadIdx.x; i < fsize; i += blockDim.x)
            ((int*)&v4)[i] = src[i];
        __syncthreads();

        //--------------------------------------------------------------
        char _v0[rst_fragcache];
        char _v1[rst_fragcache];
        char _v[rst_fragcache];
        short2 px0, px1;
        float t0, t1;
        float a0, a1;
        float b0, b1;
        //-------------------loop--1--------------------
        t0 = asf4(v4[2 * fsize])->y; if (t0 >= +1) goto _loop2;
        t1 = asf4(v4[0 * fsize])->y; if (t1 <= -1) goto _loop2;
        px0.y = yconv(max(-1, t0)); a0 = (max(-1, t0) - t0) / (t1 - t0);
        px1.y = yconv(min(+1, t1)); a1 = (min(+1, t1) - t0) / (t1 - t0);
        a1 = (a1 - a0) / (px1.y - px0.y);
        while (px0.y <= px1.y)
        {
            fraglerp(_v0, v4[2 * fsize], v4[0 * fsize], a0);
            fraglerp(_v1, v4[3 * fsize], v4[0 * fsize], a0);


            t0 = asf4(_v0)->x; if (t0 >= +1) goto _loop1_inc;
            t1 = asf4(_v1)->x; if (t1 <= -1) goto _loop1_inc;
            px0.x = xconv(max(-1, t0)); b0 = (max(-1, t0) - t0) / (t1 - t0);
            px1.x = xconv(min(+1, t1)); b1 = (min(+1, t1) - t0) / (t1 - t0);


            b1 = (b1 - b0) / (px1.x - px0.x);
            b0 += b1 * threadIdx.x;
            px0.x += threadIdx.x;
            b1 *= blockDim.x;
            while (px0.x <= px1.x)
            {
                fraglerp(_v, _v0, _v1, b0);
                fshader(px0, &_v, fdata);
                px0.x += blockDim.x;
                b0 += b1;
            }
        _loop1_inc:
            __syncthreads();
            px0.y++;
            a0 += a1;
        }

    _loop2:
        __syncthreads();
        //-------------------loop--2--------------------
        t0 = asf4(v4[2 * fsize])->y; if (t0 <= -1) return;
        t1 = asf4(v4[1 * fsize])->y; if (t1 >= +1) return;
        px0.y = yconv(min(+1, t0)); a0 = (min(+1, t0) - t0) / (t1 - t0);
        px1.y = yconv(max(-1, t1)); a1 = (max(-1, t1) - t0) / (t1 - t0);
        a1 = (a1 - a0) / (px0.y - px1.y);
        while (px0.y > px1.y)
        {
            fraglerp(_v0, v4[2 * fsize], v4[1 * fsize], a0);
            fraglerp(_v1, v4[3 * fsize], v4[1 * fsize], a0);


            t0 = asf4(_v0)->x; if (t0 >= +1) goto _loop2_inc;
            t1 = asf4(_v1)->x; if (t1 <= -1) goto _loop2_inc;
            px0.x = xconv(max(-1, t0)); b0 = (max(-1, t0) - t0) / (t1 - t0);
            px1.x = xconv(min(+1, t1)); b1 = (min(+1, t1) - t0) / (t1 - t0);


            b1 = (b1 - b0) / (px1.x - px0.x);
            b0 += b1 * threadIdx.x;
            px0.x += threadIdx.x;
            b1 *= blockDim.x;
            while (px0.x <= px1.x)
            {
                fraglerp(_v, _v0, _v1, b0);
                fshader(px0, &_v, fdata);
                px0.x += blockDim.x;
                b0 += b1;
            }
        _loop2_inc:
            __syncthreads();
            px0.y--;
            a0 += a1;
        }
    }
    void rasterize(int fn, int fsize)
    {
        int bsize = 64;
        int blocks = (fn + bsize - 1) / bsize;
        char* v4;
        cudaMalloc(&v4, bsize * 4 * fsize * blocks);
        void* data;
        cudaMalloc(&data, fn * fsize * 4);
        vtxkernel<<<blocks, bsize>>>(data, v4);

        cudaDeviceSynchronize();
        cudaFree(v4);

        rasterizer2<<<fn, rst_cores>>>(data);
        cudaDeviceSynchronize();
        cudaFree(data);
    }


    __global__ void rasterizer(void)
    {
        __shared__ char vtx[rst_fragcache * 4];
        if (threadIdx.x < 3)
        {
            mshader(blockIdx.x * 3 + threadIdx.x, &vtx[threadIdx.x * fsize], mdata);
            bshader(&vtx[threadIdx.x * fsize], bdata);
            vshader(&vtx[threadIdx.x * fsize], vdata);
        }
        __syncthreads();

        if (threadIdx.x < 1)
        {
            if (asf4(vtx[1 * fsize])->y > asf4(vtx[0 * fsize])->y) fragswap(vtx[1 * fsize], vtx[0 * fsize]);
            if (asf4(vtx[2 * fsize])->y > asf4(vtx[0 * fsize])->y) fragswap(vtx[2 * fsize], vtx[0 * fsize]);
            if (asf4(vtx[1 * fsize])->y > asf4(vtx[2 * fsize])->y) fragswap(vtx[1 * fsize], vtx[2 * fsize]);
        }
        __syncthreads();
        //-----------------------------------
        float t0, t1;
        t0 = (asf4(vtx[2 * fsize])->y - asf4(vtx[1 * fsize])->y) / (asf4(vtx[0 * fsize])->y - asf4(vtx[1 * fsize])->y);
        if (threadIdx.x < 1)
        {
            fraglerp(vtx[3 * fsize], vtx[1 * fsize], vtx[0 * fsize], t0);
            if (asf4(vtx[3 * fsize])->x < asf4(vtx[2 * fsize])->x)
                fragswap(vtx[3 * fsize], vtx[2 * fsize]);
        }
        __syncthreads();

        
        //--------------------------------------------------------------
        __shared__ char _v0[rst_fragcache];
        __shared__ char _v1[rst_fragcache];
        char _v[rst_fragcache];
        short2 px0, px1;
        float a0, a1;
        float b0, b1;

        __shared__ char cache[rst_cachesize];
        for (int i = threadIdx.x; i < fdata_size / sizeof(int); i += blockDim.x)
            ((int*)&cache[0])[i] = ((int*)fdata)[i];
        __syncthreads();
        //-------------------loop--1--------------------
        t0 = asf4(vtx[2 * fsize])->y; if (t0 >= +1) goto _loop2;
        t1 = asf4(vtx[0 * fsize])->y; if (t1 <= -1) goto _loop2;
        px0.y = yconv(max(-1, t0)); a0 = (max(-1, t0) - t0) / (t1 - t0);
        px1.y = yconv(min(+1, t1)); a1 = (min(+1, t1) - t0) / (t1 - t0);
        a1 = (a1 - a0) / (px1.y - px0.y);
        while (px0.y <= px1.y)
        {
            for (int i = threadIdx.x; i < fsize / sizeof(float); i += blockDim.x)
            {
                t1 = farr(vtx[0 * fsize])[i];
                t0 = farr(vtx[2 * fsize])[i];
                farr(_v0)[i] = lerp(t0, t1, a0);
                t0 = farr(vtx[3 * fsize])[i];
                farr(_v1)[i] = lerp(t0, t1, a0);
            }
            __syncthreads();

            t0 = asf4(_v0)->x; if (t0 >= +1) goto _loop1_inc;
            t1 = asf4(_v1)->x; if (t1 <= -1) goto _loop1_inc;
            px0.x = xconv(max(-1, t0)); b0 = (max(-1, t0) - t0) / (t1 - t0);
            px1.x = xconv(min(+1, t1)); b1 = (min(+1, t1) - t0) / (t1 - t0);


            b1 = (b1 - b0) / (px1.x - px0.x);
            b0 += b1 * threadIdx.x;
            px0.x += threadIdx.x;
            b1 *= blockDim.x;
            while (px0.x <= px1.x)
            {
                fraglerp(_v, _v0, _v1, b0);
                fshader(px0, _v, cache);
                px0.x += blockDim.x;
                b0 += b1;
            }
        _loop1_inc:
            __syncthreads();
            px0.y++;
            a0 += a1;
        }

    _loop2:
        __syncthreads();
        //-------------------loop--2--------------------
        t0 = asf4(vtx[2 * fsize])->y; if (t0 <= -1) return;
        t1 = asf4(vtx[1 * fsize])->y; if (t1 >= +1) return;
        px0.y = yconv(min(+1, t0)); a0 = (min(+1, t0) - t0) / (t1 - t0);
        px1.y = yconv(max(-1, t1)); a1 = (max(-1, t1) - t0) / (t1 - t0);
        a1 = (a1 - a0) / (px0.y - px1.y);
        px0.y--; a0 += a1;
        while (px0.y >= px1.y)
        {
            for (int i = threadIdx.x; i < fsize / sizeof(float); i += blockDim.x)
            {
                t1 = farr(vtx[1 * fsize])[i];
                t0 = farr(vtx[2 * fsize])[i];
                farr(_v0)[i] = lerp(t0, t1, a0);
                t0 = farr(vtx[3 * fsize])[i];
                farr(_v1)[i] = lerp(t0, t1, a0);
            }
            __syncthreads();

            t0 = asf4(_v0)->x; if (t0 >= +1) goto _loop2_inc;
            t1 = asf4(_v1)->x; if (t1 <= -1) goto _loop2_inc;
            px0.x = xconv(max(-1, t0)); b0 = (max(-1, t0) - t0) / (t1 - t0);
            px1.x = xconv(min(+1, t1)); b1 = (min(+1, t1) - t0) / (t1 - t0);


            b1 = (b1 - b0) / (px1.x - px0.x);
            b0 += b1 * threadIdx.x;
            px0.x += threadIdx.x;
            b1 *= blockDim.x;
            while (px0.x <= px1.x)
            {
                fraglerp(_v, _v0, _v1, b0);
                fshader(px0, _v, cache);
                px0.x += blockDim.x;
                b0 += b1;
            }
        _loop2_inc:
            __syncthreads();
            px0.y--;
            a0 += a1;
        }
    }

    __global__ void rasterizer2(void)
    {
        const short tidx = threadIdx.x % 32, tcount = 32;
        const short widx = threadIdx.x / 32, wcount = blockDim.x / 32;
        
        //-----------------------------------
        __shared__ char vfrag[4][rst_fragcache];
        if (threadIdx.x < 3)
        {
            mshader(blockIdx.x * 3 + threadIdx.x, &vfrag[threadIdx.x][0], mdata);
            bshader(                              &vfrag[threadIdx.x][0], bdata);
            vshader(                              &vfrag[threadIdx.x][0], vdata);
        }
        __syncthreads();
        
        if (threadIdx.x < 1)
        {
            if (asf4(vfrag[1][0])->y > asf4(vfrag[0][0])->y) fragswap(vfrag[1][0], vfrag[0][0]);
            if (asf4(vfrag[2][0])->y > asf4(vfrag[0][0])->y) fragswap(vfrag[2][0], vfrag[0][0]);
            if (asf4(vfrag[1][0])->y > asf4(vfrag[2][0])->y) fragswap(vfrag[1][0], vfrag[2][0]);
        }
        __syncthreads();
        //-----------------------------------
        float k = (asf4(vfrag[2][0])->y - asf4(vfrag[1][0])->y) / (asf4(vfrag[0][0])->y - asf4(vfrag[1][0])->y);
        if (threadIdx.x < 1)
        {
            fraglerp(vfrag[3][0], vfrag[1][0], vfrag[0][0], k);
            if (asf4(vfrag[3][0])->x < asf4(vfrag[2][0])->x)
                fragswap(vfrag[3][0], vfrag[2][0]);
        }
        __syncthreads();
        
        //--------------------------------------------------------------
        __shared__ char _v0[rst_fragcache];
        __shared__ char _v1[rst_fragcache];
        char _v[rst_fragcache];
        short2 px0, px1;
        float t0, t1;
        float a0, a1;
        float b0, b1;

        __shared__ char cache[rst_cachesize];
        for (int i = threadIdx.x; i < fsize; i += blockDim.x * sizeof(int))
            ((int*)&cache[0])[i] = ((int*)fdata)[i];
        __syncthreads();
        //-------------------loop--1--------------------
        t0 = asf4(vfrag[2][0])->y; if (t0 >= +1) goto _loop2;
        t1 = asf4(vfrag[0][0])->y; if (t1 <= -1) goto _loop2;
        px0.y = yconv(max(-1, t0)); a0 = (max(-1, t0) - t0) / (t1 - t0);
        px1.y = yconv(min(+1, t1)); a1 = (min(+1, t1) - t0) / (t1 - t0);


        a1 = (a1 - a0) / (px1.y - px0.y);
        a0 += a1 * widx;
        px0.y += widx;
        a1 *= wcount;
        while (px0.y <= px1.y)
        {
            fraglerp(_v0, vfrag[2][0], vfrag[0][0], a0);
            fraglerp(_v1, vfrag[3][0], vfrag[0][0], a0);


            t0 = asf4(_v0)->x; if (t0 >= +1) goto _loop1_inc;
            t1 = asf4(_v1)->x; if (t1 <= -1) goto _loop1_inc;
            px0.x = xconv(max(-1, t0)); b0 = (max(-1, t0) - t0) / (t1 - t0);
            px1.x = xconv(min(+1, t1)); b1 = (min(+1, t1) - t0) / (t1 - t0);


            b1 = (b1 - b0) / (px1.x - px0.x);
            b0 += b1 * tidx;
            px0.x += tidx;
            b1 *= tcount;
            while (px0.x <= px1.x)
            {
                fraglerp(_v, _v0, _v1, b0);
                fshader(px0, &_v, cache);
                px0.x += tcount;
                b0 += b1;
            }
        _loop1_inc:
            __syncthreads();
            px0.y += wcount;
            a0 += a1;
        }
        
    _loop2:
        __syncthreads();
        //-------------------loop--2--------------------
        t0 = asf4(vfrag[2][0])->y; if (t0 <= -1) return;
        t1 = asf4(vfrag[1][0])->y; if (t1 >= +1) return;
        px0.y = yconv(min(+1, t0)); a0 = (min(+1, t0) - t0) / (t1 - t0);
        px1.y = yconv(max(-1, t1)); a1 = (max(-1, t1) - t0) / (t1 - t0);


        a1 = (a1 - a0) / (px0.y - px1.y);
        a0 += a1 * widx;
        px0.y -= widx;
        a1 *= wcount;
        while (px0.y > px1.y)
        {
            fraglerp(_v0, vfrag[2][0], vfrag[1][0], a0);
            fraglerp(_v1, vfrag[3][0], vfrag[1][0], a0);


            t0 = asf4(_v0)->x; if (t0 >= +1) goto _loop2_inc;
            t1 = asf4(_v1)->x; if (t1 <= -1) goto _loop2_inc;
            px0.x = xconv(max(-1, t0)); b0 = (max(-1, t0) - t0) / (t1 - t0);
            px1.x = xconv(min(+1, t1)); b1 = (min(+1, t1) - t0) / (t1 - t0);


            b1 = (b1 - b0) / (px1.x - px0.x);
            b0 += b1 * tidx;
            px0.x += tidx;
            b1 *= tcount;
            while (px0.x <= px1.x)
            {
                fraglerp(_v, _v0, _v1, b0);
                fshader(px0, &_v, cache);
                px0.x += tcount;
                b0 += b1;
            }
        _loop2_inc:
            __syncthreads();
            px0.y -= wcount;
            a0 += a1;
        }
    }
};
using namespace graphics;


static ushort2 framesize;
GLuint vbo; struct cudaGraphicsResource* cuda_vbo_resource;
void create_vbo(int w, int h)
{
    framesize = { (ushort)w, (ushort)h };
    cudaMemcpyToSymbol(screensize, &framesize, sizeof(ushort2));
    size_t size = w * h * sizeof(uchar4);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, vbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, size, NULL, GL_DYNAMIC_DRAW);
    cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard);


    void* devptr;
    cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer(&devptr, &size, cuda_vbo_resource);
    cudaMemcpyToSymbol(display, &devptr, sizeof(uchar4*));

    cudaMalloc(&devptr, w * h * sizeof(float)); cudaMemcpyToSymbol(zbuffer, &devptr, sizeof(float*));
    cudaMalloc(&devptr, w * h * sizeof(int)); cudaMemcpyToSymbol(mtxbuffer, &devptr, sizeof(int*));
    cudaMemset(devptr, 0, w * h * sizeof(int));
}
void delete_vbo(void)
{
    cudaGraphicsUnregisterResource(cuda_vbo_resource);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, vbo);
    glDeleteBuffers(1, &vbo); vbo = 0;

    void* devptr;
    cudaMemcpyFromSymbol(&devptr, &zbuffer, sizeof(float*)); cudaFree(devptr);
    cudaMemcpyFromSymbol(&devptr, &mtxbuffer, sizeof(char*)); cudaFree(devptr);
}
void map_buffer(void)
{
    void* devptr; size_t size;
    cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer(&devptr, &size, cuda_vbo_resource);
    cudaMemcpyToSymbol(display, &devptr, sizeof(uchar4*));
}
void unmap_buffer(void)
{
    cudaDeviceSynchronize();
    cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
}


mat4 mat_camera;
mat4 mat_global;
static ushort blocksize = 128;
void resize(int w, int h)
{
    delete_vbo();
    create_vbo(w, h);
}
void clear(void)
{
    int blocks = (framesize.x * framesize.y + blocksize - 1) / blocksize;
    clear_display<<<blocks, blocksize>>>(float4{ 0, 0, 0, 1 });
    clear_zbuffer<<<blocks, blocksize>>>(0);
}


//---------------------------------------STRUCTURES-------------------------------------

//-------------------Bitmap--fp32-------------------------------
void Bitmap_fp32::init(int w, int h)
{
    this->w = w;
    this->h = h;
    data = new float4[w * h];
}
void Bitmap_fp32::release(void)
{
    w = h = 0;
    data = 0;
}
bool Bitmap_fp32::load_bmp(void)
{
    ifstream file(src, ios_base::binary);
    if (!file.is_open())
        return false;

    struct bmptags
    {
        WORD    bfType;
        DWORD   bfSize;
        WORD    bfReserved1;
        WORD    bfReserved2;
        DWORD   bfOffBits;
    } _bmptags;
    struct bmpinfo
    {
        DWORD  biSize;
        LONG   biWidth;
        LONG   biHeight;
        WORD   biPlanes;
        WORD   biBitCount;
        DWORD  biCompression;
        DWORD  biSizeImage;
        LONG   biXPelsPerMeter;
        LONG   biYPelsPerMeter;
        DWORD  biClrUsed;
        DWORD  biClrImportant;
    } _bmpinfo;

    file.seekg(0);
    file.read((char*)&_bmptags.bfType, sizeof(bmptags::bfType));
    file.read((char*)&_bmptags.bfSize, sizeof(bmptags::bfSize));
    file.read((char*)&_bmptags.bfReserved1, sizeof(bmptags::bfReserved1));
    file.read((char*)&_bmptags.bfReserved2, sizeof(bmptags::bfReserved2));
    file.read((char*)&_bmptags.bfOffBits, sizeof(bmptags::bfOffBits));

    file.read((char*)&_bmpinfo.biSize, sizeof(bmpinfo::biSize));
    file.read((char*)&_bmpinfo.biWidth, sizeof(bmpinfo::biWidth));
    file.read((char*)&_bmpinfo.biHeight, sizeof(bmpinfo::biHeight));
    file.read((char*)&_bmpinfo.biPlanes, sizeof(bmpinfo::biPlanes));
    file.read((char*)&_bmpinfo.biBitCount, sizeof(bmpinfo::biBitCount));
    file.read((char*)&_bmpinfo.biCompression, sizeof(bmpinfo::biCompression));
    file.read((char*)&_bmpinfo.biSizeImage, sizeof(bmpinfo::biSizeImage));
    file.read((char*)&_bmpinfo.biXPelsPerMeter, sizeof(bmpinfo::biXPelsPerMeter));
    file.read((char*)&_bmpinfo.biYPelsPerMeter, sizeof(bmpinfo::biYPelsPerMeter));
    file.read((char*)&_bmpinfo.biClrUsed, sizeof(bmpinfo::biClrUsed));
    file.read((char*)&_bmpinfo.biClrImportant, sizeof(bmpinfo::biClrImportant));

    init(_bmpinfo.biWidth, _bmpinfo.biHeight);
    file.seekg(_bmptags.bfOffBits, ios_base::beg);

    unsigned char* t = new unsigned char[w * 3];
    for (int y = 0; y < h; y++)
    {
        file.read((char*)t, w * 3);
        for (int x = 0; x < w; x++)
        {
            float4& px = data[y * w + x];
            px.x = (float)t[x * 3 + 2] / 0xff;
            px.y = (float)t[x * 3 + 1] / 0xff;
            px.z = (float)t[x * 3 + 0] / 0xff;
            px.w = 1;
        }
    }
    delete[] t;
    file.close();
    return true;
}
void Bitmap_fp32::alpha_cut(float4 col, float lim)
{
    for (int i = 0; i < w * h; i++)
    {
        float d = dot(col, data[i]) / 3;
        if (d < lim) continue;
        data[i].w = (1 - d) / (1 - lim);
    }
}

Bitmap_fp32::Bitmap_fp32(std::string src)
{
    this->src = src;
}
void Bitmap_fp32::gpu_init(void)
{
    texdesc = cudaCreateChannelDesc<float4>();
    cudaMallocArray(&texarr, &texdesc, w, h);
    cudaMemcpy2DToArray(texarr, 0, 0, data, w * sizeof(float4), w * sizeof(float4), h, cudaMemcpyHostToDevice);
}
void Bitmap_fp32::gpu_free(void)
{
    cudaFreeArray(texarr); texarr = NULL;
}

//-------------------Submesh---------------------------------
__device__ void mshader_0(int i, void* frag, void* data)
{
    FragData* fd = (FragData*)frag;
    Submesh::MultimeshData* mmd = &((Submesh*)data)->data;
    fd->v = mmd->ver[mmd->ver_idx[i]];
    fd->n = mmd->nor[mmd->nor_idx[i]];
    fd->t = mmd->txy[mmd->txy_idx[i]];
}
__device__ MShader mshader_0_ptr = mshader_0;


bool Submesh::init(void)
{
    cudaMemcpyFromSymbol(&dev_mshader, mshader_0_ptr, sizeof(mshader));
    cudaMalloc(&dev_data, dev_dsize = sizeof(*this));
    return true;
}
void Submesh::release(void)
{
    size = 0;
    dev_dsize = 0; cudaFree(dev_data); dev_data = NULL;
}
void Submesh::bind(void)
{
    cudaMemcpy(dev_data, this, dev_dsize, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(mdata_size, &dev_dsize, sizeof(dev_dsize));
    cudaMemcpyToSymbol(mshader, &dev_mshader, sizeof(mshader));
    cudaMemcpyToSymbol(mdata, &dev_data, sizeof(dev_data));

    int size = sizeof(FragData);
    cudaMemcpyToSymbol(fsize, &size, sizeof(size));
}
void Submesh::unbind(void)
{
}
Mesh* Submesh::clone(void)
{
    Submesh* r = new Submesh(*this);
    r->init();
    return r;
}

//-------------------MultiSubmesh---------------------------------
MultiSubmesh::MultiSubmesh(string src)
{
    this->src = src;
}
bool MultiSubmesh::init(void)
{
    ifstream file(src);
    if (!file.is_open())
        return false;

    vector<float4> ver; vector<ushort> ver_idx;
    vector<float3> nor; vector<ushort> nor_idx;
    vector<float2> txy; vector<ushort> txy_idx;
    vector<Submesh*> sms;
    int idx = 0;

    string s;
    size_t i1, i2;
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0);
    while ((size_t)file.tellg() < size)
    {
        file >> s;
        if (strcmp(s.c_str(), "v") == 0)
        {
            float4 t = { 0, 0, 0, 1 };
            file >> t.x >> t.y >> t.z;
            ver.push_back(t);
        }
        if (strcmp(s.c_str(), "vn") == 0)
        {
            float3 t;
            file >> t.x >> t.y >> t.z;
            nor.push_back(t);
        }
        if (strcmp(s.c_str(), "vt") == 0)
        {
            float2 t;
            file >> t.x >> t.y;
            txy.push_back(t);
        }
        if (strcmp(s.c_str(), "f") == 0)
        {
            getline(file, s);
            s = s.substr(1) + " ";
            vector<ushort3> oof;
            while (s.size() > 0)
            {
                size_t k = s.find_first_of(' ');
                string str = s.substr(0, k);
                s = s.substr(k + 1);

                i1 = str.find_first_of('/');
                i2 = str.find_last_of('/');
                ushort3 t;
                t.x = atoi(str.substr(0, i1).c_str()) - 1;
                t.z = atoi(str.substr(i1 + 1, i2).c_str()) - 1;
                t.y = atoi(str.substr(i2 + 1).c_str()) - 1;
                oof.push_back(t);
            }
            if (oof.size() == 3)
            {
                ver_idx.push_back(oof[0].x); nor_idx.push_back(oof[0].y); txy_idx.push_back(oof[0].z);
                ver_idx.push_back(oof[1].x); nor_idx.push_back(oof[1].y); txy_idx.push_back(oof[1].z);
                ver_idx.push_back(oof[2].x); nor_idx.push_back(oof[2].y); txy_idx.push_back(oof[2].z);
                idx++;
            }
            if (oof.size() == 4)
            {
                ver_idx.push_back(oof[0].x); nor_idx.push_back(oof[0].y); txy_idx.push_back(oof[0].z);
                ver_idx.push_back(oof[1].x); nor_idx.push_back(oof[1].y); txy_idx.push_back(oof[1].z);
                ver_idx.push_back(oof[3].x); nor_idx.push_back(oof[3].y); txy_idx.push_back(oof[3].z);

                ver_idx.push_back(oof[1].x); nor_idx.push_back(oof[1].y); txy_idx.push_back(oof[1].z);
                ver_idx.push_back(oof[2].x); nor_idx.push_back(oof[2].y); txy_idx.push_back(oof[2].z);
                ver_idx.push_back(oof[3].x); nor_idx.push_back(oof[3].y); txy_idx.push_back(oof[3].z);
                idx += 2;
            }
            oof.clear();
            // vt 0.750000 0.250000 rd  2--1
            // vt 0.750000 0.312500 ru  | /|
            // vt 0.718750 0.312500 lu  |/ |
            // vt 0.718750 0.250000 ld  3--0
        }
        if (strcmp(s.c_str(), "usemtl") == 0)
        {
            Submesh* sm = new Submesh();
            sm->init();
            sm->offset = idx;
            if (sms.size() > 0)
                sms[sms.size() - 1]->size = idx - sms[sms.size() - 1]->offset;
            sms.push_back(sm);
        }
    }
    sms[sms.size() - 1]->size = idx - sms[sms.size() - 1]->offset;
    file.close();
    size = idx;

    cudaMalloc(&data.ver, sizeof(float4) * (data.ver_size = (ushort)ver.size())); cudaMemcpy(data.ver, ver.data(), sizeof(float4) * ver.size(), cudaMemcpyHostToDevice); ver.clear();
    cudaMalloc(&data.nor, sizeof(float3) * (data.nor_size = (ushort)nor.size())); cudaMemcpy(data.nor, nor.data(), sizeof(float3) * nor.size(), cudaMemcpyHostToDevice); nor.clear();
    cudaMalloc(&data.txy, sizeof(float2) * (data.txy_size = (ushort)txy.size())); cudaMemcpy(data.txy, txy.data(), sizeof(float2) * txy.size(), cudaMemcpyHostToDevice); txy.clear();
    cudaMalloc(&data.ver_idx, sizeof(ushort) * (data.ver_idx_size = (ushort)ver_idx.size())); cudaMemcpy(data.ver_idx, ver_idx.data(), sizeof(ushort) * ver_idx.size(), cudaMemcpyHostToDevice); ver_idx.clear();
    cudaMalloc(&data.nor_idx, sizeof(ushort) * (data.nor_idx_size = (ushort)nor_idx.size())); cudaMemcpy(data.nor_idx, nor_idx.data(), sizeof(ushort) * nor_idx.size(), cudaMemcpyHostToDevice); nor_idx.clear();
    cudaMalloc(&data.txy_idx, sizeof(ushort) * (data.txy_idx_size = (ushort)txy_idx.size())); cudaMemcpy(data.txy_idx, txy_idx.data(), sizeof(ushort) * txy_idx.size(), cudaMemcpyHostToDevice); txy_idx.clear();

    submesh = new Submesh * [count = (int)sms.size()];
    for (int i = 0; i < count; i++)
    {
        Submesh* sm = sms[i];
        sm->data = data;
        sm->data.ver_idx += sm->offset * 3;
        sm->data.nor_idx += sm->offset * 3;
        sm->data.txy_idx += sm->offset * 3;
        submesh[i] = sm;
    }
    sms.clear();

    cudaMalloc(&dev_data, dev_dsize = sizeof(*this));
    return true;
}
void MultiSubmesh::release(void)
{
    for (int i = 0; i < count; i++)
    {
        submesh[i]->release();
        delete submesh[i];
    }
    count = 0; 
    delete[] submesh; 
    submesh = 0;

    size = 0;
    cudaFree(data.ver_idx); cudaFree(data.ver);
    cudaFree(data.nor_idx); cudaFree(data.nor);
    cudaFree(data.txy_idx); cudaFree(data.txy);
    memset(&data, sizeof(data), 0);

    dev_dsize = 0; cudaFree(dev_data); dev_data = NULL;
}
void MultiSubmesh::bind(void)
{
    cudaMemcpy(dev_data, this, dev_dsize, cudaMemcpyHostToDevice);
}
void MultiSubmesh::unbind(void)
{
}
Mesh* MultiSubmesh::clone(void)
{
    MultiSubmesh* r = new MultiSubmesh(src);
    r->init();
    return r;
}

//-------------------MeshPlane---------------------------------
__device__ void mshader_1(int i, void* frag, void* data)
{
    FragData* fd = (FragData*)frag;
    MeshPlane::PlaneData* pd = &((MeshPlane*)data)->data;
    fd->v = pd->ver[i];
    fd->n = pd->nor[i];
    fd->t = pd->txy[i];
}
__device__ MShader mshader_1_ptr = mshader_1;


bool MeshPlane::init(void)
{
    size = 2;
    const int n = 6;
    float4 _ver[n] = {
        { -1, -1, 0, 1 }, { -1, +1, 0, 1 }, { +1, +1, 0, 1 }, 
        { -1, -1, 0, 1 }, { +1, -1, 0, 1 }, { +1, +1, 0, 1 }
    };
    float3 _nor[n] = {
        { 0, 0, -1 }, { 0, 0, -1 }, { 0, 0, -1 },
        { 0, 0, -1 }, { 0, 0, -1 }, { 0, 0, -1 }
    };
    float2 _txy[n] = {
        { 0, 0 }, { 0, 1 }, { 1, 1 },
        { 0, 0 }, { 1, 0 }, { 1, 1 }
    };
    cudaMalloc(&data.ver, n * sizeof(float4));
    cudaMemcpy(data.ver, _ver, n * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMalloc(&data.nor, n * sizeof(float3));
    cudaMemcpy(data.nor, _nor, n * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMalloc(&data.txy, n * sizeof(float2));
    cudaMemcpy(data.txy, _txy, n * sizeof(float2), cudaMemcpyHostToDevice);

    cudaMemcpyFromSymbol(&dev_mshader, mshader_1_ptr, sizeof(mshader));
    cudaMalloc(&dev_data, dev_dsize = sizeof(*this));
    return true;
}
void MeshPlane::release(void)
{
    size = 0;
    cudaFree(data.ver); data.ver = NULL;
    cudaFree(data.nor); data.nor = NULL;
    cudaFree(data.txy); data.txy = NULL;

    dev_dsize = 0; cudaFree(dev_data); dev_data = NULL;
}
void MeshPlane::bind(void)
{
    cudaMemcpy(dev_data, this, dev_dsize, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(mdata_size, &dev_dsize, sizeof(dev_dsize));
    cudaMemcpyToSymbol(mshader, &dev_mshader, sizeof(mshader));
    cudaMemcpyToSymbol(mdata, &dev_data, sizeof(dev_data));

    int size = sizeof(FragData);
    cudaMemcpyToSymbol(fsize, &size, sizeof(size));
}
void MeshPlane::unbind(void)
{
}
Mesh* MeshPlane::clone(void)
{
    MeshPlane* r = new MeshPlane();
    r->init();
    return r;
}

//-------------------Rigidbody---------------------------------
__device__ void bshader_0(void* frag, void* data)
{
    FragData* d = (FragData*)frag;
    Rigidbody* b = (Rigidbody*)data;
    d->v = mul(d->v, b->transform);
    d->n = mul(d->n, b->transform);
}
__device__ BShader bshader_0_ptr = bshader_0;


bool Rigidbody::init(void)
{
    cudaMemcpyFromSymbol(&dev_bshader, bshader_0_ptr, sizeof(bshader));
    cudaMalloc(&dev_data, dev_dsize = sizeof(*this));
    return true;
}
void Rigidbody::release(void)
{
    dev_dsize = 0; cudaFree(dev_data); dev_data = NULL;
}
void Rigidbody::bind(void)
{
    transform = mul(mat_local, *mat_basis);
    cudaMemcpy(dev_data, this, dev_dsize, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(bdata_size, &dev_dsize, sizeof(dev_dsize));
    cudaMemcpyToSymbol(bshader, &dev_bshader, sizeof(bshader));
    cudaMemcpyToSymbol(bdata, &dev_data, sizeof(dev_data));
}
void Rigidbody::unbind(void)
{
}
Rigidbody* Rigidbody::clone(void)
{
    Rigidbody* r = new Rigidbody(*this);
    r->init();
    return r;
}

//-------------------Material--Texture-------------------------------
__device__ void fshader_0(short2 px, void* frag, void* data)
{
    FragData* f = (FragData*)frag;
    MaterialTexture* m = (MaterialTexture*)data;
    if (!axyz_test(px, f->v))
        return;

    float4 ldir = nor(sub(f->v, wconv(m->lpos)));
    float l = max(0, -dot(*(float4*)&f->n, ldir)) * m->lpow.z;

    float4 edir = nor(refl(*(float4*)&f->n, sub(f->v, { 0, 0, 0, 1 })));
    l += powf(max(0, dot(edir, ldir)), m->lpow.w);

    l = (m->lpow.y + l) * m->lpow.x;
    float4 c = tex2D(tex0, f->t.x, f->t.y);
    
    c.x = lerp(c.x, m->col.x, m->col.w);
    c.y = lerp(c.y, m->col.y, m->col.w);
    c.z = lerp(c.z, m->col.z, m->col.w);
    setpixel(px, f->v, mul(l, c));
}
__device__ FShader fshader_0_ptr = fshader_0;


bool MaterialTexture::init(void)
{
    cudaMemcpyFromSymbol(&dev_fshader, fshader_0_ptr, sizeof(fshader));
    cudaMalloc(&dev_data, dev_dsize = sizeof(*this));
    return true;
}
void MaterialTexture::release(void)
{
    dev_dsize = 0; cudaFree(dev_data); dev_data = NULL;
}
void MaterialTexture::bind(void)
{
    lpos = *lpos_ptr;
    lpow = *lpow_ptr;
    cudaMemcpy(dev_data, this, dev_dsize, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(fdata_size, &dev_dsize, sizeof(dev_dsize));
    cudaMemcpyToSymbol(fshader, &dev_fshader, sizeof(fshader));
    cudaMemcpyToSymbol(fdata, &dev_data, sizeof(dev_data));

    tex0.normalized = true;
    tex0.filterMode = cudaFilterModeLinear;
    tex0.addressMode[0] = cudaAddressModeClamp;
    tex0.addressMode[1] = cudaAddressModeClamp;
    cudaBindTextureToArray(&tex0, bmp->texarr, &bmp->texdesc);
}
void MaterialTexture::unbind(void)
{
    cudaUnbindTexture(&tex0);
}
Material* MaterialTexture::clone(void)
{
    MaterialTexture* r = new MaterialTexture(*this);
    r->init();
    return r;
}

//-------------------Renderer-------------------------------
__device__ void vshader_0(void* frag, void* data)
{
    Renderer* r = (Renderer*)data;
    FragData* fd = (FragData*)frag;
    fd->v = wconv(mul(fd->v, r->camera));
    fd->n = mul(fd->n, r->camera);
}
__device__ VShader vshader_0_ptr = vshader_0;


bool Renderer::init(void)
{
    cudaMemcpyFromSymbol(&dev_vshader, vshader_0_ptr, sizeof(vshader));
    cudaMalloc(&dev_data, dev_dsize = sizeof(*this));
    return true;
}
void Renderer::release(void)
{
    dev_dsize = 0; cudaFree(dev_data); dev_data = NULL;
    body->release(); delete body; body = NULL;
    mesh->release(); delete mesh; mesh = NULL;
    mtl->release(); delete mtl; mtl = NULL;
}
void Renderer::bind(void)
{
    camera = mat_camera;
    cudaMemcpy(dev_data, this, dev_dsize, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(vdata_size, &dev_dsize, sizeof(dev_dsize));
    cudaMemcpyToSymbol(vshader, &dev_vshader, sizeof(vshader));
    cudaMemcpyToSymbol(vdata, &dev_data, sizeof(dev_data));
}
void Renderer::unbind(void)
{
}


void Renderer::render(void)
{
    rasterizer<<<mesh->size, rst_cores>>>();
}
Renderer* Renderer::clone(void)
{
    Renderer* rnd = new Renderer();
    rnd->init();
    rnd->mtl = mtl->clone();
    rnd->mesh = mesh->clone();
    rnd->body = body->clone();
    return rnd;
}


//-------------------Renderer--MultiSubmesh-------------------------------
bool RendererMultiSubmesh::init(void)
{
    renderer = new Renderer * [count];
    return true;
}
void RendererMultiSubmesh::release(void)
{
    for (int i = 0; i < count; i++)
    {
        renderer[i]->release();
        delete renderer[i];
    }
    count = 0; 
    delete[] renderer; renderer = NULL;
    body->release(); delete body; body = NULL;
}
void RendererMultiSubmesh::bind(void)
{
}
void RendererMultiSubmesh::unbind(void)
{
}


void RendererMultiSubmesh::render(void)
{
    for (int i = 0; i < count; i++)
    {
        Renderer*& r = renderer[i];
        r->bind(); 
        r->mtl->bind();
        r->mesh->bind();
        r->body->bind();
        r->render();
        cudaDeviceSynchronize();
        r->body->unbind();
        r->mesh->unbind();
        r->mtl->unbind();
        r->unbind();
    }
}
Renderer* RendererMultiSubmesh::clone(void)
{
    RendererMultiSubmesh* rnd = new RendererMultiSubmesh();
    rnd->body = body->clone();

    rnd->count = count;
    rnd->RendererMultiSubmesh::init();
    for (int i = 0; i < count; i++)
    {
        Renderer* r = renderer[i]->clone();
        r->body->release();
        delete r->body; r->body = rnd->body;
        rnd->renderer[i] = r;
    }
    return rnd;
}

