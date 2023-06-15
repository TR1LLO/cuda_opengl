#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

//-----------------------------------UTILS----------------------------------
typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned short ushort;


typedef void (*MShader)(int,    void*, void*);
typedef void (*BShader)(        void*, void*);
typedef void (*VShader)(        void*, void*);
typedef void (*FShader)(short2, void*, void*);


#define identity {  1, 0, 0, 0,     0, 1, 0, 0,     0, 0, 1, 0,     0, 0, 0, 1 }
struct mat4 { float v[16] = identity; };

namespace float4_math
{
    //--------------------------float4----------------------
    __host__ __device__ __inline__ float4 sum(const float4 v0, const float4 v1);
    __host__ __device__ __inline__ float4 sub(const float4 v0, const float4 v1);
    __host__ __device__ __inline__ float4 mul(const float s, const float4 v);
    __host__ __device__ __inline__ float4 div(const float s, const float4 v);

    __host__ __device__ __inline__ float3 sum(const float3 v0, const float3 v1);
    __host__ __device__ __inline__ float3 sub(const float3 v0, const float3 v1);
    __host__ __device__ __inline__ float3 mul(const float s, const float3 v);
    __host__ __device__ __inline__ float3 div(const float s, const float3 v);

    __host__ __device__ __inline__ float dot(const float4 v0, const float4 v1);
    __host__ __device__ __inline__ float dot(const float3 v0, const float3 v1);
    __host__ __device__ __inline__ float len(const float4 v);
    __host__ __device__ __inline__ float len(const float3 v);
    __host__ __device__ __inline__ float4 nor(const float4 v);
    __host__ __device__ __inline__ float3 nor(const float3 v);

    __host__ __device__ __inline__ float4 wconv(const float4 v);
    __host__ __device__ __inline__ float4 refl(const float4 n, const float4 v);


    //--------------------------mat4----------------------
    __host__ __device__ float4 mul(const float4 v, const mat4 m);
    __host__ __device__ float3 mul(const float3 v, const mat4 m);
    __host__ __device__ mat4 mul(const mat4 m1, const mat4 m2);

    __host__ __device__ float det(mat4 m, int x, int y);
    __host__ __device__ float det(mat4 m);
    __host__ __device__ mat4 inv(mat4 m);
    
    __host__ __device__ mat4 rotx(mat4 mat, float angle);
    __host__ __device__ mat4 roty(mat4 mat, float angle);
    __host__ __device__ mat4 rotz(mat4 mat, float angle);
    __host__ __device__ mat4 move(mat4 mat, float x, float y, float z);
    __host__ __device__ mat4 scale(mat4 mat, float x, float y, float z);

    __host__ __device__ mat4 perspective(float angle, float zmin, float ratio);
}

//-----------------------------------MODULE----------------------------------
extern ushort2 framesize;
void create_vbo(int w, int h);
void delete_vbo(void);

void map_buffer(void);
void unmap_buffer(void);

extern ushort blocksize;
void resize(int w, int h);
void clear(void);

//-----------------------------------STRUCTURES----------------------------------
struct FragData
{
    float4 v;
    float3 n;
    float2 t;
};
struct Bitmap_fp32
{
    int w, h;
    float4* data;
    void init(int w, int h);
    void release(void);

    std::string src;
    Bitmap_fp32(std::string src);
    bool load_bmp(void);
    void alpha_cut(float4 col, float lim);

    cudaArray* texarr;
    cudaChannelFormatDesc texdesc;
    void gpu_init(void);
    void gpu_free(void);
};

//--------------------------MESHES------------------------------
struct Mesh
{
    int size;
    mat4 transform;

    int dev_dsize;
    void* dev_data;
    MShader dev_mshader;

    virtual bool init(void) abstract;
    virtual void release(void) abstract;
    virtual void bind(void) abstract;
    virtual void unbind(void) abstract;
    virtual Mesh* clone(void) abstract;
};
struct Submesh : Mesh
{
    struct MultimeshData
    {
        ushort ver_size; float4* ver; uint ver_idx_size; ushort* ver_idx;
        ushort nor_size; float3* nor; uint nor_idx_size; ushort* nor_idx;
        ushort txy_size; float2* txy; uint txy_idx_size; ushort* txy_idx;
    } data;
    int offset;

    virtual bool init(void);
    virtual void release(void);
    virtual void bind(void);
    virtual void unbind(void);
    Mesh* clone(void);
};
struct MultiSubmesh : Submesh
{
    int count;
    Submesh** submesh;

    std::string src;
    MultiSubmesh(std::string src);

    bool init(void);
    void release(void);
    void bind(void);
    void unbind(void);
    Mesh* clone(void);
};

struct MeshPlane : Mesh
{
    struct PlaneData
    {
        float4* ver;
        float3* nor;
        float2* txy;
    } data;

    bool init(void);
    void release(void);
    void bind(void);
    void unbind(void);
    Mesh* clone(void);
};


//--------------------------RIGIDBODIES------------------------------
extern mat4 mat_global;
struct Rigidbody
{
    mat4 mat_local;
    mat4* mat_basis = &mat_global;
    mat4 transform;

    mat4 mat_delta;

    int dev_dsize;
    void* dev_data;
    BShader dev_bshader;

    virtual bool init(void);
    virtual void release(void);
    virtual void bind(void);
    virtual void unbind(void);
    virtual Rigidbody* clone(void);
};


//--------------------------MATERIALS------------------------------
struct Material
{
    int dev_dsize;
    void* dev_data;
    FShader dev_fshader;

    virtual bool init(void) abstract;
    virtual void release(void) abstract;
    virtual void bind(void) abstract;
    virtual void unbind(void) abstract;
    virtual Material* clone(void) abstract;
};
struct MaterialTexture : Material
{
    float4 col;
    float4 lpos;
    float4 lpow;
    float4* lpos_ptr;
    float4* lpow_ptr;

    Bitmap_fp32* bmp;
    bool init(void);
    void release(void);
    void bind(void);
    void unbind(void);
    Material* clone(void);
};


//--------------------------RENDERERS------------------------------
extern mat4 mat_camera;
struct Renderer
{
    Mesh* mesh;
    Material* mtl;
    Rigidbody* body;
    mat4 camera;
 
    int dev_dsize;
    void* dev_data;
    VShader dev_vshader;

    virtual bool init(void);
    virtual void release(void);
    virtual void bind(void);
    virtual void unbind(void);

    virtual void render(void);
    virtual Renderer* clone(void);
};
struct RendererMultiSubmesh : Renderer
{
    int count;
    Renderer** renderer;

    bool init(void);
    void release(void);
    void bind(void);
    void unbind(void);

    void render(void);
    Renderer* clone(void);
};
