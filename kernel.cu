#include <iostream>
#include <fstream>
#include <vector>

#include <cuda.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>


using namespace std;
typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned short ushort;
#define MAXFLOAT (std::numeric_limits<float>::max())
#define MINFLOAT (-MAXFLOAT)


//_____________________________________________________________________________
//__________________________________INTRUMENTS_________________________________
namespace models
{
    struct model
    {
        vector<float4> v;
        vector<float3> n;
        vector<float2> t;
        vector<ushort3> f;
        void release(void)
        {
            v.clear();
            n.clear();
            t.clear();
            f.clear();
        }

        void vcent(void)
        {
            int n = (int)v.size();
            float x = 0, y = 0, z = 0;
            for (int i = 0; i < n; i++)
            {
                x += v[i].x;
                y += v[i].y;
                z += v[i].z;
            }
            x /= n;
            y /= n;
            z /= n;
            for (int i = 0; i < n; i++)
            {
                v[i].x -= x;
                v[i].y -= y;
                v[i].z -= z;
            }
        }
        void vnorm(void)
        {
            float s = 0;
            for (int i = 0; i < v.size(); i++)
            {
                s = max(s, abs(v[i].x));
                s = max(s, abs(v[i].y));
                s = max(s, abs(v[i].z));
            }
            s += s == 0 ? 1 : 0;
            for (int i = 0; i < v.size(); i++)
            {
                v[i].x /= s;
                v[i].y /= s;
                v[i].z /= s;
            }
        }
        void scale(float x, float y, float z)
        {
            for (int i = 0; i < v.size(); i++)
            {
                v[i].x *= x;
                v[i].y *= y;
                v[i].z *= z;
            }
            for (int i = 0; i < n.size(); i++)
            {
                n[i].x *= x;
                n[i].y *= y;
                n[i].z *= z;
            }
        }
        

        void ntswap(void)
        {
            for (int i = 0; i < f.size(); i++)
                std::swap(f[i].y, f[i].z);
        }
        void ttran(void)
        {
            for (int i = 0; i < t.size(); i++)
                std::swap(t[i].x, t[i].y);
        }
    };

    bool loadobj(string obj, model* m)
    {
        ifstream file(obj);
        if (!file.is_open())
            return false;

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
                file >> t.x;
                file >> t.y;
                file >> t.z;
                m->v.push_back(t);
            }
            if (strcmp(s.c_str(), "vn") == 0)
            {
                float3 t;
                file >> t.x;
                file >> t.y;
                file >> t.z;
                m->n.push_back(t);
            }
            if (strcmp(s.c_str(), "vt") == 0)
            {
                float2 t;
                file >> t.x; // oof
                file >> t.y; // OOOOOOOF
                m->t.push_back(t);
            }
            if (strcmp(s.c_str(), "f") == 0)
            {
                for (int i = 0; i < 3; i++)
                {
                    file >> s;
                    i1 = s.find_first_of('/'); i2 = s.find_last_of('/');
                    ushort3 t; // IIIIIIII HAAAAAAAATEEEE WAAAAAAVEFROONT OOOOOOOOOOBJ
                    t.x = atoi(s.substr(0, i1).c_str()) - 1;
                    t.z = atoi(s.substr(i1 + 1, i2).c_str()) - 1;
                    t.y = atoi(s.substr(i2 + 1).c_str()) - 1;
                    m->f.push_back(t);
                }
            }
        }
        file.close();
        return true;
    }
    void savemodel(string path, model* m)
    {
        ofstream file(path);
        file << "v " << m->v.size() << endl;
        for (int i = 0; i < m->v.size(); i++)
            file << m->v[i].x << ' ' << m->v[i].y << ' ' << m->v[i].z << endl;
        file << endl;

        file << "n " << m->n.size() << endl;
        for (int i = 0; i < m->n.size(); i++)
            file << m->n[i].x << ' ' << m->n[i].y << ' ' << m->n[i].z << endl;
        file << endl;

        file << "t " << m->t.size() << endl;
        for (int i = 0; i < m->t.size(); i++)
            file << m->t[i].x << ' ' << m->t[i].y << endl;
        file << endl;

        file << "f " << m->f.size() << endl;
        for (int i = 0; i < m->f.size(); i++)
        {
            file << m->f[i + 0].x << ' ' << m->f[i + 0].y << ' ' << m->f[i + 0].z << '\t';
            if (i % 3 == 2) file << endl;
        }
        file.close();
    }
    bool loadmodel(string path, model* m)
    {
        ifstream file(path);
        if (!file.is_open())
            return false;
        string s;
        int n;

        file >> s >> n;
        m->v.resize(n);
        for (int i = 0; i < n; i++)
        {
            file >> m->v[i].x >> m->v[i].y >> m->v[i].z;
            m->v[i].w = 1;
        }

        file >> s >> n;
        m->n.resize(n);
        for (int i = 0; i < n; i++)
            file >> m->n[i].x >> m->n[i].y >> m->n[i].z;

        file >> s >> n;
        m->t.resize(n);
        for (int i = 0; i < n; i++)
            file >> m->t[i].x >> m->t[i].y;

        file >> s >> n; m->f.resize(n);
        for (int i = 0; i < n; i++)
            file >> m->f[i].x >> m->f[i].y >> m->f[i].z;
        file.close();
        return true;
    }
};
using namespace models;

namespace materials
{
    struct material
    {
        int w, h, s;
        float4* tex;
        void release(void)
        {
            delete[] tex; 
            tex = NULL;
            w = h = s = 0;
        }
    };


    bool loadbmp(string src, material* m)
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


        m->w = _bmpinfo.biWidth;
        m->h = _bmpinfo.biHeight;
        m->s = m->w * m->h;
        m->tex = new float4[m->s];
        file.seekg(_bmptags.bfOffBits, ios_base::beg);

        uchar* t = new uchar[m->w * 3];
        for (int y = 0; y < m->h; y++)
        {
            file.read((char*)t, m->w * 3);
            for (int x = 0; x < m->w; x++)
            {
                float4& px = m->tex[y * m->w + x];
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
    bool loadmaterial(string src, material* m)
    {
        ifstream file(src, ios::binary);
        if (!file.is_open())
            return false;

        file.read((char*)&m->w, sizeof(int));
        file.read((char*)&m->h, sizeof(int));
        m->tex = new float4[m->s = m->w * m->h];
        file.read((char*)m->tex, m->s * sizeof(float4));
        return true;
    }
    void savematerial(string src, material* m)
    {
        ofstream file(src, ios_base::binary);
        file.write((char*)&m->w, sizeof(int));
        file.write((char*)&m->h, sizeof(int));
        file.write((char*)m->tex, m->s * sizeof(float4));
        file.close();
        return;
    }
};
using namespace materials;

namespace cuda
{
    #define cores 64
    #define quad 4
    #define quadn (cores / quad)
    #define csize 0x4000
    #define cudacall(kernel, n, ...) (kernel<<<(n+cores-1)/cores, cores>>>(__VA_ARGS__))
    

    template<class t>
    __device__ __inline__ inline void swap(t& t1, t& t2)
    {
        t tmp = t1;
        t1 = t2;
        t2 = tmp;
    }

    //------------------ariphmetics-------------------

    __device__ __inline__ inline float4 sum(const float4 v1, const float4 v2)
    {
        return { v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w };
    }
    __device__ __inline__ inline float4 dif(const float4 v1, const float4 v2)
    {
        return { v1.x - v2.x, v1.y - v2.y, v1.z - v2.z, v1.w - v2.w };
    }
    __device__ __inline__ inline float4 mul(const float s, const float4 v)
    {
        return { v.x * s, v.y * s, v.z * s, v.w * s };
    }
    __device__ __inline__ inline float4 div(const float s, const float4 v)
    {
        return { v.x / s, v.y / s, v.z / s, v.w / s };
    }
    __device__ __inline__ inline float len2(const float4 v)
    {
        return v.x * v.x + v.y * v.y + v.z * v.z;
    }
    __device__ __inline__ inline float len(const float4 v)
    {
        return sqrtf(len2(v));
    }
    __device__ __inline__ inline float4 nor(const float4 v)
    {
        float s = len(v);
        return { v.x / s, v.y / s, v.z / s, v.w };
    }
    __device__ __inline__ inline float4 nor(const float4 v, float& s)
    {
        s = len(v);
        return { v.x / s, v.y / s, v.z / s, v.w };
    }


    __device__ __inline__ inline float dot(const float4 v1, const float4 v2)
    {
        return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    }
    __device__ __inline__ inline float4 cross(const float4 v1, const float4 v2)
    {
        float4 v;
        v.x = v1.y * v2.z - v2.y * v1.z;
        v.y = v1.z * v2.x - v2.z * v1.x;
        v.z = v1.x * v2.y - v2.x * v1.y;
        return v;
    }
    __device__ __inline__ inline float4 reflect(const float4 nor, const float4 dir)
    {
        return dif(dir, mul(2 * dot(dir, nor), nor));
    }


    __device__ __inline__ inline float3 sum(const float3 v1, const float3 v2)
    {
        return { v1.x + v2.x, v1.y + v2.y, v1.z + v2.z };
    }
    __device__ __inline__ inline float3 dif(const float3 v1, const float3 v2)
    {
        return { v1.x - v2.x, v1.y - v2.y, v1.z - v2.z };
    }
    __device__ __inline__ inline float3 mul(const float s, const float3 v)
    {
        return { v.x * s, v.y * s, v.z * s };
    }
    __device__ __inline__ inline float3 div(const float s, const float3 v)
    {
        return { v.x / s, v.y / s, v.z / s };
    }

    __device__ __inline__ inline float2 sum(const float2 v1, const float2 v2)
    {
        return { v1.x + v2.x, v1.y + v2.y };
    }
    __device__ __inline__ inline float2 dif(const float2 v1, const float2 v2)
    {
        return { v1.x - v2.x, v1.y - v2.y };
    }
    __device__ __inline__ inline float2 mul(const float s, const float2 v)
    {
        return { v.x * s, v.y * s };
    }
    __device__ __inline__ inline float2 div(const float s, const float2 v)
    {
        return { v.x / s, v.y / s };
    }


    //------------------display-----------------------

    __constant__ uchar4* screen = NULL;
    __constant__ float* zbuffer = NULL;
    __constant__ int screenw, screenh;

    __device__ int mtx = 0;
    __device__ __inline__ inline uchar4 cconv(float4 c)
    {
        float v = max(max(0.0f, c.x), max(c.y, c.z));
        if (v > 1)
        {
            v = 1 - expf(1 - v);
            c.x += v; c.y += v; c.z += v;
        }
        uchar4 r; uint t;
        t = (uint)(c.x * 0xff); r.x = t > 0xff ? 0xff : t;
        t = (uint)(c.y * 0xff); r.y = t > 0xff ? 0xff : t;
        t = (uint)(c.z * 0xff); r.z = t > 0xff ? 0xff : t;
        return r;
    }
    __device__ __inline__ inline bool inframe(const int x, const int y, const float z)
    {
        if ((x < 0 || x >= screenw) || (y < 0 || y > screenh))
            return false;
        return z < zbuffer[x + screenw * y];
    }
    __device__ __inline__ inline void insert(const int x, const int y, const float z, const float4 c)
    {
        zbuffer[x + screenw * y] = z;
        screen[x + screenw * y] = cconv(c);
    }
    __device__ __inline__ inline bool setpx(const int x, const int y, const float z, const float4 c)
    {
        if (!inframe(x, y, z))
            return false;
        insert(x, y, z, c);
        return true;
    }
    

    //------------------coloring---------------------

    __constant__ float4 col = { 1, 1, 1, 1 };
    __device__ inline float4 shade(const float v, const float4 c)
    {
        return { c.x * v, c.y * v, c.z * v, c.w };
    }
    __device__ inline void drawline(const float4 v1, const float4 v2, const float4 c)
    {
        float x0 = screenw / 2, y0 = screenh / 2, s = y0;
        float4 v = v1, dv = dif(v2, v1);
        int n = (int)(max(abs(dv.x), abs(dv.y)) * s);
        dv = div(n, dv);
        while (n--)
        {
            setpx((int)(v.x * s + x0), (int)(v.y * s + y0), v.z, shade(-v.z, c));
            v = sum(v, dv);
        }
    }
    __device__ inline void drawpoint(float4 pos, float4 col, int r)
    {
        col = shade(-pos.z, col);
        float x0 = screenw / 2, y0 = screenh / 2, s = y0;
        int cx = (int)(pos.x * s + x0);
        int cy = (int)(pos.y * s + y0);
        for (int x = cx - r; x < cx + r; x++)
            for (int y = cy - r; y < cy + r; y++)
                setpx(x, y, pos.z, col);
    }

    //------------------animation-------------------
    __constant__ float4 fill_col = { 0, 0, 0, 0 };
    __constant__ float fill_z = MAXFLOAT;
    __global__ void fill_screen(void)
    {
        screen[threadIdx.x + blockIdx.x * blockDim.x] = cconv(fill_col);
    }
    __global__ void fill_zbuffer(void)
    {
        zbuffer[threadIdx.x + blockIdx.x * blockDim.x] = fill_z;
    }
    __global__ void screentest(int x, int r)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int _y = i / r, _x = (x + i % r + screenw + _y) % screenw;
        float c = (float)(i % r) / r;
        setpx(_x, _y, 0, { col.x * c, col.y * c, col.z * c, col.w * c });
    }

    //----------------transformations----------------
    __device__ inline float4 mul(const float4 v, const float* m16)
    {
        float4 r;
        r.x = v.x * m16[0x0] + v.y * m16[0x1] + v.z * m16[0x2] + v.w * m16[0x3];
        r.y = v.x * m16[0x4] + v.y * m16[0x5] + v.z * m16[0x6] + v.w * m16[0x7];
        r.z = v.x * m16[0x8] + v.y * m16[0x9] + v.z * m16[0xa] + v.w * m16[0xb];
        r.w = v.x * m16[0xc] + v.y * m16[0xd] + v.z * m16[0xe] + v.w * m16[0xf];
        return r;
    }
    __device__ inline float3 mul(const float3 v, const float* m16)
    {
        float3 r;
        r.x = v.x * m16[0x0] + v.y * m16[0x1] + v.z * m16[0x2];
        r.y = v.x * m16[0x4] + v.y * m16[0x5] + v.z * m16[0x6];
        r.z = v.x * m16[0x8] + v.y * m16[0x9] + v.z * m16[0xa];
        return r;
    }
    

    __constant__ float m16[16];
    __global__ void transform(float4* ver)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        ver[i] = mul(ver[i], m16);
    }
    __global__ void transform(float3* nor)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        nor[i] = mul(nor[i], m16);
    }


    //------------------rendering-------------------
    texture<float4, cudaTextureType2D> tex0; cudaArray* texarr0;
    __global__ void textest(void)
    {
        float2 t;
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int x = i % screenw; t.x = (float)x / screenw;
        int y = i / screenw; t.y = (float)y / screenh;
        float4 c = tex2D(tex0, t.x, t.y);
        setpx(x, y, 0, c);
    }

    
    __constant__ float lpow = 8;
    __device__ float4 lpos = { 0.6f, 0.6f, -0.6, 1 };
    __device__ void frag_shader(int2 px, float4 v, float4 n, float2 t)
    {
        if (!inframe(px.x, px.y, v.z)) 
            return;

        float4 ldir = dif(lpos, v);
        float l; ldir = nor(ldir, l);
        l = lpow / l;

        float4 edir = nor(v);
        l *= max(0.0f, dot(ldir, n));
        l *= max(0.0f, dot(edir, n));

        float4 c = tex2D(tex0, t.x, t.y);
        insert(px.x, px.y, v.z, shade(l, c));
    }
    __global__ void onevertex(float4 col, int r)
    {
        drawpoint(lpos, col, r);
    }


    __global__ void tria(ushort3* fac, float4* ver, float3* nor, float2* txy)
    {
        const int i = threadIdx.x + blockIdx.x * blockDim.x;
        const int idx = i % quad;
        const int qoff = i / quad;
        const int qidx = qoff % quadn;
        const float sx = screenw / 2, sy = screenh / 2, s = sy;


        //----------register--load-----------------
        struct vtx { float4 v; float4 n; float4 t; };
        __shared__ vtx v[quadn][4];
        if (idx < 3)
        {
            ushort3 f = fac[qoff * 3 + idx];
            v[qidx][idx].v = ver[f.x];
            *((float3*)&(v[qidx][idx].n)) = nor[f.y];
            *((float2*)&(v[qidx][idx].t)) = txy[f.z];
        }
        __syncthreads();

        if (idx == 0)
        {
            if (v[qidx][1].v.y > v[qidx][0].v.y) swap(v[qidx][1], v[qidx][0]);
            if (v[qidx][2].v.y > v[qidx][0].v.y) swap(v[qidx][2], v[qidx][0]);
            if (v[qidx][1].v.y > v[qidx][2].v.y) swap(v[qidx][1], v[qidx][2]);
        }
        __syncthreads();

        float k = (v[qidx][2].v.y - v[qidx][1].v.y) / (v[qidx][0].v.y - v[qidx][1].v.y);
        ((float*)&(v[qidx][3].v.x))[idx] = ((float*)&(v[qidx][1].v.x))[idx] + k * (((float*)&(v[qidx][0].v.x))[idx] - ((float*)&(v[qidx][1].v.x))[idx]);
        ((float*)&(v[qidx][3].n.x))[idx] = ((float*)&(v[qidx][1].n.x))[idx] + k * (((float*)&(v[qidx][0].n.x))[idx] - ((float*)&(v[qidx][1].n.x))[idx]);
        ((float*)&(v[qidx][3].t.x))[idx] = ((float*)&(v[qidx][1].t.x))[idx] + k * (((float*)&(v[qidx][0].t.x))[idx] - ((float*)&(v[qidx][1].t.x))[idx]);
        
        if (idx == 0)
            if (v[qidx][3].v.x < v[qidx][2].v.x) swap(v[qidx][3], v[qidx][2]);
        __syncthreads();
 
        //---------parallel--rasterization---------
        __shared__ vtx vt[quadn][2], dvt[quadn][2], dvx[quadn];
        vtx& v1 = vt[qidx][0], & dv1 = dvt[qidx][0];
        vtx& v2 = vt[qidx][1], & dv2 = dvt[qidx][1];
        vtx vx, & dv = dvx[qidx];
        int2 px, dpx;

        // v[2] v[3] --> v[0]
        px.y = (int)(v[qidx][2].v.y * s + sy);
        dpx.y = (int)(v[qidx][0].v.y * s + sy) - px.y;
        if (idx < 2) vt[qidx][idx] = v[qidx][idx + 2];
        __syncthreads();


        ((float*)&(dv1.v.x))[idx] = (((float*)&(v[qidx][0].v.x))[idx] - ((float*)&(v1.v.x))[idx]) / dpx.y;
        ((float*)&(dv1.n.x))[idx] = (((float*)&(v[qidx][0].n.x))[idx] - ((float*)&(v1.n.x))[idx]) / dpx.y;
        ((float*)&(dv1.t.x))[idx] = (((float*)&(v[qidx][0].t.x))[idx] - ((float*)&(v1.t.x))[idx]) / dpx.y;
        ((float*)&(dv2.v.x))[idx] = (((float*)&(v[qidx][0].v.x))[idx] - ((float*)&(v2.v.x))[idx]) / dpx.y;
        ((float*)&(dv2.n.x))[idx] = (((float*)&(v[qidx][0].n.x))[idx] - ((float*)&(v2.n.x))[idx]) / dpx.y;
        ((float*)&(dv2.t.x))[idx] = (((float*)&(v[qidx][0].t.x))[idx] - ((float*)&(v2.t.x))[idx]) / dpx.y;
        while (dpx.y >= 0)
        {
            px.x = (int)(v1.v.x * s + sx);
            dpx.x = (int)(v2.v.x * s + sx) - px.x;
            ((float*)&(dv.v.x))[idx] = (((float*)&(v2.v.x))[idx] - ((float*)&(v1.v.x))[idx]) / dpx.x;
            ((float*)&(dv.n.x))[idx] = (((float*)&(v2.n.x))[idx] - ((float*)&(v1.n.x))[idx]) / dpx.x;
            ((float*)&(dv.t.x))[idx] = (((float*)&(v2.t.x))[idx] - ((float*)&(v1.t.x))[idx]) / dpx.x;
        
            vx.v = { v1.v.x + dv.v.x * idx, v1.v.y + dv.v.y * idx, v1.v.z + dv.v.z * idx, v1.v.w + dv.v.w * idx };
            vx.n = { v1.n.x + dv.n.x * idx, v1.n.y + dv.n.y * idx, v1.n.z + dv.n.z * idx, 0 };
            vx.t = { v1.t.x + dv.t.x * idx, v1.t.y + dv.t.y * idx, 0, 0 };

            ((float*)&(dv.v.x))[idx] *= quad;
            ((float*)&(dv.n.x))[idx] *= quad;
            ((float*)&(dv.t.x))[idx] *= quad;
            px.x += idx; dpx.x -= idx;
            while (dpx.x >= 0)
            {
                frag_shader(px, vx.v, vx.n, *((float2*)&(vx.t.x)));
                vx.v = { vx.v.x + dv.v.x, vx.v.y + dv.v.y, vx.v.z + dv.v.z, vx.v.w + dv.v.w };
                vx.n = { vx.n.x + dv.n.x, vx.n.y + dv.n.y, vx.n.z + dv.n.z, 0 };
                vx.t = { vx.t.x + dv.t.x, vx.t.y + dv.t.y, 0, 0 };
                px.x += quad; dpx.x -= quad;
            }
            __syncthreads();

            ((float*)&(v1.v.x))[idx] += ((float*)&(dv1.v.x))[idx];
            ((float*)&(v1.n.x))[idx] += ((float*)&(dv1.n.x))[idx];
            ((float*)&(v1.t.x))[idx] += ((float*)&(dv1.t.x))[idx];
            ((float*)&(v2.v.x))[idx] += ((float*)&(dv2.v.x))[idx];
            ((float*)&(v2.n.x))[idx] += ((float*)&(dv2.n.x))[idx];
            ((float*)&(v2.t.x))[idx] += ((float*)&(dv2.t.x))[idx];
            px.y++; dpx.y--;
        }
        __syncthreads();

        // v[2] v[3] --> v[1]
        px.y = (int)(v[qidx][2].v.y * s + sy);
        dpx.y = px.y - (int)(v[qidx][1].v.y * s + sy);
        if (idx < 2) vt[qidx][idx] = v[qidx][idx + 2];
        __syncthreads();

        ((float*)&(dv1.v.x))[idx] = (((float*)&(v[qidx][1].v.x))[idx] - ((float*)&(v1.v.x))[idx]) / dpx.y;
        ((float*)&(dv1.n.x))[idx] = (((float*)&(v[qidx][1].n.x))[idx] - ((float*)&(v1.n.x))[idx]) / dpx.y;
        ((float*)&(dv1.t.x))[idx] = (((float*)&(v[qidx][1].t.x))[idx] - ((float*)&(v1.t.x))[idx]) / dpx.y;
        ((float*)&(dv2.v.x))[idx] = (((float*)&(v[qidx][1].v.x))[idx] - ((float*)&(v2.v.x))[idx]) / dpx.y;
        ((float*)&(dv2.n.x))[idx] = (((float*)&(v[qidx][1].n.x))[idx] - ((float*)&(v2.n.x))[idx]) / dpx.y;
        ((float*)&(dv2.t.x))[idx] = (((float*)&(v[qidx][1].t.x))[idx] - ((float*)&(v2.t.x))[idx]) / dpx.y;
        while (dpx.y >= 0)
        {
            px.x = (int)(v1.v.x * s + sx);
            dpx.x = (int)(v2.v.x * s + sx) - px.x;
            ((float*)&(dv.v.x))[idx] = (((float*)&(v2.v.x))[idx] - ((float*)&(v1.v.x))[idx]) / dpx.x;
            ((float*)&(dv.n.x))[idx] = (((float*)&(v2.n.x))[idx] - ((float*)&(v1.n.x))[idx]) / dpx.x;
            ((float*)&(dv.t.x))[idx] = (((float*)&(v2.t.x))[idx] - ((float*)&(v1.t.x))[idx]) / dpx.x;
            
            vx.v = { v1.v.x + dv.v.x * idx, v1.v.y + dv.v.y * idx, v1.v.z + dv.v.z * idx, v1.v.w + dv.v.w * idx };
            vx.n = { v1.n.x + dv.n.x * idx, v1.n.y + dv.n.y * idx, v1.n.z + dv.n.z * idx, 0 };
            vx.t = { v1.t.x + dv.t.x * idx, v1.t.y + dv.t.y * idx, 0, 0 };

            ((float*)&(dv.v.x))[idx] *= quad;
            ((float*)&(dv.n.x))[idx] *= quad;
            ((float*)&(dv.t.x))[idx] *= quad;
            px.x += idx; dpx.x -= idx;
            while (dpx.x >= 0)
            {
                frag_shader(px, vx.v, vx.n, *((float2*)&(vx.t.x)));
                vx.v = { vx.v.x + dv.v.x, vx.v.y + dv.v.y, vx.v.z + dv.v.z, vx.v.w + dv.v.w };
                vx.n = { vx.n.x + dv.n.x, vx.n.y + dv.n.y, vx.n.z + dv.n.z, 0 };
                vx.t = { vx.t.x + dv.t.x, vx.t.y + dv.t.y, 0, 0 };
                px.x += quad; dpx.x -= quad;
            }
            __syncthreads();
            ((float*)&(v1.v.x))[idx] += ((float*)&(dv1.v.x))[idx];
            ((float*)&(v1.n.x))[idx] += ((float*)&(dv1.n.x))[idx];
            ((float*)&(v1.t.x))[idx] += ((float*)&(dv1.t.x))[idx];
            ((float*)&(v2.v.x))[idx] += ((float*)&(dv2.v.x))[idx];
            ((float*)&(v2.n.x))[idx] += ((float*)&(dv2.n.x))[idx];
            ((float*)&(v2.t.x))[idx] += ((float*)&(dv2.t.x))[idx];
            px.y--; dpx.y--;
        }
    }
    __global__ void triangles(ushort3* fac, float4* ver, float3* nor, float2* txy)
    {
        int idx = threadIdx.x;
        int sidx = blockIdx.x * 3;
        float x0 = screenw / 2, y0 = screenh / 2, s = y0;


        //----------register--load-----------------
        ushort3 f1 = fac[sidx + 0]; float4 v1 = ver[f1.x];
        ushort3 f2 = fac[sidx + 1]; float4 v2 = ver[f2.x];
        ushort3 f3 = fac[sidx + 2]; float4 v3 = ver[f3.x];
        if (v2.y > v1.y) { swap(v2, v1); swap(f2, f1); }
        if (v3.y > v1.y) { swap(v3, v1); swap(f3, f1); }
        if (v2.y > v3.y) { swap(v2, v3); swap(f2, f3); }
        float3 n1 = nor[f1.y]; float2 t1 = txy[f1.z];
        float3 n2 = nor[f2.y]; float2 t2 = txy[f2.z];
        float3 n3 = nor[f3.y]; float2 t3 = txy[f3.z];

        float k = (v3.y - v2.y) / (v1.y - v2.y);
        float4 v0 = sum(v2, mul(k, dif(v1, v2)));
        float3 n0 = sum(n2, mul(k, dif(n1, n2)));
        float2 t0 = sum(t2, mul(k, dif(t1, t2)));
        if (v3.x > v0.x)
        {
            swap(v3, v0);
            swap(n3, n0);
            swap(t3, t0);
        }

        //---------parallel--rasterization---------
        int step = blockDim.x;
        float4 va, dva; float3 na, dna; float2 ta, dta;
        float4 vb, dvb; float3 nb, dnb; float2 tb, dtb;
        int x, xb, y, dy;


        // v3 v0 --> v1
        y = (int)(v3.y * s + y0);
        dy = (int)(v1.y * s + y0) - y;
        va = v3; dva = div(dy, dif(v1, va));
        vb = v0; dvb = div(dy, dif(v1, vb));
        na = n3; dna = div(dy, dif(n1, na));
        nb = n0; dnb = div(dy, dif(n1, nb));
        ta = t3; dta = div(dy, dif(t1, ta));
        tb = t0; dtb = div(dy, dif(t1, tb));
        while (dy-- >= 0)
        {
            x = (int)(va.x * s + x0);
            xb = (int)(vb.x * s + x0);
            float4 v = va, dv = div(xb - x, dif(vb, va));
            float3 n = na, dn = div(xb - x, dif(nb, na));
            float2 t = ta, dt = div(xb - x, dif(tb, ta));
            x += idx;
            v = sum(v, mul(idx, dv)); dv = mul(step, dv);
            n = sum(n, mul(idx, dn)); dn = mul(step, dn);
            t = sum(t, mul(idx, dt)); dt = mul(step, dt);
            while (x <= xb)
            {
                frag_shader({ x, y }, v, { n.x, n.y, n.z, 0 }, t);
                v = sum(v, dv); n = sum(n, dn); t = sum(t, dt);
                x += step;
            }
            va = sum(va, dva); vb = sum(vb, dvb);
            na = sum(na, dna); nb = sum(nb, dnb);
            ta = sum(ta, dta); tb = sum(tb, dtb);
            y++;
        }
        // v3 v0 --> v2
        y = (int)(v3.y * s + y0);
        dy = y - (int)(v2.y * s + y0);
        va = v3; dva = div(dy, dif(v2, va));
        vb = v0; dvb = div(dy, dif(v2, vb));
        na = n3; dna = div(dy, dif(n2, na));
        nb = n0; dnb = div(dy, dif(n2, nb));
        ta = t3; dta = div(dy, dif(t2, ta));
        tb = t0; dtb = div(dy, dif(t2, tb));
        while (dy-- >= 0)
        {
            x = (int)(va.x * s + x0);
            xb = (int)(vb.x * s + x0);
            float4 v = va, dv = div(xb - x, dif(vb, va));
            float3 n = na, dn = div(xb - x, dif(nb, na));
            float2 t = ta, dt = div(xb - x, dif(tb, ta));
            x += idx;
            v = sum(v, mul(idx, dv)); dv = mul(step, dv);
            n = sum(n, mul(idx, dn)); dn = mul(step, dn);
            t = sum(t, mul(idx, dt)); dt = mul(step, dt);
            while (x <= xb)
            {
                frag_shader({ x, y }, v, { n.x, n.y, n.z, 0 }, t);
                v = sum(v, dv); n = sum(n, dn); t = sum(t, dt);
                x += step;
            }
            va = sum(va, dva); vb = sum(vb, dvb);
            na = sum(na, dna); nb = sum(nb, dnb);
            ta = sum(ta, dta); tb = sum(tb, dtb);
            y--;
        }

    }
    __global__ void triline(ushort3* fac, int count, float4* ver, float3* nor)
    {
        int i = (threadIdx.x + blockIdx.x * blockDim.x) * 3;
        if (i >= count) return;
        float4 v1 = ver[fac[i + 0].x];
        float4 v2 = ver[fac[i + 1].x];
        float4 v3 = ver[fac[i + 2].x];
        drawline(v1, v2, col);
        drawline(v2, v3, col);
        drawline(v3, v1, col);

        float3 n1 = nor[fac[i + 0].y];
        float3 n2 = nor[fac[i + 1].y];
        float3 n3 = nor[fac[i + 2].y];

        float3 nn = div(3, sum(n1, sum(n2, n3)));
        float4 n = { nn.x, nn.y, nn.z, 0 };
        float4 v = div(3, sum(v1, sum(v2, v3)));

        float4 d1 = cuda::nor(dif(lpos, v));
        float4 d2 = cuda::nor(reflect(d1, n));
        drawline(v, sum(v, mul(0.1f, d2)), { 1, 0, 0, 1 });
        drawline(v, sum(v, mul(0.1f, d1)), { 1, 1, 0, 1 });
        drawline(v, sum(v, mul(0.1f, n)), { 0, 0, 1, 1 });
    }
    __global__ void vertices(float4* ver, int count)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= count) return;
        float4 t = ver[i];
        int x0 = screenw / 2, y0 = screenh / 2, s = y0;
        int x = (int)(t.x * s + x0);
        int y = (int)(t.y * s + y0);
        setpx(x, y, t.z, col);
    }
};
using namespace cuda;

namespace matrix
{
#define identity {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}

    void mult(float* b, float* a)
    {
        float t[16];
        t[0x0] = a[0x0] * b[0x0] + a[0x1] * b[0x1] + a[0x2] * b[0x2] + a[0x3] * b[0x3];
        t[0x1] = a[0x4] * b[0x0] + a[0x5] * b[0x1] + a[0x6] * b[0x2] + a[0x7] * b[0x3];
        t[0x2] = a[0x8] * b[0x0] + a[0x9] * b[0x1] + a[0xa] * b[0x2] + a[0xb] * b[0x3];
        t[0x3] = a[0xc] * b[0x0] + a[0xd] * b[0x1] + a[0xe] * b[0x2] + a[0xf] * b[0x3];

        t[0x4] = a[0x0] * b[0x4] + a[0x1] * b[0x5] + a[0x2] * b[0x6] + a[0x3] * b[0x7];
        t[0x5] = a[0x4] * b[0x4] + a[0x5] * b[0x5] + a[0x6] * b[0x6] + a[0x7] * b[0x7];
        t[0x6] = a[0x8] * b[0x4] + a[0x9] * b[0x5] + a[0xa] * b[0x6] + a[0xb] * b[0x7];
        t[0x7] = a[0xc] * b[0x4] + a[0xd] * b[0x5] + a[0xe] * b[0x6] + a[0xf] * b[0x7];

        t[0x8] = a[0x0] * b[0x8] + a[0x1] * b[0x9] + a[0x2] * b[0xa] + a[0x3] * b[0xb];
        t[0x9] = a[0x4] * b[0x8] + a[0x5] * b[0x9] + a[0x6] * b[0xa] + a[0x7] * b[0xb];
        t[0xa] = a[0x8] * b[0x8] + a[0x9] * b[0x9] + a[0xa] * b[0xa] + a[0xb] * b[0xb];
        t[0xb] = a[0xc] * b[0x8] + a[0xd] * b[0x9] + a[0xe] * b[0xa] + a[0xf] * b[0xb];

        t[0xc] = a[0x0] * b[0xc] + a[0x1] * b[0xd] + a[0x2] * b[0xe] + a[0x3] * b[0xf];
        t[0xd] = a[0x4] * b[0xc] + a[0x5] * b[0xd] + a[0x6] * b[0xe] + a[0x7] * b[0xf];
        t[0xe] = a[0x8] * b[0xc] + a[0x9] * b[0xd] + a[0xa] * b[0xe] + a[0xb] * b[0xf];
        t[0xf] = a[0xc] * b[0xc] + a[0xd] * b[0xd] + a[0xe] * b[0xe] + a[0xf] * b[0xf];
        memcpy(b, t, 16 * sizeof(float));
    }
    void rotx(float* mat, float rot)
    {
        float c = cosf(rot), s = sinf(rot);
        float t[16] = {
            1,  0,  0,  0,
            0, +c, -s,  0,
            0, +s, +c,  0,
            0,  0,  0,  1
        };
        mult(mat, t);
    }
    void roty(float* mat, float rot)
    {
        float c = cosf(rot), s = sinf(rot);
        float t[16] = {
            +c, 0, -s,  0,
            0,  1,  0,  0,
            +s, 0, +c,  0,
            0,  0,  0,  1
        };
        mult(mat, t);
    }
    void rotz(float* mat, float rot)
    {
        float c = cosf(rot), s = sinf(rot);
        float t[16] = {
            +c, -s, 0,  0,
            +s, +c, 0,  0,
            0,  0,  1,  0,
            0,  0,  0,  1
        };
        mult(mat, t);
    }
    void scale(float* mat, float x, float y, float z)
    {
        float t[16] = {
            x,  0,  0,  0,
            0,  y,  0,  0,
            0,  0,  z,  0,
            0,  0,  0,  1
        };
        mult(mat, t);
    }
    void scale(float* mat, float s)
    {
        scale(mat, s, s, s);
    }
    void move(float* mat, float x, float y, float z)
    {
        float t[16] = {
            1,  0,  0,  x,
            0,  1,  0,  y,
            0,  0,  1,  z,
            0,  0,  0,  1
        };
        mult(mat, t);
    }
};
using namespace matrix;


//_____________________________________________________________________________
//__________________________________GPU__SHIT__________________________________
int2 wsize = { 560, 480 };
int2 wpos = { 200, 50 };
uchar4* dev_screen;
size_t screensize;

GLuint vbo;
struct cudaGraphicsResource* cuda_vbo_resource;
void createVBO(void)
{
    unsigned int size = wsize.x * wsize.y * sizeof(uchar4);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, vbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, size, NULL, GL_DYNAMIC_DRAW);
    cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard);
}
void deleteVBO(void)
{
    cudaGraphicsUnregisterResource(cuda_vbo_resource);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, vbo);
    glDeleteBuffers(1, &vbo);
    vbo = 0;
}


//_____________________________________________________________________________
//__________________________________MAIN_______________________________________
model mod;
material mtl1;
// cub.model tria.model kiti2.model
string src0 = "model/kiti2.model"; 
string src1 = "model/m1.mat";
string src2 = "model/m2.mat";
string src3 = "model/m3.mat";


int x = 0, r = 16;
float* dev_zbuffer;
float4* dev_ver;
float3* dev_nor;
float2* dev_txy;
ushort3* dev_fac;
float4* dev_tex;
float4* dev_lpos;


void testdisplay()
{
    x = (x + 1) % wsize.x;
    cudacall(screentest, r * r, x, r);
}
void cleardisplay()
{
    cudacall(fill_screen, wsize.x * wsize.y);
    cudacall(fill_zbuffer, wsize.x * wsize.y);
}


float rotx0[16] = identity, rotx1[16] = identity; uchar2 keyUD = { 0, 0 };
float roty0[16] = identity, roty1[16] = identity; uchar2 keyLR = { 0, 0 };
float rotz0[16] = identity, rotz1[16] = identity; uchar2 keyAS = { 0, 0 };

float movx0[16] = identity, movx1[16] = identity; uchar2 keyAD = { 0, 0 };
float movy0[16] = identity, movy1[16] = identity; uchar2 keyWS = { 0, 0 };
float movz0[16] = identity, movz1[16] = identity; uchar2 keyQE = { 0, 0 };

uchar2 key46 = { 0, 0 }, key28 = { 0, 0 }, key79 = { 0, 0 };


float scal_amp = 1;
float host_lpow = 1, lpow_amp = 1;
void idle(void)
{
    cudaDeviceSynchronize();
    host_lpow *= lpow_amp;
    cudaMemcpyToSymbol(lpow, &host_lpow, sizeof(float));


    float mat[16] = identity;
    if (keyUD.x) mult(mat, rotx0); if (keyUD.y) mult(mat, rotx1);
    if (keyLR.x) mult(mat, roty0); if (keyLR.y) mult(mat, roty1);
    if (keyAS.x) mult(mat, rotz0); if (keyAS.y) mult(mat, rotz1);
    
    if (keyAD.x) mult(mat, movx0); if (keyAD.y) mult(mat, movx1);
    if (keyWS.x) mult(mat, movy0); if (keyWS.y) mult(mat, movy1);
    if (keyQE.x) mult(mat, movz0); if (keyQE.y) mult(mat, movz1);
    scale(mat, scal_amp);

    cudaMemcpyToSymbol(m16, mat, sizeof(float) * 16);
    cudacall(transform, (int)mod.v.size(), dev_ver);
    cudacall(transform, (int)mod.v.size(), dev_nor);
    cudacall(transform, 1, dev_lpos);


    float lmat[16] = identity;
    if (key46.x) mult(lmat, movx1); if (key46.y) mult(lmat, movx0);
    if (key28.x) mult(lmat, movy1); if (key28.y) mult(lmat, movy0);
    if (key79.x) mult(lmat, movz1); if (key79.y) mult(lmat, movz0);

    cudaMemcpyToSymbol(m16, lmat, sizeof(float) * 16);
    cudacall(transform, 1, dev_lpos);


    glutPostRedisplay();
}
void display(void)
{
    cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&dev_screen, &screensize, cuda_vbo_resource);
    cudaMemcpyToSymbol(screen, &dev_screen, sizeof(uchar4*));


    cleardisplay();
    //cudacall(tria, (int)mod.f.size() / 3 * quad, dev_fac, dev_ver, dev_nor, dev_txy);
    cudacall(triangles, (int)mod.f.size() / 3 * cores, dev_fac, dev_ver, dev_nor, dev_txy);
    //cudacall(triline, (int)mod.f.size() / 3, dev_fac, (int)mod.f.size(), dev_ver, dev_nor);
    //cudacall(textest, wsize.x * wsize.y);
    cudacall(onevertex, 1, { 1, 0, 0, 1 }, 4);
    
    testdisplay();
    cudaDeviceSynchronize();
    cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);


    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawPixels(wsize.x, wsize.y, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glutSwapBuffers();
}
void reshape(int w, int h)
{
    cudaDeviceSynchronize();
    deleteVBO();
    wsize = { w, h };
    createVBO();

    cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&dev_screen, &screensize, cuda_vbo_resource);
    cudaMemcpyToSymbol(screen, &dev_screen, sizeof(uchar4*));
    cudaMemcpyToSymbol(screenw, &wsize.x, sizeof(int));
    cudaMemcpyToSymbol(screenh, &wsize.y, sizeof(int));
    
    cudaFree(dev_zbuffer);
    cudaMalloc(&dev_zbuffer, screensize);
    cudaMemcpyToSymbol(zbuffer, &dev_zbuffer, sizeof(float*));
}

void load(void)
{
    cudaGetSymbolAddress((void**)&dev_lpos, lpos);

    loadmodel(src0, &mod);
    cudaMalloc(&dev_ver, sizeof(float4) * mod.v.size());
    cudaMemcpy(dev_ver, &mod.v[0], sizeof(float4) * mod.v.size(), cudaMemcpyHostToDevice);
    cudaMalloc(&dev_nor, sizeof(float3) * mod.n.size());
    cudaMemcpy(dev_nor, &mod.n[0], sizeof(float3) * mod.n.size(), cudaMemcpyHostToDevice);
    cudaMalloc(&dev_txy, sizeof(float2) * mod.t.size());
    cudaMemcpy(dev_txy, &mod.t[0], sizeof(float2) * mod.t.size(), cudaMemcpyHostToDevice);
    cudaMalloc(&dev_fac, sizeof(ushort3) * mod.f.size());
    cudaMemcpy(dev_fac, &mod.f[0], sizeof(ushort3) * mod.f.size(), cudaMemcpyHostToDevice);


    loadmaterial(src3, &mtl1);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    tex0.addressMode[0] = cudaAddressModeClamp;
    tex0.addressMode[1] = cudaAddressModeClamp;
    tex0.filterMode = cudaFilterModeLinear;
    tex0.normalized = true;

    cudaMallocArray(&texarr0, &channelDesc, mtl1.w, mtl1.h);    
    cudaMemcpy2DToArray(texarr0, 0, 0, mtl1.tex, mtl1.w * sizeof(float4), mtl1.w * sizeof(float4), mtl1.h, cudaMemcpyHostToDevice);
    cudaBindTextureToArray(&tex0, texarr0, &channelDesc);
}
void init(void)
{
    float mat[16] = identity;
    scale(mat, 0.75f);
    cudaMemcpyToSymbol(m16, mat, sizeof(float) * 16);
    cudacall(transform, (int)mod.v.size(), dev_ver);
    cudacall(transform, (int)mod.n.size(), dev_nor);
    

    float rot = 0.02f;
    rotx(rotx0, -rot); rotx(rotx1, +rot);
    roty(roty0, -rot); roty(roty1, +rot);
    rotz(rotz0, -rot); rotz(rotz1, +rot);

    float mov = 0.02f;
    move(movx0, -mov, 0, 0); move(movx1, +mov, 0, 0);
    move(movy0, 0, -mov, 0); move(movy1, 0, +mov, 0);
    move(movz0, 0, 0, -mov); move(movz1, 0, 0, +mov);
}
void release(void)
{
    cudaUnbindTexture(tex0);
    cudaFreeArray(texarr0);
    cudaFree(dev_ver);
    cudaFree(dev_nor);
    cudaFree(dev_txy);
    cudaFree(dev_fac);
    mod.release();
    mtl1.release();
    cudaFree(dev_tex);
    deleteVBO();
}


void keyboardproc(uchar key, int x, int y, uchar stat)
{
    switch (key)
    {
    case 27:    exit(0); return;
    case '+':   scal_amp = stat ? 1.01f : 1; return;
    case '-':   scal_amp = stat ? 0.99f : 1; return;

    case 'd':   keyAD.x = stat; return;
    case 'a':   keyAD.y = stat; return;
    case 'w':   keyWS.x = stat; return;
    case 's':   keyWS.y = stat; return;
    case 'q':   keyQE.x = stat; return;
    case 'e':   keyQE.y = stat; return;

    case '6':   key46.x = stat; return;
    case '4':   key46.y = stat; return;
    case '8':   key28.x = stat; return;
    case '2':   key28.y = stat; return;
    case '9':   key79.x = stat; return;
    case '7':   key79.y = stat; return;

    case '3':   lpow_amp = stat ? 1.01f : 1; return;
    case '1':   lpow_amp = stat ? 0.99f : 1; return;
    }
}
void keyboardprocdown(uchar key, int x, int y)
{
    keyboardproc(key, x, y, 1);
}
void keyboardprocup(uchar key, int x, int y)
{
    keyboardproc(key, x, y, 0);
}


void specialproc(int key, int x, int y, uchar stat)
{
    switch (key)
    {
    case GLUT_KEY_LEFT:     keyLR.x = stat; return;
    case GLUT_KEY_RIGHT:    keyLR.y = stat; return;
    case GLUT_KEY_UP:       keyUD.x = stat; return;
    case GLUT_KEY_DOWN:     keyUD.y = stat; return;
    }
}
void specialprocdown(int key, int x, int y)
{
    specialproc(key, x, y, 1);
}
void specialprocup(int key, int x, int y)
{
    specialproc(key, x, y, 0);
}


int main(int argc, char** argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(wsize.x, wsize.y);
    glutInitWindowPosition(wpos.x, wpos.y);
    glutCreateWindow("Test");

    
    glutDisplayFunc(display);
    glutIdleFunc(idle);
    glutCloseFunc(release);
    glutKeyboardFunc(keyboardprocdown);
    glutKeyboardUpFunc(keyboardprocup);
    glutSpecialFunc(specialprocdown);
    glutSpecialUpFunc(specialprocup);
    glutReshapeFunc(reshape);


    load();
    init();
    
    
    glewInit();
    glutMainLoop();
}