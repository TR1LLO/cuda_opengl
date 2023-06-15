#include <iostream>
#include <vector>

#include <cuda.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>


#include "c2_engine.cuh"
using namespace std;
using namespace float4_math;

int4 wsize = { 200, 50, 560, 480 };
void reshape(int w, int h)
{
    cudaDeviceSynchronize();
    wsize.z = w;
    wsize.w = h;
    resize(w, h);
}


float rot = -0.05f;
uchar2 keyUD = { 0, 0 }; mat4 rotx0 = rotx(identity, -rot), rotx1 = rotx(identity, +rot);
uchar2 keyLR = { 0, 0 }; mat4 roty0 = roty(identity, -rot), roty1 = roty(identity, +rot);
uchar2 keyRF = { 0, 0 }; mat4 rotz0 = rotz(identity, -rot), rotz1 = rotz(identity, +rot);

float mov = 0.1f;
uchar2 keyAD = { 0, 0 }; mat4 movx0 = move(identity, -mov, 0, 0), movx1 = move(identity, +mov, 0, 0);
uchar2 keyWS = { 0, 0 }; mat4 movy0 = move(identity, 0, -mov, 0), movy1 = move(identity, 0, +mov, 0);
uchar2 keyQE = { 0, 0 }; mat4 movz0 = move(identity, 0, 0, -mov), movz1 = move(identity, 0, 0, +mov);

float scal_amp = 1, lpow_amp = 1;
uchar2 key46 = { 0, 0 }, key28 = { 0, 0 }, key79 = { 0, 0 };

float ftime = 0;
float dftime = 0.95f;
HANDLE console = GetStdHandle(STD_OUTPUT_HANDLE);
void print(mat4 m, int x, int y)
{
    for (int i = 0; i < 16; i++)
    {
        SetConsoleCursorPosition(console, { (short)(x + (i % 4) * 16), (short)(y + i / 4) });
        cout << m.v[i] << "  ";
    }
}
void print(float4 v, int x, int y)
{
    for (int i = 0; i < 4; i++)
    {
        SetConsoleCursorPosition(console, { (short)(x + i * 16), (short)y });
        cout << ((float*)&v)[i] << "  ";
    }
}

mat4 cam_pos = move(identity, 0, 0, 4);


//model_3D mod1(L"model/kiti.model");
//material_2D mtl1(L"model/m4.mat");
//material_2D mtl2(L"model/m2.mat");

//model_3D mod2(L"model/kiti.model");
//material_2D mtl3(L"model/m3.mat");

float4 lpos = { 1, 1, 1, 1 };
float4 lpow = { 0.01f, 0.6f, 256, 1 };

MultiSubmesh mesh0("model/kiti.obj");
Bitmap_fp32 bmp1("model/k0.bmp");
Bitmap_fp32 bmp2("model/k1.bmp");

vector<Renderer*> rnds;

bool anim = true;
void idle(void)
{
    lpow.x *= lpow_amp;

    cam_pos = scale(cam_pos, scal_amp, scal_amp, scal_amp);
    if (keyUD.x) cam_pos = mul(cam_pos, rotx0); if (keyUD.y) cam_pos = mul(cam_pos, rotx1);
    if (keyLR.x) cam_pos = mul(cam_pos, roty0); if (keyLR.y) cam_pos = mul(cam_pos, roty1);
    if (keyRF.x) cam_pos = mul(cam_pos, rotz0); if (keyRF.y) cam_pos = mul(cam_pos, rotz1);

    if (keyAD.x) cam_pos = mul(cam_pos, movx0); if (keyAD.y) cam_pos = mul(cam_pos, movx1);
    if (keyWS.x) cam_pos = mul(cam_pos, movy0); if (keyWS.y) cam_pos = mul(cam_pos, movy1);
    if (keyQE.x) cam_pos = mul(cam_pos, movz0); if (keyQE.y) cam_pos = mul(cam_pos, movz1);
    mat_camera = mul(cam_pos, perspective(4, 0, (float)wsize.w / wsize.z));


    mat4 lmat = inv(cam_pos);
    if (key46.x) lmat = mul(lmat, movx0); if (key46.y) lmat = mul(lmat, movx1);
    if (key28.x) lmat = mul(lmat, movy0); if (key28.y) lmat = mul(lmat, movy1);
    if (key79.x) lmat = mul(lmat, movz0); if (key79.y) lmat = mul(lmat, movz1);
    lpos = mul(lpos, mul(cam_pos, lmat));

    if (anim)
    {
        for (auto rnd : rnds)
            rnd->body->mat_local = mul(rnd->body->mat_local, rnd->body->mat_delta);
    }

    glutPostRedisplay();
}
void draw(void)
{
    cudaDeviceSynchronize();
    map_buffer();
    clear();

    cudaEvent_t e1, e2;
    cudaEventCreate(&e1);
    cudaEventCreate(&e2);
    cudaEventRecord(e1);
    //-----------------------------------------------
    
    for (int i = 0; i < rnds.size(); i++)
        rnds[i]->render();
    //-----------------------------------------------
    
    cudaEventRecord(e2);
    cudaEventSynchronize(e2);
    float time;
    cudaEventElapsedTime(&time, e1, e2);
    SetConsoleCursorPosition(console, { 0, 0 });
    ftime = ftime * dftime + time * (1 - dftime);
    cout << (int)(1000 / ftime) << " fps  " << endl;
    cudaEventDestroy(e1);
    cudaEventDestroy(e2);

    //-----------------------------------------------
    unmap_buffer();
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawPixels(wsize.z, wsize.w, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glutSwapBuffers();
}


#define randf ((float)rand()/RAND_MAX * 2 - 1)
void init(void)
{
    cam_pos = move(rotz(identity, 0), 0, 0, 4);

    mesh0.init();

    bmp1.load_bmp(); 
    bmp1.gpu_init(); bmp1.release();
    bmp2.load_bmp(); 
    bmp2.alpha_cut(bmp2.data[0], 0.6f);
    bmp2.gpu_init(); bmp2.release();


    RendererMultiSubmesh* rnd0 = new RendererMultiSubmesh();
    rnd0->count = mesh0.count; rnd0->init();
    rnd0->body = new Rigidbody(); rnd0->body->init(); 
    MaterialTexture* mtl1 = new MaterialTexture(); mtl1->init();
    mtl1->col = { 1, 0, 0, 0.125f }; mtl1->bmp = &bmp1;
    mtl1->lpos_ptr = &lpos; mtl1->lpow_ptr = &lpow;
    MaterialTexture* mtl2 = new MaterialTexture(); mtl2->init();
    mtl2->col = { 0, 1, 0, 0.125f }; mtl2->bmp = &bmp2;
    mtl2->lpos_ptr = &lpos; mtl2->lpow_ptr = &lpow;
    Renderer* rnd1 = new Renderer(); rnd1->init();
    rnd1->mtl = mtl1; rnd1->body = rnd0->body;
    rnd1->mesh = mesh0.submesh[0]; rnd0->renderer[0] = rnd1;
    Renderer* rnd2 = new Renderer(); rnd2->init();
    rnd2->mtl = mtl2; rnd2->body = rnd0->body;
    rnd2->mesh = mesh0.submesh[1]; rnd0->renderer[1] = rnd2;

    rnd0->body->mat_local = scale(identity, 0.05f, 0.05f, 0.05f);
    rnd0->body->mat_delta = roty0;
    rnds.push_back(rnd0);
    for (int i = 0; i < 6; i++)
    {
        auto rnd = (RendererMultiSubmesh*)rnd0->clone();
        rnd->body->mat_basis = &rnd0->body->transform;
        rnd->body->mat_local = move(scale(identity, 0.8f, 0.8f, 0.8f), randf * 48, randf * 24, randf * 48);
        rnd->body->mat_delta = rotx(rnd->body->mat_delta, randf * rot);
        rnd->body->mat_delta = roty(rnd->body->mat_delta, randf * rot);
        rnd->body->mat_delta = rotz(rnd->body->mat_delta, randf * rot);

        float4 col;
        col.x = (3 + randf) / 4;
        col.y = (3 + randf) / 4;
        col.z = (3 + randf) / 4;
        col.w = 0.125f;
        ((MaterialTexture*)rnd->renderer[0]->mtl)->col = col;
        rnds.push_back(rnd);
    }
}
void release(void)
{
    cudaDeviceSynchronize();
    delete_vbo();

    for (auto rnd : rnds)
    {
        rnd->release();
        delete rnd;
    }
    rnds.clear();

    bmp1.gpu_free();
    bmp2.gpu_free();
    mesh0.release();
}

//----------------------INPUT-------------------------
void keyboardproc       (uchar key, int x, int y, uchar stat)
{
    switch (key)
    {
    case 27:    exit(0); return;
    case '+':   scal_amp = stat ? 1.01f : 1; return;
    case '-':   scal_amp = stat ? 0.99f : 1; return;
    case ' ':   anim = stat ? !anim : anim; return;
    
    case 'd': case 'D': case 'â': case 'Â':   keyAD.x = stat; return;
    case 'a': case 'A': case 'ô': case 'Ô':   keyAD.y = stat; return;
    case 'w': case 'W': case 'ö': case 'Ö':   keyWS.x = stat; return;
    case 's': case 'S': case 'û': case 'Û':   keyWS.y = stat; return;
    case 'q': case 'Q': case 'é': case 'É':   keyQE.x = stat; return;
    case 'e': case 'E': case 'ó': case 'Ó':   keyQE.y = stat; return;
    case 'r': case 'R': case 'ê': case 'Ê':   keyRF.x = stat; return;
    case 'f': case 'F': case 'à': case 'À':   keyRF.y = stat; return;

    case '4':   key46.x = stat; return;
    case '6':   key46.y = stat; return;
    case '2':   key28.x = stat; return;
    case '8':   key28.y = stat; return;
    case '7':   key79.x = stat; return;
    case '9':   key79.y = stat; return;

    case '3':   lpow_amp = stat ? 1.01f : 1; return;
    case '1':   lpow_amp = stat ? 0.99f : 1; return;
    }
}
void keyboardprocdown   (uchar key, int x, int y)
{
    keyboardproc(key, x, y, 1);
}
void keyboardprocup     (uchar key, int x, int y)
{
    keyboardproc(key, x, y, 0);
}

void specialproc        (int key, int x, int y, uchar stat)
{
    switch (key)
    {
    case GLUT_KEY_UP:       keyUD.x = stat; return;
    case GLUT_KEY_DOWN:     keyUD.y = stat; return;
    case GLUT_KEY_RIGHT:    keyLR.x = stat; return;
    case GLUT_KEY_LEFT:     keyLR.y = stat; return;
    }
}
void specialprocdown    (int key, int x, int y)
{
    specialproc(key, x, y, 1);
}
void specialprocup      (int key, int x, int y)
{
    specialproc(key, x, y, 0);
}

//----------------------ENTRY-------------------------
int main(int argc, char** argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(wsize.z, wsize.w);
    glutInitWindowPosition(wsize.x, wsize.y);
    glutCreateWindow("Test");

    
    glutDisplayFunc(draw);
    glutIdleFunc(idle);
    glutCloseFunc(release);
    glutKeyboardFunc(keyboardprocdown);
    glutKeyboardUpFunc(keyboardprocup);
    glutSpecialFunc(specialprocdown);
    glutSpecialUpFunc(specialprocup);
    glutReshapeFunc(reshape);

    init();
    
    
    glewInit();
    glutMainLoop();
}