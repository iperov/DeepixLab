#include "glsl/glsl.h"
#include "glsl/index.h"
#include "glsl/hash.h"
#include "glsl/sd.h"

#include "gen_ispc.h"

extern int _fltused;
int _fltused = 0;
extern "C"  size_t __chkstk(size_t size)
{
    __asm__(
        "movq $0x0, %r11 \r\n"
        "movl $0xFFFFFFFF, %r11d \r\n"
        "and %r11, %rax \r\n"
        "retn \r\n"
    );
    return 0;
}

extern "C" __declspec(dllexport) void c_test_gen(float* out, uint32_t W, uint32_t H, uint32_t seed)
{
    ispc::test_gen(out, W, H, seed );
}

extern "C" __declspec(dllexport) void c_noise(float* img, uint32_t size, uint32_t seed)
{
    ispc::noise(img, size, seed);
}

extern "C" __declspec(dllexport) void c_bezier(float* img, uint32_t W, uint32_t H, float ax, float ay, float bx, float by, float cx, float cy, float width)
{
    ispc::bezier(img, W, H, ax, ay, bx, by, cx, cy, width);
}

extern "C" __declspec(dllexport) void c_bezier_inner_area(float* img, int32_t W, int32_t H, float ax, float ay, float bx, float by, float cx, float cy)
{
    for (int32_t y=0; y<H; ++y)
    for (int32_t x=0; x<W; ++x)
    { 
        vec2 pt = vec2(x, y);
        float dist = sd_bezier(pt, vec2(ax, ay), vec2(bx, by), vec2(cx, cy) );
        img[ IDX2D(H,W,y,x) ] = dist >= 0 ? 1.0f : 0.0f;
    }
}

extern "C" __declspec(dllexport) void c_circle_faded(float* img, uint32_t W, uint32_t H, float cx, float cy, float fs, float fe)
{
    ispc::circle_faded(img, W, H, cx, cy, fs, fe);
}

extern "C" __declspec(dllexport) void c_cut_edges_mask(float* img, uint32_t W, uint32_t H, float angle_deg, float edge_dist, float cx, float cy, float cw, float ch, bool init)
{
    ispc::cut_edges_mask(img, W, H, angle_deg, edge_dist, cx, cy, cw, ch, init);
}

extern "C" __declspec(dllexport) void c_icon_loading(float* img, uint32_t H, uint32_t W, uint32_t C, float R_inner, float R_outter, float edge_smooth, 
                                                     float* bg_color, float* fg_color, float u_time)
{    
    ispc::icon_loading(img, H, W, C, R_inner, R_outter, edge_smooth, bg_color, fg_color, u_time);
}
