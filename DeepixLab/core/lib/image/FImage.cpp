#include "glsl/glsl.h"
#include "glsl/index.h"
#include "glsl/color.h"
#include "FImage_ispc.h"

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

extern "C" __declspec(dllexport) void c_u8_to_f32(  float* O, uint8_t* I, uint32_t size) { ispc::u8_to_f32(O,I,size); }
extern "C" __declspec(dllexport) void c_f32_to_u8(  uint8_t* O, float* I, uint32_t size) { ispc::f32_to_u8(O,I,size); }

extern "C" __declspec(dllexport) void c_mul_u8_u8(uint8_t* O, uint8_t* A, uint8_t* B, uint32_t size) { ispc::mul_u8_u8(O, A, B, size); }
extern "C" __declspec(dllexport) void c_mul_u8_f32(uint8_t* O, uint8_t* A, float* B, uint32_t size) { ispc::mul_u8_f32(O, A, B, size); }
extern "C" __declspec(dllexport) void c_mul_f32_u8(float* O, float* A, uint8_t* B, uint32_t size) { ispc::mul_f32_u8(O, A, B, size); }
extern "C" __declspec(dllexport) void c_mul_f32_f32(float* O, float* A, float* B, uint32_t size) { ispc::mul_f32_f32(O, A, B, size); }

extern "C" __declspec(dllexport) void c_sub_u8_u8(uint8_t* O, uint8_t* A, uint8_t* B, uint32_t size) { ispc::sub_u8_u8(O, A, B, size); }
extern "C" __declspec(dllexport) void c_sub_u8_f32(uint8_t* O, uint8_t* A, float* B, uint32_t size) { ispc::sub_u8_f32(O, A, B, size); }
extern "C" __declspec(dllexport) void c_sub_f32_u8(float* O, float* A, uint8_t* B, uint32_t size) { ispc::sub_f32_u8(O, A, B, size); }
extern "C" __declspec(dllexport) void c_sub_f32_f32(float* O, float* A, float* B, uint32_t size) { ispc::sub_f32_f32(O, A, B, size); }

extern "C" __declspec(dllexport) void c_add_u8_u8(uint8_t* O, uint8_t* A, uint8_t* B, uint32_t size) { ispc::add_u8_u8(O, A, B, size); }
extern "C" __declspec(dllexport) void c_add_u8_f32(uint8_t* O, uint8_t* A, float* B, uint32_t size) { ispc::add_u8_f32(O, A, B, size); }
extern "C" __declspec(dllexport) void c_add_f32_u8(float* O, float* A, uint8_t* B, uint32_t size) { ispc::add_f32_u8(O, A, B, size); }
extern "C" __declspec(dllexport) void c_add_f32_f32(float* O, float* A, float* B, uint32_t size) { ispc::add_f32_f32(O, A, B, size); }


extern "C" __declspec(dllexport) void c_histogram_u8_uint32( uint32_t* hist_out, uint8_t* img, int32_t HW, int32_t C)
{
    for (int32_t c=0; c<C; ++c)
    {
        for (int32_t bin=0; bin<256; ++bin)
            hist_out[IDX2D(C,256,c,bin)] = 0;
        
        for (int32_t hw=0; hw<HW;++hw)
            hist_out[IDX2D(C,256,c,img[IDX2D(HW,C,hw,c)])] += 1;
    }
}

extern "C" __declspec(dllexport) void c_histogram_u8_f32( float* hist_out, uint8_t* img, int32_t HW, int32_t C)
{
    uint32_t* int_hist_out = (uint32_t*) hist_out;
    
    for (int32_t c=0; c<C; ++c)
    {
        for (int32_t bin=0; bin<256; ++bin)
            int_hist_out[IDX2D(C,256,c,bin)] = 0;
        
        for (int32_t hw=0; hw<HW;++hw)
            int_hist_out[IDX2D(C,256,c,img[IDX2D(HW,C,hw,c)])] += 1;
    }
    
    for (int32_t c=0; c<C; ++c)
    for (int32_t bin=0; bin<256; ++bin)
        hist_out[ IDX2D(C,256,c,bin) ] = (float)int_hist_out[ IDX2D(C,256,c,bin) ] / (float)HW;
}

extern "C" __declspec(dllexport) void c_channel_exposure_u8(uint8_t* O, uint8_t* I, uint32_t H, uint32_t W, uint32_t C, float* exposure) { ispc::channel_exposure_u8(O, I, H, W, C, exposure); }
extern "C" __declspec(dllexport) void c_channel_exposure_f32(float* O, float* I, uint32_t H, uint32_t W, uint32_t C, float* exposure) { ispc::channel_exposure_f32(O, I, H, W, C, exposure); }


extern "C" __declspec(dllexport) void c_levels_u8(uint8_t* O, uint8_t* I, uint32_t H, uint32_t W, uint32_t C, 
                                                  float* in_b, float* in_w, float* in_g, float* out_b, float* out_w)
{    
    ispc::levels_u8(O, I, H, W, C, in_b, in_w, in_g, out_b, out_w);
}


extern "C" __declspec(dllexport) void c_levels_f32(float* O, float* I, uint32_t H, uint32_t W, uint32_t C, 
                                                   float* in_b, float* in_w, float* in_g, float* out_b, float* out_w)
{    
    ispc::levels_f32(O, I, H, W, C, in_b, in_w, in_g, out_b, out_w);
}


extern "C" __declspec(dllexport) void c_hsv_shift_u8(uint8_t* O, uint8_t* I, int32_t H, int32_t W, int32_t C, float h_offset, float s_offset, float v_offset)
{
    ispc::hsv_shift_u8(O, I, H, W, C, h_offset, s_offset, v_offset);
}

extern "C" __declspec(dllexport) void c_hsv_shift_f32(  float* O, float* I, int32_t H, int32_t W, int32_t C, float h_offset, float s_offset, float v_offset)
{
    ispc::hsv_shift_f32(O, I, H, W, C, h_offset, s_offset, v_offset);
}

extern "C" __declspec(dllexport) void c_blend_u8 (  uint8_t* O, int32_t OH, int32_t OW, int32_t OC, 
                                                    uint8_t* A, 
                                                    float* B, int32_t BH, int32_t BW, int32_t BC, 
                                                    float* M, int32_t MH, int32_t MW, int32_t MC, 
                                                    float alpha )
{
    ispc::blend_u8(O, OH, OW, OC, A, B, BH, BW, BC, M, MH, MW, MC, alpha);
}

extern "C" __declspec(dllexport) void c_blend_f32(  float* O, int32_t OH, int32_t OW, int32_t OC, 
                                                    float* A, 
                                                    float* B, int32_t BH, int32_t BW, int32_t BC, 
                                                    float* M, int32_t MH, int32_t MW, int32_t MC, 
                                                    float alpha )
{
    ispc::blend_f32(O, OH, OW, OC, A, B, BH, BW, BC, M, MH, MW, MC, alpha);
}

extern "C" __declspec(dllexport) void c_satushift_u8(uint8_t* O, uint8_t* I, uint32_t size) { ispc::satushift_u8(O, I, size); }
extern "C" __declspec(dllexport) void c_satushift_f32(float* O, float* I, uint32_t size) { ispc::satushift_f32(O, I, size); }
