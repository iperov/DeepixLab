#include "glsl/glsl.h"
#include "glsl/index.h"
#include "glsl/hash.h"

extern "C" __declspec(dllexport) void c_gen_grid(float* out_grid, int32_t OW, int32_t OH, 
                                        int32_t cell_count, float intensity, uint32_t seed)
{
    float cell_var = intensity * (1.0f/cell_count) * 0.4f;
    
    int32_t IW = cell_count;
    int32_t IH = cell_count;
    
    float xs = (float)OW / (float)IW;
    float ys = (float)OH / (float)IH;
    for (int32_t y=0; y<OH; ++y)
    {
        float py_f  = (float)y / ys;
        int32_t py  = (int32_t) py_f;
        float py_m  = py_f-py;
        
        for (int x=0; x<OW; ++x)
        {
            float px_f = (float)x / xs;
            int32_t px = (int32_t) px_f;
            float px_m  = px_f-px;
            
            float a00 = (1.0-py_m) * (1.0-px_m);
            float a01 = (1.0-py_m) * px_m;
            float a10 = py_m       * (1.0-px_m);
            float a11 = py_m       * px_m;
            
            int32_t px0 = px;
            int32_t px1 = px+1;
            int32_t py0 = py;
            int32_t py1 = py+1;
            
            bool px0_ok = (px0 > 0 & px0 < IW);
            bool px1_ok = (px1 > 0 & px1 < IW);
            bool py0_ok = (py0 > 0 & py0 < IH);
            bool py1_ok = (py1 > 0 & py1 < IH);
            
            for (int32_t c=0; c<2; ++c)
            {
                
                float v = 0;
                if (py0_ok)
                {
                    v += (1.0f - 2.0f*hashf(seed+px0, py0, c)) *a00 *cell_var * px0_ok;
                    v += (1.0f - 2.0f*hashf(seed+px1, py0, c)) *a01 *cell_var * px1_ok;
                }
                
                if (py1_ok)
                {
                    v += (1.0f - 2.0f*hashf(seed+px0, py1, c)) *a10 *cell_var * px0_ok;
                    v += (1.0f - 2.0f*hashf(seed+px1, py1, c)) *a11 *cell_var * px1_ok;
                }
                
                out_grid[ IDX3D(OH,OW,2,y,x,c) ] = c == 0 ? x + v*OW : y + v*OH;
            }
        }
    }
}
