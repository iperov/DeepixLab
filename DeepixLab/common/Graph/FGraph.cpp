#include "glsl/glsl.h"
#include "glsl/index.h"

extern "C" __declspec(dllexport) void c_draw_bg(float* img, int32_t H, int32_t W)
{    
    for (int32_t y=0; y<H; ++y)
    for (int32_t x=0; x<W; ++x)
    {
        size_t idx = IDX3D(H,W,4,y,x,0);
        img[idx + 0] = 1.0f;
        img[idx + 1] = 1.0f;
        img[idx + 2] = 1.0f;
        
        if ((y % 8) < 2)
            img[idx + 3] = 0.05f;
        else
            img[idx + 3] = 0.0f;
    }
}


extern "C" __declspec(dllexport) void c_preprocess_data(float* graph, float* img_graph, int32_t N, int32_t W, int32_t C, float* out_g_min, float* out_g_max)
{    
    float N_per_pix = max(1.0f, (float)N/(float)W );
    float g_min = INFINITY;
    float g_max = -INFINITY;
    
    for (int32_t n=0; n<min(W,N); ++n)
    {
        int32_t n_start = int32_t(    n*N_per_pix);
        int32_t n_end   = int32_t((n+1)*N_per_pix);
        
        for (int32_t c=0; c<C; ++c)
        {
            float a_v = 0.0f;
            float v_min, v_max;
            
            if ((n_end - n_start) != 0)
            {
                v_min = INFINITY;
                v_max = -INFINITY;
                
                for (int32_t i=n_start; i<n_end; ++i)
                {
                    float v = graph[IDX2D(N,C,i,c)];
                
                    v_min = min(v_min, v);
                    v_max = max(v_max, v);
                    g_min = min(g_min, v);
                    g_max = max(g_max, v);
                
                    a_v += v;
                }
                a_v /= n_end - n_start;
            }
            else
            {
                v_min = 0.0f; v_max = 0.0f;
            }
            img_graph[IDX3D(W,C,3,n,c,0)] = a_v;
            img_graph[IDX3D(W,C,3,n,c,1)] = v_min;
            img_graph[IDX3D(W,C,3,n,c,2)] = v_max;
        }
        
    }
    *out_g_min = g_min;
    *out_g_max = g_max;
}


extern "C" __declspec(dllexport) void c_overlay_graph(float* img, float* graph, int32_t C, int32_t H, int32_t W, float g_min, float g_max, float* colors)
{ 
    float g_d = g_max-g_min;
    if (g_d != 0)
    {
        for (int32_t w=0; w<W; ++w)
        for (int32_t c=0; c<C; ++c)
        {
            float a_v   = graph[IDX3D(W, C, 3, w, c, 0)];
            float v_min = graph[IDX3D(W, C, 3, w, c, 1)];
            float v_max = graph[IDX3D(W, C, 3, w, c, 2)];
            
            float b = colors[IDX2D(C, 3, c, 0)];
            float g = colors[IDX2D(C, 3, c, 1)];
            float r = colors[IDX2D(C, 3, c, 2)];
            
            
            if (v_min != 0 | v_max != 0)
            {
                if (v_min == v_max)
                {
                    if (w > 0)
                    {
                        float prev_a_v = graph[IDX3D(W, C, 3, w-1, c, 0)];
                        if (a_v >= prev_a_v)
                        {
                            v_min = prev_a_v;
                            v_max = a_v;
                        } else
                        {
                            v_min = a_v;
                            v_max = prev_a_v;
                        }
                    }
                }
                
                float a_v_f   = (a_v-g_min)   / g_d;
                float v_min_f = (v_min-g_min) / g_d;
                float v_max_f = (v_max-g_min) / g_d;
                
                int32_t h_a_v = clamp( (int32_t)((H-1)*a_v_f  ), 0, (int32_t)H-1);
                int32_t h_min = clamp( (int32_t)((H-1)*v_min_f), 0, (int32_t)H-1);
                int32_t h_max = clamp( (int32_t)((H-1)*v_max_f), 0, (int32_t)H-1);
                
                for (int32_t h=h_min; h<h_max+1; ++h)
                {
                    float a = 1.0f;
                    
                    if (h >= h_a_v)
                    {
                        float d = (h_a_v-h_max);
                        if (d != 0.0f)
                            a = (h - h_max) / d;
                    } else
                    {
                        float d = (h_a_v-h_min);
                        if (d != 0.0f)
                            a = (h - h_min) / d;
                    }
                    
                    size_t img_idx = IDX3D(H, W, 4, H-1-h, w, 0);
                    img[img_idx+0] = b;
                    img[img_idx+1] = g;
                    img[img_idx+2] = r;
                    img[img_idx+3] = a;
                }
            }
        }
    }
}
