#include "glsl/glsl.h"
#include "glsl/index.h"

template <typename T, size_t C, size_t BINS>
int32_t c_find_nearest_hist( T* hist_ar, int32_t S, int32_t c_idx, int32_t c_avg_count, int32_t s_idx, int32_t e_idx)
{
    // Computing average reference hist
    int32_t c_idx_start = max(c_idx-c_avg_count, 0);
    int32_t c_idx_end = c_idx;
    int32_t c_total = (c_idx_end-c_idx_start+1);
    
    float hist_ref[C][BINS] = {0};
    for (int32_t idx=c_idx_start; idx<=c_idx_end; ++idx)
    for (int32_t c=0; c<C; ++c)
    for (int32_t bin=0; bin<BINS; ++bin)
        hist_ref[c][bin] += (float)hist_ar[ IDX3D(S,C,BINS,idx,c,bin) ];
    
    for (int32_t c=0; c<C; ++c)
    for (int32_t bin=0; bin<BINS; ++bin)
        hist_ref[c][bin] /= (float)c_total;
        
    float c_avg[C];
    for (int32_t c=0; c<C; ++c)
    {
        c_avg[c] = 0;
        for (int32_t bin=0; bin<BINS; ++bin)
            c_avg[c] += (float)hist_ar[ IDX3D(S,C,BINS,c_idx,c,bin) ];
        c_avg[c] /= (float)BINS;
    }
        
    float min_d = INFINITY;
    int32_t min_idx = -1;
                
    for (int32_t t_idx=s_idx; t_idx<e_idx; ++t_idx)
    {
        float t_avg[C];
        float v[C];
        
        for (int32_t c=0; c<C; ++c)
        {   
            t_avg[c] = 0;
            v[c] = 0;
            for (int32_t bin=0; bin<BINS; ++bin)
            {
                float c_bin = hist_ref[c][bin]; 
                float t_bin = (float)hist_ar[ IDX3D(S,C,BINS,t_idx,c,bin) ];
                
                t_avg[c] += t_bin;
                v[c] += sqrt( c_bin * t_bin ); 
            }
            t_avg[c] /= (float)BINS;
        }
            
        float q[C];
        for (int32_t c=0; c<C; ++c)
            q[c] = 1.0f / ( sqrt(c_avg[c] * t_avg[c]) * (float)BINS );
        
        float d = 0.0f;
        for (int32_t c=0; c<C; ++c)
            d += sqrt(1.0f - q[c]*v[c]);
        
        if (d < min_d)
        {
            min_d = d;            
            min_idx = t_idx;
        }
    }
    
    return min_idx;
}

extern "C" __declspec(dllexport) int32_t c_find_nearest_hist_u8( int32_t* hist_ar, int32_t S, int32_t c_idx, int32_t c_avg_count, int32_t s_idx, int32_t e_idx)
{
    return c_find_nearest_hist<int32_t, 3, 256>(hist_ar, S, c_idx, c_avg_count, s_idx, e_idx);
}

extern "C" __declspec(dllexport) int32_t c_find_nearest_hist_f32( float* hist_ar, int32_t S, int32_t c_idx, int32_t c_avg_count, int32_t s_idx, int32_t e_idx)
{
    return c_find_nearest_hist<float, 3, 256>(hist_ar, S, c_idx, c_avg_count, s_idx, e_idx);
}


/*

__kernel calc(uint64 gid, int32 userparam1)
{
    return gid;
}

export int c_main_func( float* img, int32 W, int32 H, int32 userparam1 )
{
    run_gang(calc, img, W*H, userparam1)
}


extern "C" __declspec(dllexport) int c_find_nearest_hist( int32_t* hist_ar, int S,  int c_idx, int s_idx, int e_idx)
{
    //const int BINS = 256;// int BINS, int C,
    #define BINS (256)
    #define C (3)
    
    float c_avg[3];
    
    for (int c=0; c<C; ++c)
    {
        c_avg[c] = 0;
        for (int bin=0; bin<BINS; ++bin)
            c_avg[c] += (float)hist_ar[ IDX3D(S,C,BINS,c_idx,c,bin) ];
        c_avg[c] /= (float)BINS;
    }
        
    float min_d = INFINITY;
    int min_idx = -1;
                
    for (int t_idx=s_idx; t_idx<e_idx; ++t_idx)
    {
        float d = 0.0f;
        
        for (int c=0; c<C; ++c)
        {
            float t_avg = 0;
            float v = 0;
            
            for (int bin=0; bin<BINS; ++bin)
            {
                int32_t c_bin = hist_ar[ IDX3D(S,C,BINS,c_idx,c,bin) ];
                int32_t t_bin = hist_ar[ IDX3D(S,C,BINS,t_idx,c,bin) ];
                
                t_avg += (float)t_bin;
                
                v += sqrt( (float)c_bin * (float)t_bin );
            }
            t_avg /= (float)BINS;
            
            float q = 1.0f / ( sqrt(c_avg[c] * t_avg) * (float)BINS );
            
            d += sqrt(1.0f - q*v);
        }
        
        if (d < min_d)
        {
            min_d = d;            
            min_idx = t_idx;
        }
    }
    
    return min_idx;
}
*/
