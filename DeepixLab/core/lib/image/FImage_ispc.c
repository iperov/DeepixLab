#include "glsl.h"
#include "index.h"
#include "color.h"

#define U8_TO_F32(x) (x / 255.0f)
#define F32_TO_U8(x) ((uint8)clamp(x*255.0f, 0.0f, 255.0f) )
#define F32_CLAMP(x) clamp(x, 0.0f, 1.0f)


export void u8_to_f32(uniform float O[], const uniform uint8 I[], uniform uint32 size) { foreach(i = 0 ... size) { O[i] = U8_TO_F32(I[i]); } }
export void f32_to_u8(uniform uint8 O[], const uniform float I[], uniform uint32 size) { foreach(i = 0 ... size) { O[i] = F32_TO_U8(I[i]); } }

export void mul_u8_u8(uniform uint8 O[], const uniform uint8 A[], const uniform uint8 B[], uniform uint32 size)
{ foreach(i = 0 ... size) { O[i] = F32_TO_U8( U8_TO_F32(A[i]) * U8_TO_F32(B[i]) ); } }
export void mul_u8_f32(uniform uint8 O[], const uniform uint8 A[], const uniform float B[], uniform uint32 size)
{ foreach(i = 0 ... size) { O[i] = F32_TO_U8( U8_TO_F32(A[i]) * B[i] ); } }
export void mul_f32_u8(uniform float O[], const uniform float A[], const uniform uint8 B[], uniform uint32 size)
{ foreach(i = 0 ... size) { O[i] = F32_CLAMP( A[i] * U8_TO_F32(B[i]) ); } }
export void mul_f32_f32(uniform float O[], const uniform float A[], const uniform float B[], uniform uint32 size)
{ foreach(i = 0 ... size) { O[i] = F32_CLAMP( A[i] * B[i] ); } }

export void sub_u8_u8(uniform uint8 O[], const uniform uint8 A[], const uniform uint8 B[], uniform uint32 size)
{ foreach(i = 0 ... size) { O[i] = F32_TO_U8( U8_TO_F32(A[i]) - U8_TO_F32(B[i]) ); } }
export void sub_u8_f32(uniform uint8 O[], const uniform uint8 A[], const uniform float B[], uniform uint32 size)
{ foreach(i = 0 ... size) { O[i] = F32_TO_U8( U8_TO_F32(A[i]) - B[i] ); } }
export void sub_f32_u8(uniform float O[], const uniform float A[], const uniform uint8 B[], uniform uint32 size)
{ foreach(i = 0 ... size) { O[i] = F32_CLAMP( A[i] - U8_TO_F32(B[i]) ); } }
export void sub_f32_f32(uniform float O[], const uniform float A[], const uniform float B[], uniform uint32 size)
{ foreach(i = 0 ... size) { O[i] = F32_CLAMP( A[i] - B[i] ); } }

export void add_u8_u8(uniform uint8 O[], const uniform uint8 A[], const uniform uint8 B[], uniform uint32 size)
{ foreach(i = 0 ... size) { O[i] = F32_TO_U8( U8_TO_F32(A[i]) + U8_TO_F32(B[i]) ); } }
export void add_u8_f32(uniform uint8 O[], const uniform uint8 A[], const uniform float B[], uniform uint32 size)
{ foreach(i = 0 ... size) { O[i] = F32_TO_U8( U8_TO_F32(A[i]) + B[i] ); } }
export void add_f32_u8(uniform float O[], const uniform float A[], const uniform uint8 B[], uniform uint32 size)
{ foreach(i = 0 ... size) { O[i] = F32_CLAMP( A[i] + U8_TO_F32(B[i]) ); } }
export void add_f32_f32(uniform float O[], const uniform float A[], const uniform float B[], uniform uint32 size)
{ foreach(i = 0 ... size) { O[i] = F32_CLAMP( A[i] + B[i] ); } }


export void channel_exposure_u8(uniform uint8 O[], uniform uint8 I[], uniform uint32 H, uniform uint32 W, uniform uint32 C, 
                                uniform float exposure[])
{    
    uniform uint32 HW = H*W;
    foreach(yx = 0 ... HW)
    for (uniform uint32 c=0; c<C; ++c)
    {
        float v = U8_TO_F32(I[IDX2D(HW,C,yx,c)]);
        O[IDX2D(HW,C,yx,c)] = F32_TO_U8( v * pow(2,exposure[c]) );
    }
}

export void channel_exposure_f32(uniform float O[], uniform float I[], uniform uint32 H, uniform uint32 W, uniform uint32 C, 
                                 uniform float exposure[])
{    
    uniform uint32 HW = H*W;
    foreach(yx = 0 ... HW)
    for (uniform uint32 c=0; c<C; ++c)
    {
        float v = I[IDX2D(HW,C,yx,c)];
        O[IDX2D(HW,C,yx,c)] = F32_CLAMP( v * pow(2,exposure[c]) );
    }
}

export void levels_u8(uniform uint8 O[], const uniform uint8 I[], uniform uint32 H, uniform uint32 W, uniform uint32 C, 
                        const uniform float in_b[], const uniform float in_w[], const uniform float in_g[], const uniform float out_b[], const uniform float out_w[])
{    
    uniform uint32 HW = H*W;
    foreach(yx = 0 ... HW)
    {
        for (uniform uint32 c=0; c<C; ++c)
        {
            float v = U8_TO_F32(I[IDX2D(HW,C,yx,c)]);
            
            v = clamp( (v-in_b[c]) / (in_w[c]-in_b[c]), 0.0f, 1.0f);
            v = pow(v, 1.0f/in_g[c]) * (out_w[c] - out_b[c]) + out_b[c];
            
            O[IDX2D(HW,C,yx,c)] = F32_TO_U8(v);
        }
    }
}

export void levels_f32(uniform float O[], const uniform float I[], uniform uint32 H, uniform uint32 W, uniform uint32 C, 
                        const uniform float in_b[], const uniform float in_w[], const uniform float in_g[], const uniform float out_b[], const uniform float out_w[])
{    
    uniform uint32 HW = H*W;
    foreach(yx = 0 ... HW)
    {
        for (uniform uint32 c=0; c<C; ++c)
        {
            float v = I[IDX2D(HW,C,yx,c)];
            
            v = clamp( (v-in_b[c]) / (in_w[c]-in_b[c]), 0.0f, 1.0f);
            v = pow(v, 1.0f/in_g[c]) * (out_w[c] - out_b[c]) + out_b[c];
            
            O[IDX2D(HW,C,yx,c)] = F32_CLAMP(v);
        }
    }
}


export void hsv_shift_u8(uniform uint8 O[], uniform uint8 I[], uniform uint32 H, uniform uint32 W, uniform uint32 C, 
                            uniform float h_offset, uniform float s_offset, uniform float v_offset)
{
    uniform uint32 HW = H*W;
    
    foreach(yx = 0 ... HW)
    {
        float b = U8_TO_F32(I[IDX2D(HW,C,yx,0)]);
        float g = U8_TO_F32(I[IDX2D(HW,C,yx,1)]);
        float r = U8_TO_F32(I[IDX2D(HW,C,yx,2)]);
        float h, s, v;
        bgr_to_hsv(b,g,r,h,s,v);
        
        h = mod(h + h_offset, 1.0f);
        s = clamp(s + s_offset, 0.0f, 1.0f);
        v = clamp(v + v_offset, 0.0f, 1.0f);
        
        hsv_to_bgr(h,s,v,b,g,r);
        
        O[IDX2D(HW,C,yx,0)] = F32_TO_U8(b);
        O[IDX2D(HW,C,yx,1)] = F32_TO_U8(g);
        O[IDX2D(HW,C,yx,2)] = F32_TO_U8(r);
    }
}

export void hsv_shift_f32(uniform float O[], uniform float I[], uniform uint32 H, uniform uint32 W, uniform uint32 C, 
                          uniform float h_offset, uniform float s_offset, uniform float v_offset)
{
    uniform uint32 HW = H*W;
    uniform uint32 idx = 0;
    
    foreach(yx = 0 ... HW)
    {
        float b,g,r,h,s,v;
        
        aos_to_soa3(&I[idx], &b, &g, &r);
        
        bgr_to_hsv(b,g,r,h,s,v);
        
        h = mod(h + h_offset, 1.0f);
        s = clamp(s + s_offset, 0.0f, 1.0f);
        v = clamp(v + v_offset, 0.0f, 1.0f);
        
        hsv_to_bgr(h,s,v,b,g,r);
        
        soa_to_aos3(F32_CLAMP(b), F32_CLAMP(g), F32_CLAMP(r), &O[idx]);
        
        idx += programCount *3;
    }
}

export void blend_u8 (  uniform uint8 O[], uniform uint32 OH, uniform uint32 OW, uniform uint32 OC, 
                        uniform uint8 A[], 
                        uniform float B[], uniform uint32 BH, uniform uint32 BW, uniform uint32 BC, 
                        uniform float M[], uniform uint32 MH, uniform uint32 MW, uniform uint32 MC, 
                        uniform float alpha )
{
    uniform uint32 OHOW = OH*OW;
    uniform uint32 BHBW = BH*BW;
    uniform uint32 MHMW = MH*MW;
    foreach(yx = 0 ... OHOW)
    for (uniform uint32 c=0; c<OC; ++c)
    {
        float a = U8_TO_F32(A[IDX2D(OHOW,OC,yx,c)]);
        float b = B[IDX2D(BHBW,BC,yx,c % BC)];
        float m = M[IDX2D(MHMW,MC,yx,c % MC)];
        float v = a*(1.0f-(m*alpha) ) + b*(m*alpha);
        
        O[IDX2D(OHOW,OC,yx,c)] = F32_TO_U8(v);
    }
}

export void blend_f32 ( uniform float O[], uniform uint32 OH, uniform uint32 OW, uniform uint32 OC, 
                        uniform float A[], 
                        uniform float B[], uniform uint32 BH, uniform uint32 BW, uniform uint32 BC, 
                        uniform float M[], uniform uint32 MH, uniform uint32 MW, uniform uint32 MC, 
                        uniform float alpha )
{
    uniform uint32 OHOW = OH*OW;
    uniform uint32 BHBW = BH*BW;
    uniform uint32 MHMW = MH*MW;
    foreach(yx = 0 ... OHOW)
    for (uniform uint32 c=0; c<OC; ++c)
    {
        float a = A[IDX2D(OHOW,OC,yx,c)];
        float b = B[IDX2D(BHBW,BC,yx,c % BC)];
        float m = M[IDX2D(MHMW,MC,yx,c % MC)];
        float v = a*(1.0f-(m*alpha) ) + b*(m*alpha);
        
        O[IDX2D(OHOW,OC,yx,c)] = F32_CLAMP(v);
    }
}



export void satushift_u8(uniform uint8 O[], uniform uint8 I[], uniform uint32 size)
{ 
    uniform float I_min = FLT_MAX;
    uniform float I_max = FLT_MIN;
    
    foreach(i = 0 ... size) 
    { 
        I_min = min(I_min, reduce_min( U8_TO_F32(I[i]) ));
        I_max = max(I_max, reduce_max( U8_TO_F32(I[i]) ));
    }
    uniform float mod = I_max-I_min;
    if (mod == 0)
        mod = 1.0;
    foreach(i = 0 ... size) 
    { 
        O[i] = F32_TO_U8( (U8_TO_F32(I[i])-I_min)/mod );
    }
}
export void satushift_f32(uniform float O[], uniform float I[], uniform uint32 size)
{ 
    uniform float I_min = FLT_MAX;
    uniform float I_max = FLT_MIN;
    
    foreach(i = 0 ... size) 
    { 
        I_min = min(I_min, reduce_min(I[i]));
        I_max = max(I_max, reduce_max(I[i]));
    }
    uniform float mod = I_max-I_min;
    if (mod == 0)
        mod = 1.0;
    foreach(i = 0 ... size) 
    { 
        O[i] = F32_CLAMP( (I[i]-I_min)/mod );
    }
}

