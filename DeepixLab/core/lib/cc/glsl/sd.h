#ifndef _SD_SHAPES_H_
#define _SD_SHAPES_H_

#include "glsl.h"


// Signed distance from point to line (not segment)
float sd_point_to_line(const vec2& l0, const vec2& l1, const vec2& pt)
{
    float a = l0.y - l1.y;
    float b = l1.x - l0.x;
    float c = l0.x*l1.y - l1.x*l0.y;
    
    return (a*pt.x+b*pt.y+c) / sqrt(a*a+b*b);
}

float _cro( const vec2& a, const vec2& b ) { return a.x*b.y-a.y*b.x; }

float sd_bezier(const vec2& pos, const vec2& A, const vec2& B, const vec2& C)
{
    vec2 a = B - A;
    vec2 b = A - 2.0*B + C;
    vec2 c = a * 2.0;
    vec2 d = A - pos;

    float kk = 1.0/dot(b,b);
    float kx = kk * dot(a,b);
    float ky = kk * (2.0*dot(a,a)+dot(d,b))/3.0;
    float kz = kk * dot(d,a);      

    float res = 0.0;

    float p  = ky - kx*kx;
    float q  = kx*(2.0*kx*kx - 3.0*ky) + kz;
    float p3 = p*p*p;
    float q2 = q*q;
    float h  = q2 + 4.0*p3;
    
    float sgn;
    if( h>=0.0 ) 
    {   // 1 root
        h = sqrt(h);
        vec2 x = (vec2(h,-h)-q)/2.0;

        vec2 uv = sign(x)*pow(abs(x), vec2(1.0/3.0));
        float t = clamp( uv.x+uv.y-kx, 0.0f, 1.0f );
        vec2  w = d+(c+b*t)*t;
        res = dot(w,w);
        sgn = _cro(c+2.0*b*t,w);
    }
    else 
    {   // 3 roots
        float z = sqrt(-p);
        float v = acos( q/(p*z*2.0) ) / 3.0;
        float m = cos(v);
        float n = sin(v)*1.732050808;
        
        vec3  t = clamp( vec3(m+m,-n-m,n-m)*z-kx, 0.0f, 1.0f );
        vec2  qx = d+(c+b*t.x)*t.x; 
        vec2  qy = d+(c+b*t.y)*t.y; 
        float dx = dot(qx, qx);
        float dy = dot(qy, qy);
        float sx = _cro(a+b*t.x,qx);
        float sy = _cro(a+b*t.y,qy);
        if( dx<dy ) 
        {
            res=dx;
            sgn=sx;
        } else 
        {
            res=dy;
            sgn=sy;
        }
    }
    
    return sqrt( res ) * sign(sgn); 
}

// Signed distance from point to line (not segment)
float sd_loading_circle(const vec2& pix_coord, const vec2& center, float u_time, float R_inner, float R_outter, float edge_smooth)
{
    vec2 d = center - pix_coord;
    
    float r     = length(d);
    float theta = atan(d.y,d.x);
            
    // fix edge_smooth 
    edge_smooth = min(edge_smooth, (R_outter-R_inner)/2.0f);
    
    return fract( 0.5f*(1.0f+theta/M_PI) -u_time + 0.25f ) 
            * (float)(r >= R_inner) * (float)(r <= R_outter)
            * smoothstep(0.0f, edge_smooth, abs(r - R_inner) ) 
            * smoothstep(0.0f, edge_smooth, abs(r - R_outter) ) ;
}

#endif //_SD_SHAPES_H_