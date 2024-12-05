#include "glsl.h"

inline vec2 catmull_spline(vec2 p0, vec2 p1, vec2 p2, vec2 p3, float t) 
{
    float alpha = 1.0;
    float tension = 0.0;
    
    float t01 = pow(distance(p0, p1), alpha);
	float t12 = pow(distance(p1, p2), alpha);
	float t23 = pow(distance(p2, p3), alpha);

	vec2 m1 = (1.0f - tension) * (p2 - p1 + t12 * ((p1 - p0) / t01 - (p2 - p0) / (t01 + t12)));
	vec2 m2 = (1.0f - tension) * (p2 - p1 + t12 * ((p3 - p2) / t23 - (p3 - p1) / (t12 + t23)));
	vec2 a = 2.0f * (p1 - p2) + m1 + m2;
	vec2 b = -3.0f * (p1 - p2) - m1 - m1 - m2;
	vec2 c = m1;
	vec2 d = p1;

	return a * t * t * t + b * t * t + c * t + d;
}



float sd_catmull_spline(vec2 p, vec2 p0, vec2 p1, vec2 p2, vec2 p3, int32 segment_count, float& spline_w)
{
    vec2 a = p1;
    
    float dist = FLT_MAX;
    float t_frac = 1 / (float) segment_count;
    
    for (int32 i=1; i< segment_count+1;++i)
    {
        float t = i * t_frac;
        
        vec2 b = catmull_spline(p0, p1, p2, p3, t);
        
        float segment_w;
        float segment_d = sd_segment(p, a, b, segment_w);
        
        if (segment_d < dist)
        {
            dist = segment_d;
            
            spline_w = mix(t-t_frac, t, segment_w);
        }
        
        a = b;
    }
    return dist;
}

float sd_catmull_spline(vec2 p, vec2 p0, vec2 p1, vec2 p2, vec2 p3, int32 segment_count)
{
    float spline_w;
    return sd_catmull_spline(p, p0, p1, p2, p3, segment_count, spline_w);
}

float sd_catmull_spline(vec2 p, uniform vec2 pts[], int32 pts_count, int32 segment_count, float& spline_w)
{
    float dist = FLT_MAX;
    
    int32 spline_count = pts_count-3;
    float i_frac = 1 / (float) spline_count;
    
    for (int32 i=0; i < spline_count; ++i)
    {
        float s_w;
        float s_d = sd_catmull_spline(p, pts[i], pts[i+1], pts[i+2], pts[i+3], segment_count, s_w);
        
        if (s_d < dist)
        {
            dist = s_d;
            spline_w = mix(i*i_frac, (i+1)*i_frac, s_w);        
        }
    }
    return dist;
}

float sd_catmull_spline(vec2 p, uniform vec2 pts[], int32 pts_count, int32 segment_count)
{
    float spline_w;
    return sd_catmull_spline(p, pts, pts_count, segment_count, spline_w);
}