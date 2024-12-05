#pragma once 

typedef float<2> vec2;
typedef float<3> vec3;

vec2 vec2_(float v)
{
    vec2 r = {v,v};
    return r;
}
vec2 vec2_(float x, float y)
{
    vec2 r = {x,y};
    return r;
}

vec3 vec3_(float v)
{
    vec3 r = {v,v,v};
    return r;
}
vec3 vec3_(float x, float y, float z)
{
    vec3 r = {x,y,z};
    return r;
}

vec2 abs( const vec2& v ) { return vec2_( abs(v.x), abs(v.y) ); }
vec3 abs( const vec3& v ) { return vec3_( abs(v.x), abs(v.y), abs(v.z) ); }

vec2 clamp( const vec2& v, const vec2& minv, const vec2& maxv) { return vec2_( clamp(v.x, minv.x, maxv.x), clamp(v.y, minv.y, maxv.y) ); }
vec2 clamp( const vec2& v, float minv, float maxv ) { return vec2_( clamp(v.x, minv, maxv), clamp(v.y, minv, maxv) ); }

vec3 clamp( const vec3& v, const vec3& minv, const vec3& maxv) { return vec3_( clamp(v.x, minv.x, maxv.x), clamp(v.y, minv.y, maxv.y), clamp(v.z, minv.z, maxv.z) ); }
vec3 clamp( const vec3& v, float minv, float maxv ) { return vec3_( clamp(v.x, minv, maxv), clamp(v.y, minv, maxv), clamp(v.z, minv, maxv) ); }


float dot( const vec2& a, const vec2& b ) { return a.x*b.x + a.y*b.y; }
float dot( const vec3& a, const vec3& b ) { return a.x*b.x + a.y*b.y + a.z*b.z; }

float fract(float x) { return x - floor(x); }
vec2 fract(const vec2& v) { return vec2_( fract(v.x), fract(v.y) ); }
vec3 fract(const vec3& v) { return vec3_( fract(v.x), fract(v.y), fract(v.z) ); }

float mod(float x, float y) { return x - trunc(x/y)*y; }

float length(const vec2& v) { return sqrt(v.x*v.x + v.y*v.y); }
float length(const vec3& v) { return sqrt(v.x*v.x + v.y*v.y + v.z*v.z); }

vec2 pow( const vec2& base, const vec2& exponent ) { return vec2_( pow(base.x, exponent.x), pow(base.y, exponent.y) ); }
vec3 pow( const vec3& base, const vec3& exponent ) { return vec3_( pow(base.x, exponent.x), pow(base.y, exponent.y), pow(base.z, exponent.z) ); }

float smoothstep( float edge0, float edge1, float v ) { float t = clamp( (v - edge0)/(edge1 - edge0), 0.f, 1.f ); return t*t*(3.f - 2.f*t); }
float step( float edge, float v ) { return v < edge ? 0.f : 1.f; }

float sign(float x) { return (x < 0.0f ? -1.0f : x > 0.0 ? 1.0f : 0.0f ); }
vec2 sign(const vec2& v) { return vec2_( sign(v.x), sign(v.y)); }
vec3 sign(const vec3& v) { return vec3_( sign(v.x), sign(v.y), sign(v.z)); }

