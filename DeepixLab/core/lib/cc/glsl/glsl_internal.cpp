#include "glsl_internal.h"

#include "math.h"

using glsl::vec2;
using glsl::vec3;


float glsl::abs (float x) {  return ::fabs(x); }
vec2 glsl::abs( const vec2& v ) { return vec2( glsl::abs(v[0]), glsl::abs(v[1]) ); }
vec3 glsl::abs( const vec3& v ) { return vec3( glsl::abs(v[0]), glsl::abs(v[1]), glsl::abs(v[2]) ); }
float glsl::acos(float x) { return ::acos(x); }

float glsl::atan(float y, float x) { return ::atan2(y,x); }

vec2 glsl::clamp( const vec2& v, float minv, float maxv ) { return vec2( glsl::clamp(v.x, minv, maxv), glsl::clamp(v.y, minv, maxv) ); }
vec2 glsl::clamp( const vec2& v, const vec2& minv, const vec2& maxv ) { return vec2( glsl::clamp(v.x, minv.x, maxv.x), glsl::clamp(v.y, minv.y, maxv.y) ); }

vec3 glsl::clamp( const vec3& v, float minv, float maxv ) { return vec3( glsl::clamp(v.x, minv, maxv), glsl::clamp(v.y, minv, maxv), glsl::clamp(v.z, minv, maxv) ); }
vec3 glsl::clamp( const vec3& v, const vec3& minv, const vec3& maxv ) { return vec3( glsl::clamp(v.x, minv.x, maxv.x), glsl::clamp(v.y, minv.y, maxv.y), glsl::clamp(v.z, minv.z, maxv.z) ); }

float glsl::cos(float x) { return ::cos(x); }

float glsl::distance( float p0, float p1 ) { return length( p1 - p0 ); }
float glsl::distance( const vec2& p0, const vec2& p1 ) { return length( p0 - p1 ); }
float glsl::dot( const vec2& a, const vec2& b ) { return a.x*b.x + a.y*b.y; }
float glsl::dot( const vec3& a, const vec3& b ) { return a.x*b.x + a.y*b.y + a.z*b.z; }

float glsl::exp(float x) { return ::exp(x); }
float glsl::fract(float x) { return x - floor(x); }

float glsl::floor(float x) { return ::floor(x); }
vec2 glsl::floor(const vec2& v) { return vec2 ( glsl::floor(v.x), glsl::floor(v.y) ); }
vec3 glsl::floor(const vec3& v) { return vec3 ( glsl::floor(v.x), glsl::floor(v.y), glsl::floor(v.z) ); }

float glsl::length( const vec2& v ) { return ::sqrt( v.x*v.x + v.y*v.y ); }
float glsl::length( const vec3& v ) { return ::sqrt( v.x*v.x + v.y*v.y + v.z*v.z ); }

vec2 glsl::max( const vec2& a, const vec2& b ) { return vec2( max(a[0],b[0]), max(a[1],b[1]) ); }
vec3 glsl::max( const vec3& a, const vec3& b ) { return vec3( max(a[0],b[0]), max(a[1],b[1]), max(a[2],b[2]) ); }

vec2 glsl::min( const vec2& a, const vec2& b ) { return vec2( min(a[0],b[0]), min(a[1],b[1]) ); }
vec3 glsl::min( const vec3& a, const vec3& b ) { return vec3( min(a[0],b[0]), min(a[1],b[1]), min(a[2],b[2]) ); }

vec2 glsl::mix( const vec2& x, const vec2& y, float a ) { return x + (y-x)*a; }
vec3 glsl::mix( const vec3& x, const vec3& y, float a ) { return x + (y-x)*a; }

float glsl::modf(float x, float *iptr) { return ::modff(x, iptr); }

float glsl::mod (float x, float div) {  return x - glsl::floor(x/div)*div; }

vec2 glsl::neg( const vec2& a ) { return -a; }
vec3 glsl::neg( const vec3& a ) { return -a; }
vec2 glsl::normalize( const vec2& v ) { float len = length(v); return vec2 (v.x/len, v.y/len); }
vec3 glsl::normalize( const vec3& v ) { float len = length(v); return vec3 (v.x/len, v.y/len, v.z/len); }

float glsl::pow( float base, float exponent ) { return ::pow(base, exponent); }
vec2 glsl::pow( const vec2& base, const vec2& exponent ) { return vec2 ( glsl::pow(base.x, exponent.x), glsl::pow(base.y, exponent.y) ); }

float glsl::saturate( float v ) { return glsl::clamp(v, 0.0f, 1.0f); }
vec2 glsl::saturate( const vec2& v ) { return glsl::clamp(v, 0.0f, 1.0f); }
vec3 glsl::saturate( const vec3& v ) { return glsl::clamp(v, 0.0f, 1.0f); }

float glsl::smoothstep( float edge0, float edge1, float v ) { float t = clamp( (v - edge0)/(edge1 - edge0), 0.f, 1.f ); return t*t*(3.f - 2.f*t); }
float glsl::step( float edge, float v ) { return v < edge ? 0.f : 1.f; }

vec2 glsl::sign(const vec2& v) { return vec2( glsl::sign(v.x), glsl::sign(v.y)); }
vec3 glsl::sign(const vec3& v) { return vec3( glsl::sign(v.x), glsl::sign(v.y), glsl::sign(v.z)); }

float glsl::sin(float x) { return ::sin(x); }
float glsl::sqrt(float x) { return ::sqrt(x); }
void glsl::sincos( float a, float* sina, float* cosa ) { *sina = ::sin(a); *cosa = ::cos(a); }
float glsl::tan(float x) { return (::sin(x) / ::cos(x)); }

float glsl::trunc( float x ) { return ::trunc(x); }

vec2 glsl::operator*( float s, const vec2& v ) { return vec2(s*v.x, s*v.y); }
vec2 glsl::operator/( float s, const vec2& v ) { return vec2(s/v.x, s/v.y); }
vec2 glsl::operator+( float s, const vec2& v ) { return vec2(s+v.x, s+v.y); }
vec2 glsl::operator-( float s, const vec2& v ) { return vec2(s-v.x, s-v.y); }
vec2 glsl::operator+( const vec2& v, float s ) { return vec2(v.x+s, v.y+s); }
vec2 glsl::operator-( const vec2& v, float s ) { return vec2(v.x-s, v.y-s); }

vec3 glsl::operator*( float s, const vec3& v ) { return vec3(s*v.x*s, s*v.y, s*v.z); }
vec3 glsl::operator/( float s, const vec3& v ) { return vec3(s/v.x, s/v.y, s/v.z); }
vec3 glsl::operator+( float s, const vec3& v ) { return vec3(s+v.x, s+v.y, s+v.z); }
vec3 glsl::operator-( float s, const vec3& v ) { return vec3(s-v.x, s-v.y, s-v.z); }
vec3 glsl::operator+( const vec3& v, float s ) { return vec3(v.x+s, v.y+s, v.z+s); }
vec3 glsl::operator-( const vec3& v, float s ) { return vec3(v.x-s, v.y-s, v.z-s); }
