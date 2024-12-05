#ifndef _GLSL_INTERNAL_H_
#define _GLSL_INTERNAL_H_

#include <stddef.h>
#include <stdint.h>

namespace glsl
{
    
// CONSTANTS
#define INFINITY  (1.0/0.0)
#define NAN       (0.0/0.0)

static const float M_E       = 2.7182818284590452353602874713526625f;
static const float M_SQRT2   = 1.4142135623730950488016887242096981f;
static const float M_SQRT1_2 = 0.7071067811865475244008443621048490f;
static const float M_LOG2E   = 1.4426950408889634073599246810018921f;
static const float M_LOG10E  = 0.4342944819032518276511289189166051f;
static const float M_LN2     = 0.6931471805599453094172321214581765f;
static const float M_LN10    = 2.3025850929940456840179914546843642f;

static const float M_PI       = 3.1415926535897932384626433832795029f;
static const float M_PI_2     = 1.5707963267948966192313216916397514f;
static const float M_PI_4     = 0.7853981633974483096156608458198757f;
static const float M_1_PI     = 0.3183098861837906715377675267450287f;
static const float M_2_PI     = 0.6366197723675813430755350534900574f;
static const float M_2_SQRTPI = 1.1283791670955125738961589031215452f;


class vec2;
class vec3;

// FUNCS
float abs(float x);
vec2 abs(const vec2& v);
vec3 abs(const vec3& v);

float atan(float y, float x);
float acos(float x);

//inline float clamp(float v, float minv, float maxv) { return v < minv ? minv : (v > maxv ? maxv : v); }
//inline float clamp(float v, int minv, int maxv) { return v < minv ? minv : (v > maxv ? maxv : v); }
inline float clamp(float v, float minv, float maxv) { return v < minv ? minv : (v > maxv ? maxv : v); }

inline float clamp(float v, int minv, int maxv) 
{ 
    return v < minv ? minv : (v > maxv ? maxv : v); 
}

inline int clamp(int v, int minv, int maxv) { return v < minv ? minv : (v > maxv ? maxv : v); }

vec2 clamp(const vec2& v, float minv, float maxv);
vec2 clamp(const vec2& v, const vec2& minv, const vec2& maxv);
vec3 clamp(const vec3& v, float minv, float maxv);
vec3 clamp(const vec3& v, const vec3& minv, const vec3& maxv);

float cos(float x);
float distance(float p0, float p1);
float distance(const vec2& p0, const vec2& p1);
inline float degrees( float rad) { return rad * 57.295779513082320876798154814105f; } 
float dot(const vec2& a, const vec2& b);
float dot(const vec3& a, const vec3& b);
float exp(float x);
float fract(float x);
float floor(float x);
vec2 floor(const vec2& v);
vec3 floor(const vec3& v);

#define isnan(x)    ((x) != (x))
#define isinf(x)    (((x) == (1.0/0.0)) || ((x) == -(1.0/0.0)))
#define isfinite(x) (!(isinf(x)) && (x != (0.0/0.0) ))

inline float length( float v ) { return v < 0.f ? -v : v; } 
float length( const vec2& v );
float length( const vec3& v );

template <typename T> const T& max( const T& a, const T& b ) { return a < b ? b : a; }
vec2 max( const vec2& a, const vec2& b );
vec3 max( const vec3& a, const vec3& b );

template <typename T> const T& min( const T& a, const T& b ) { return a > b ? b : a; }
vec2 min( const vec2& a, const vec2& b );
vec3 min( const vec3& a, const vec3& b );

inline float mix( float x, float y, float a ) { return x + (y-x)*a; }
vec2 mix( const vec2& x, const vec2& y, float a );
vec3 mix( const vec3& x, const vec3& y, float a );

float mod (float x, float div);
float modf(float x, float *iptr);

vec2 neg( const vec2& a );
vec3 neg( const vec3& a );

vec2 normalize( const vec2& v );
vec3 normalize( const vec3& v );

inline float radians( float deg ) { return deg * 0.017453292519943295769236907684886f; }

float pow( float base, float exponent );
vec2 pow( const vec2& base, const vec2& exponent );

float saturate( float v );
vec2 saturate( const vec2& v );
vec3 saturate( const vec3& v );

float smoothstep( float edge0, float edge1, float v );
float step( float edge, float v );

inline float sign(float x) { return (x < 0.0f ? -1.0f : x > 0.0 ? 1.0f : 0.0f ); }
vec2 sign(const vec2& v);
vec3 sign(const vec3& v);

float sin(float x);
void sincos( float a, float* sina, float* cosa );
float sqrt(float x);
float tan( float x );

float trunc( float x );


// CLASSES
class vec2
{public:
	enum Constants { SIZE = 2, };

	float x; float y;

	vec2() {}
    explicit vec2( float v ) { y = x = v; }
	vec2( float x0, float y0 ) : x(x0), y(y0) {}

    vec2& operator+=( const vec2& o ) { x += o.x; y += o.y; return *this; }
    vec2& operator-=( const vec2& o ) { x -= o.x; y -= o.y; return *this; }
    vec2& operator*=( const vec2& o ) { x *= o.x; y *= o.y; return *this; }
    vec2& operator/=( const vec2& o ) { x /= o.x; y /= o.y; return *this; }
    vec2& operator+=( float s ) { x += s; y += s; return *this; }
    vec2& operator-=( float s ) { x -= s; y -= s; return *this; }
    vec2& operator*=( float s ) { x *= s; y *= s; return *this; }
    vec2& operator/=( float s ) { x /= s; y /= s; return *this; }
    
    vec2 operator*( float s ) const { return vec2(x*s,y*s); }
    vec2 operator/( float s ) const { return vec2(x/s,y/s); }
    vec2 operator*( const vec2& o ) const { return vec2(x*o.x,y*o.y); }
    vec2 operator/( const vec2& o ) const { return vec2(x/o.x,y/o.y); }
    vec2 operator+( const vec2& o ) const { return vec2(x+o.x,y+o.y); }
    vec2 operator-( const vec2& o ) const { return vec2(x-o.x,y-o.y); }
    vec2 operator-() const { return vec2(-x,-y); }

    bool operator==( const vec2& o ) const { return x==o.x & y==o.y; }
    bool operator!=( const vec2& o ) const { return !(*this == o); }

	float&		operator[]( size_t i )			{return (&x)[i];}
	const float& operator[]( size_t i ) const	{return (&x)[i];}
};

vec2 operator*( float s, const vec2& v );
vec2 operator/( float s, const vec2& v );
vec2 operator+( float s, const vec2& v );
vec2 operator-( float s, const vec2& v );
vec2 operator+( const vec2& v, float s);
vec2 operator-( const vec2& v, float s);

class vec3
{public:
	enum Constants { SIZE = 3, };

	float x; float y; float z;

	vec3() {}
    explicit vec3( float v ) { z = y = x = v; }
	vec3( float x0, float y0, float z0 ) : x(x0), y(y0), z(z0) {}

    vec3& operator+=( const vec3& o ) { x += o.x; y += o.y; z += o.z; return *this; }
    vec3& operator-=( const vec3& o ) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    vec3& operator*=( const vec3& o ) { x *= o.x; y *= o.y; z *= o.z; return *this; }
    vec3& operator/=( const vec3& o ) { x /= o.x; y /= o.y; z /= o.z; return *this; }
    vec3& operator+=( float s ) { x += s; y += s; z += s; return *this; }
    vec3& operator-=( float s ) { x -= s; y -= s; z -= s; return *this; }
    vec3& operator*=( float s ) { x *= s; y *= s; z *= s; return *this; }
    vec3& operator/=( float s ) { x /= s; y /= s; z /= s; return *this; }
    
    vec3 operator*( float s ) const { return vec3(x*s,y*s,z*s); }
    vec3 operator/( float s ) const { return vec3(x/s,y/s,z/s); }
    vec3 operator*( const vec3& o ) const { return vec3(x*o.x,y*o.y,z*o.z); }
    vec3 operator/( const vec3& o ) const { return vec3(x/o.x,y/o.y,z/o.z); }
    vec3 operator+( const vec3& o ) const { return vec3(x+o.x,y+o.y,z+o.z); }
    vec3 operator-( const vec3& o ) const { return vec3(x-o.x,y-o.y,z-o.z); }
    vec3 operator-() const { return vec3(-x,-y,-z); }

    bool operator==( const vec3& o ) const { return x==o.x & y==o.y & z==o.z; }
    bool operator!=( const vec3& o ) const { return !(*this == o); }

	float&		operator[]( size_t i )			{ return (&x)[i]; }
	const float& operator[]( size_t i ) const	{ return (&x)[i]; }
};
vec3 operator*( float s, const vec3& v );
vec3 operator/( float s, const vec3& v );
vec3 operator+( float s, const vec3& v );
vec3 operator-( float s, const vec3& v );
vec3 operator+( const vec3& v, float s);
vec3 operator-( const vec3& v, float s);
}

#endif //_GLSL_INTERNAL_H_