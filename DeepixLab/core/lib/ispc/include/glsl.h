#pragma once
// Types
typedef float<2>    vec2;
typedef float<3>    vec3;
typedef float<4>    vec4;
typedef int<2>      ivec2;
typedef int<3>      ivec3;
typedef int<4>      ivec4;
typedef bool<2>     bvec2;
typedef bool<3>     bvec3;
typedef bool<4>     bvec4;

struct mat2 { float m[2][2]; };
struct mat3 { float m[3][3]; };
struct mat4 { float m[4][4]; };


// Constructors

inline vec2 Float2(float x)                                     { vec2 r = { x, x }; return r; }
inline vec2 Float2(float x, float y)                            { vec2 r = { x, y }; return r; }
inline vec2 Float2(vec2 x)                                      { return x; }
inline vec3 Float3(float x)                                     { vec3 r = { x, x, x }; return r; }
inline vec3 Float3(vec2 x, float y)                             { vec3 r = { x.x, x.y, y }; return r; }
inline vec3 Float3(float x, vec2 y)                             { vec3 r = { x, y.x, y.y }; return r; }
inline vec3 Float3(float x, float y, float z)                   { vec3 r = { x, y, z }; return r; }
inline vec3 Float3(vec3 x)                                      { return x; }
inline vec4 Float4(float x)                                     { vec4 r = { x, x, x }; return r; }
inline vec4 Float4(vec2 x, vec2 y)                              { vec4 r = { x.x, x.y, y.x, y.y }; return r; }
inline vec4 Float4(vec2 x, float y, float z)                    { vec4 r = { x.x, x.y, y, z }; return r; }
inline vec4 Float4(vec3 x, float y)                             { vec4 r = { x.x, x.y, x.z, y }; return r; }
inline vec4 Float4(float x, vec3 y)                             { vec4 r = { x, y.x, y.y, y.z }; return r; }
inline vec4 Float4(float x, vec2 y, float z)                    { vec4 r = { x, y.x, y.y, z }; return r; }
inline vec4 Float4(float x, float y, vec2 z)                    { vec4 r = { x, y, z.x, z.y }; return r; }
inline vec4 Float4(float x, float y, float z, float w)          { vec4 r = { x, y, z, w }; return r; }
inline vec4 Float4(vec4 x)                                      { return x; }

inline uniform vec2 Float2(uniform float x)                                     { uniform vec2 r = { x, x }; return r; }
inline uniform vec2 Float2(uniform float x, uniform float y)                    { uniform vec2 r = { x, y }; return r; }
inline uniform vec2 Float2(uniform vec2 x)                                      { return x; }
inline uniform vec3 Float3(uniform float x)                                     { uniform vec3 r = { x, x, x }; return r; }
inline uniform vec3 Float3(uniform vec2 x, uniform float y)                     { uniform vec3 r = { x.x, x.y, y }; return r; }
inline uniform vec3 Float3(uniform float x, uniform vec2 y)                     { uniform vec3 r = { x, y.x, y.y }; return r; }
inline uniform vec3 Float3(uniform float x, uniform float y, uniform float z)   { uniform vec3 r = { x, y, z }; return r; }
inline uniform vec3 Float3(uniform vec3 x)                                      { return x; }
inline uniform vec4 Float4(uniform float x)                                     { uniform vec4 r = { x, x, x }; return r; }
inline uniform vec4 Float4(uniform vec2 x, uniform vec2 y)                      { uniform vec4 r = { x.x, x.y, y.x, y.y }; return r; }
inline uniform vec4 Float4(uniform vec2 x, uniform float y, uniform float z)    { uniform vec4 r = { x.x, x.y, y, z }; return r; }
inline uniform vec4 Float4(uniform vec3 x, uniform float y)                     { uniform vec4 r = { x.x, x.y, x.z, y }; return r; }
inline uniform vec4 Float4(uniform float x, uniform vec3 y)                     { uniform vec4 r = { x, y.x, y.y, y.z }; return r; }
inline uniform vec4 Float4(uniform float x, uniform vec2 y, uniform float z)    { uniform vec4 r = { x, y.x, y.y, z }; return r; }
inline uniform vec4 Float4(uniform float x, uniform float y, uniform vec2 z)    { uniform vec4 r = { x, y, z.x, z.y }; return r; }
inline uniform vec4 Float4(uniform float x, uniform float y, uniform float z, uniform float w) { uniform vec4 r = { x, y, z, w }; return r; }
inline uniform vec4 Float4(uniform vec4 x)                                      { return x; }

#define vec2(...)           Float2(__VA_ARGS__)
#define vec3(...)           Float3(__VA_ARGS__)
#define vec4(...)           Float4(__VA_ARGS__)

inline ivec2 Int2(int x)                                    { ivec2 r = { x, x }; return r; }
inline ivec2 Int2(int x, int y)                             { ivec2 r = { x, y }; return r; }
inline ivec3 Int3(int x)                                    { ivec3 r = { x, x, x }; return r; }
inline ivec3 Int3(ivec2 x, int y)                           { ivec3 r = { x.x, x.y, y }; return r; }
inline ivec3 Int3(int x, ivec2 y)                           { ivec3 r = { x, y.x, y.y }; return r; }
inline ivec3 Int3(int x, int y, int z)                      { ivec3 r = { x, y, z }; return r; }
inline ivec4 Int4(int x)                                    { ivec4 r = { x, x, x }; return r; }
inline ivec4 Int4(ivec2 x, ivec2 y)                         { ivec4 r = { x.x, x.y, y.x, y.y }; return r; }
inline ivec4 Int4(ivec2 x, int y, int z)                    { ivec4 r = { x.x, x.y, y, z }; return r; }
inline ivec4 Int4(ivec3 x, int y)                           { ivec4 r = { x.x, x.y, x.z, y }; return r; }
inline ivec4 Int4(int x, ivec3 y)                           { ivec4 r = { x, y.x, y.y, y.z }; return r; }
inline ivec4 Int4(int x, ivec2 y, int z)                    { ivec4 r = { x, y.x, y.y, z }; return r; }
inline ivec4 Int4(int x, int y, ivec2 z)                    { ivec4 r = { x, y, z.x, z.y }; return r; }
inline ivec4 Int4(int x, int y, int z, int w)               { ivec4 r = { x, y, z, w }; return r; }

#define ivec2(...)          Int2(__VA_ARGS__)
#define ivec3(...)          Int3(__VA_ARGS__)
#define ivec4(...)          Int4(__VA_ARGS__)

inline bvec2 Bool2(bool x)                                  { bvec2 r = { x, x }; return r; }
inline bvec2 Bool2(bool x, bool y)                          { bvec2 r = { x, y }; return r; }
inline bvec3 Bool3(bool x)                                  { bvec3 r = { x, x, x }; return r; }
inline bvec3 Bool3(bvec2 x, bool y)                         { bvec3 r = { x.x, x.y, y }; return r; }
inline bvec3 Bool3(bool x, bvec2 y)                         { bvec3 r = { x, y.x, y.y }; return r; }
inline bvec3 Bool3(bool x, bool y, bool z)                  { bvec3 r = { x, y, z }; return r; }
inline bvec4 Bool4(bool x)                                  { bvec4 r = { x, x, x }; return r; }
inline bvec4 Bool4(bvec2 x, bvec2 y)                        { bvec4 r = { x.x, x.y, y.x, y.y }; return r; }
inline bvec4 Bool4(bvec2 x, bool y, bool z)                 { bvec4 r = { x.x, x.y, y, z }; return r; }
inline bvec4 Bool4(bvec3 x, bool y)                         { bvec4 r = { x.x, x.y, x.z, y }; return r; }
inline bvec4 Bool4(bool x, bvec3 y)                         { bvec4 r = { x, y.x, y.y, y.z }; return r; }
inline bvec4 Bool4(bool x, bvec2 y, bool z)                 { bvec4 r = { x, y.x, y.y, z }; return r; }
inline bvec4 Bool4(bool x, bool y, bvec2 z)                 { bvec4 r = { x, y, z.x, z.y }; return r; }
inline bvec4 Bool4(bool x, bool y, bool z, bool w)          { bvec4 r = { x, y, z, w }; return r; }

#define bvec2(...)          Bool2(__VA_ARGS__)
#define bvec3(...)          Bool3(__VA_ARGS__)
#define bvec4(...)          Bool4(__VA_ARGS__)

inline mat2 Matrix2(vec2 col0, vec2 col1) {
    mat2 m;
    m.m[0][0] = col0.x; m.m[0][1] = col0.y;
    m.m[1][0] = col1.x; m.m[1][1] = col1.y;
    return m;
}
inline mat2 Matrix2(float x0, float y0, float x1, float y1) {
    mat2 m;
    m.m[0][0] = x0; m.m[0][1] = y0;
    m.m[1][0] = x1; m.m[1][1] = y1;
    return m;
}
inline mat3 Matrix3(vec3 col0, vec3 col1, vec3 col2) {
    mat3 m;
    m.m[0][0] = col0.x; m.m[0][1] = col0.y; m.m[0][2] = col0.z;
    m.m[1][0] = col1.x; m.m[1][1] = col1.y; m.m[1][2] = col1.z;
    m.m[2][0] = col2.x; m.m[2][1] = col2.y; m.m[2][2] = col2.z;
    return m;
}
inline mat3 Matrix3(float x0, float y0, float z0, float x1, float y1, float z1, float x2, float y2, float z2) {
    mat3 m;
    m.m[0][0] = x0; m.m[0][1] = y0; m.m[0][2] = z0;
    m.m[1][0] = x1; m.m[1][1] = y1; m.m[1][2] = z1;
    m.m[2][0] = x2; m.m[2][1] = y2; m.m[2][2] = z2;
    return m;
}
inline mat4 Matrix4(vec4 col0, vec4 col1, vec4 col2, vec4 col3) {
    mat4 m;
    m.m[0][0] = col0.x; m.m[0][1] = col0.y; m.m[0][2] = col0.z; m.m[0][3] = col0.w;
    m.m[1][0] = col1.x; m.m[1][1] = col1.y; m.m[1][2] = col1.z; m.m[1][3] = col1.w;
    m.m[2][0] = col2.x; m.m[2][1] = col2.y; m.m[2][2] = col2.z; m.m[2][3] = col2.w;
    m.m[3][0] = col3.x; m.m[3][1] = col3.y; m.m[3][2] = col3.z; m.m[3][3] = col3.w;
    return m;
}
inline mat4 Matrix4(float x0, float y0, float z0, float w0, float x1, float y1, float z1, float w1,
                    float x2, float y2, float z2, float w2, float x3, float y3, float z3, float w3) {
    mat4 m;
    m.m[0][0] = x0; m.m[0][1] = y0; m.m[0][2] = z0; m.m[0][3] = w0;
    m.m[1][0] = x1; m.m[1][1] = y1; m.m[1][2] = z1; m.m[1][3] = w1;
    m.m[2][0] = x2; m.m[2][1] = y2; m.m[2][2] = z2; m.m[2][3] = w2;
    m.m[3][0] = x3; m.m[3][1] = y3; m.m[3][2] = z3; m.m[3][3] = w3;
    return m;
}

#define mat2(...)           Matrix2(__VA_ARGS__)
#define mat3(...)           Matrix3(__VA_ARGS__)
#define mat4(...)           Matrix4(__VA_ARGS__)


// Conversions

#define float(...)          (float)(__VA_ARGS__)
#define int(...)            (int)(__VA_ARGS__)


// Operators

// Built-In Functions

// Angle and Trigonometry Functons

#define radians(degrees) (degrees * PI / 180.0f)
#define degrees(radians) (radians * 180.0f / PI)

inline vec2 sin(vec2 f) { return vec2( sin(f.x), sin(f.y) ); }
inline vec3 sin(vec3 f) { return vec3( sin(f.x), sin(f.y), sin(f.z) ); }
inline vec4 sin(vec4 f) { return vec4( sin(f.x), sin(f.y), sin(f.z), sin(f.w) ); }

inline vec2 cos(vec2 f) { return vec2( cos(f.x), cos(f.y) ); }
inline vec3 cos(vec3 f) { return vec3( cos(f.x), cos(f.y), cos(f.z) ); }
inline vec4 cos(vec4 f) { return vec4( cos(f.x), cos(f.y), cos(f.z), cos(f.w) ); }

inline vec2 tan(vec2 f) { return vec2( tan(f.x), tan(f.y) ); }
inline vec3 tan(vec3 f) { return vec3( tan(f.x), tan(f.y), tan(f.z) ); }
inline vec4 tan(vec4 f) { return vec4( tan(f.x), tan(f.y), tan(f.z), tan(f.w) ); }

inline vec2 asin(vec2 f) { return vec2( asin(f.x), asin(f.y) ); }
inline vec3 asin(vec3 f) { return vec3( asin(f.x), asin(f.y), asin(f.z) ); }
inline vec4 asin(vec4 f) { return vec4( asin(f.x), asin(f.y), asin(f.z), asin(f.w) ); }

inline vec2 acos(vec2 f) { return vec2( acos(f.x), acos(f.y) ); }
inline vec3 acos(vec3 f) { return vec3( acos(f.x), acos(f.y), acos(f.z) ); }
inline vec4 acos(vec4 f) { return vec4( acos(f.x), acos(f.y), acos(f.z), acos(f.w) ); }

inline vec2 atan(vec2 f) { return vec2( atan(f.x), atan(f.y) ); }
inline vec3 atan(vec3 f) { return vec3( atan(f.x), atan(f.y), atan(f.z) ); }
inline vec4 atan(vec4 f) { return vec4( atan(f.x), atan(f.y), atan(f.z), atan(f.w) ); }

inline float atan(float y, float x) { return atan2(y, x); }
inline vec2 atan(vec2 y, vec2 x) { return vec2( atan2(y.x, x.x), atan2(y.y, x.y) ); }
inline vec3 atan(vec3 y, vec3 x) { return vec3( atan2(y.x, x.x), atan2(y.y, x.y), atan2(y.z, x.z) ); }
inline vec4 atan(vec4 y, vec4 x) { return vec4( atan2(y.x, x.x), atan2(y.y, x.y), atan2(y.z, x.z), atan2(y.w, x.w) ); }

inline float sinh(float f) { return (exp(f) - exp(-f)) / 2.0f; }
inline vec2 sinh(vec2 f) { return vec2( sinh(f.x), sinh(f.y) ); }
inline vec3 sinh(vec3 f) { return vec3( sinh(f.x), sinh(f.y), sinh(f.z) ); }
inline vec4 sinh(vec4 f) { return vec4( sinh(f.x), sinh(f.y), sinh(f.z), sinh(f.w) ); }

inline float cosh(float f) { return (exp(f) + exp(-f)) / 2.0f; }
inline vec2 cosh(vec2 f) { return vec2( cosh(f.x), cosh(f.y) ); }
inline vec3 cosh(vec3 f) { return vec3( cosh(f.x), cosh(f.y), cosh(f.z) ); }
inline vec4 cosh(vec4 f) { return vec4( cosh(f.x), cosh(f.y), cosh(f.z), cosh(f.w) ); }

inline float tanh(float f) { return sinh(f) / cosh(f); }
inline vec2 tanh(vec2 f) { return vec2( tanh(f.x), tanh(f.y) ); }
inline vec3 tanh(vec3 f) { return vec3( tanh(f.x), tanh(f.y), tanh(f.z) ); }
inline vec4 tanh(vec4 f) { return vec4( tanh(f.x), tanh(f.y), tanh(f.z), tanh(f.w) ); }

inline float asinh(float f) { return log(f + sqrt(f * f + 1)); }
inline vec2 asinh(vec2 f) { return vec2( asinh(f.x), asinh(f.y) ); }
inline vec3 asinh(vec3 f) { return vec3( asinh(f.x), asinh(f.y), asinh(f.z) ); }
inline vec4 asinh(vec4 f) { return vec4( asinh(f.x), asinh(f.y), asinh(f.z), asinh(f.w) ); }

inline float acosh(float f) { return log(f + sqrt(f * f - 1)); }
inline vec2 acosh(vec2 f) { return vec2( acosh(f.x), acosh(f.y) ); }
inline vec3 acosh(vec3 f) { return vec3( acosh(f.x), acosh(f.y), acosh(f.z) ); }
inline vec4 acosh(vec4 f) { return vec4( acosh(f.x), acosh(f.y), acosh(f.z), acosh(f.w) ); }

inline float atanh(float f) { return log((1 + f) / (1 - f)) / 2.0f; }
inline vec2 atanh(vec2 f) { return vec2( atanh(f.x), atanh(f.y) ); }
inline vec3 atanh(vec3 f) { return vec3( atanh(f.x), atanh(f.y), atanh(f.z) ); }
inline vec4 atanh(vec4 f) { return vec4( atanh(f.x), atanh(f.y), atanh(f.z), atanh(f.w) ); }


// Exponental Functons

inline vec2 pow(vec2 f, float m) { return vec2( pow(f.x, m), pow(f.y, m) ); }
inline vec2 pow(vec2 f, vec2 m) { return vec2( pow(f.x, m.x), pow(f.y, m.y) ); }
inline vec3 pow(vec3 f, float m) { return vec3( pow(f.x, m), pow(f.y, m), pow(f.z, m) ); }
inline vec3 pow(vec3 f, vec3 m) { return vec3( pow(f.x, m.x), pow(f.y, m.y), pow(f.z, m.z) ); }
inline vec4 pow(vec4 f, float m) { return vec4( pow(f.x, m), pow(f.y, m), pow(f.z, m), pow(f.w, m) ); }
inline vec4 pow(vec4 f, vec4 m) { return vec4( pow(f.x, m.x), pow(f.y, m.y), pow(f.z, m.z), pow(f.w, m.w) ); }

inline vec2 exp(vec2 f) { return vec2( exp(f.x), exp(f.y) ); }
inline vec3 exp(vec3 f) { return vec3( exp(f.x), exp(f.y), exp(f.z) ); }
inline vec4 exp(vec4 f) { return vec4( exp(f.x), exp(f.y), exp(f.z), exp(f.w) ); }

inline vec2 log(vec2 f) { return vec2( log(f.x), log(f.y) ); }
inline vec3 log(vec3 f) { return vec3( log(f.x), log(f.y), log(f.z) ); }
inline vec4 log(vec4 f) { return vec4( log(f.x), log(f.y), log(f.z), log(f.w) ); }

inline vec2 exp2(vec2 f) { return vec2( pow(2.0f, f.x), pow(2.0f, f.y) ); }
inline vec3 exp2(vec3 f) { return vec3( pow(2.0f, f.x), pow(2.0f, f.y), pow(2.0f, f.z) ); }
inline vec4 exp2(vec4 f) { return vec4( pow(2.0f, f.x), pow(2.0f, f.y), pow(2.0f, f.z), pow(2.0f, f.w) ); }

inline float log2(float f) { return log(f) / log(2.0f); }
inline vec2 log2(vec2 f) { return vec2( log2(f.x), log2(f.y) ); }
inline vec3 log2(vec3 f) { return vec3( log2(f.x), log2(f.y), log2(f.z) ); }
inline vec4 log2(vec4 f) { return vec4( log2(f.x), log2(f.y), log2(f.z), log2(f.w) ); }

inline vec2 sqrt(vec2 f) { return vec2( sqrt(f.x), sqrt(f.y) ); }
inline vec3 sqrt(vec3 f) { return vec3( sqrt(f.x), sqrt(f.y), sqrt(f.z) ); }
inline vec4 sqrt(vec4 f) { return vec4( sqrt(f.x), sqrt(f.y), sqrt(f.z), sqrt(f.w) ); }

inline float inversesqrt(float f) { return rsqrt_fast(f); }
inline vec2 inversesqrt(vec2 f) { return vec2( rsqrt_fast(f.x), rsqrt_fast(f.y) ); }
inline vec3 inversesqrt(vec3 f) { return vec3( rsqrt_fast(f.x), rsqrt_fast(f.y), rsqrt_fast(f.z) ); }
inline vec4 inversesqrt(vec4 f) { return vec4( rsqrt_fast(f.x), rsqrt_fast(f.y), rsqrt_fast(f.z), rsqrt_fast(f.w) ); }

// Common Functions

inline vec2 abs(vec2 f) { return vec2( abs(f.x), abs(f.y) ); }
inline vec3 abs(vec3 f) { return vec3( abs(f.x), abs(f.y), abs(f.z) ); }
inline vec4 abs(vec4 f) { return vec4( abs(f.x), abs(f.y), abs(f.z), abs(f.w) ); }

inline float sign(float f) { return f == 0 ? 0 : (f < 0 ? -1 : 1); }
inline vec2 sign(vec2 f) { return vec2( f.x == 0 ? 0 : (f.x < 0 ? -1 : 1), f.y == 0 ? 0 : (f.y < 0 ? -1 : 1) ); }
inline vec3 sign(vec3 f) { return vec3( f.x == 0 ? 0 : (f.x < 0 ? -1 : 1), f.y == 0 ? 0 : (f.y < 0 ? -1 : 1), f.z == 0 ? 0 : (f.z < 0 ? -1 : 1) ); }
inline vec4 sign(vec4 f) { return vec4( f.x == 0 ? 0 : (f.x < 0 ? -1 : 1), f.y == 0 ? 0 : (f.y < 0 ? -1 : 1),
        f.z == 0 ? 0 : (f.z < 0 ? -1 : 1), f.w == 0 ? 0 : (f.w < 0 ? -1 : 1) ); }

inline vec2 floor(vec2 f) { return vec2( floor(f.x), floor(f.y) ); }
inline vec3 floor(vec3 f) { return vec3( floor(f.x), floor(f.y),floor(f.z) ); }
inline vec4 floor(vec4 f) { return vec4( floor(f.x), floor(f.y),floor(f.z), floor(f.w) ); }

inline vec2 ceil(vec2 f) { return vec2( ceil(f.x), ceil(f.y) ); }
inline vec3 ceil(vec3 f) { return vec3( ceil(f.x), ceil(f.y), ceil(f.z) ); }
inline vec4 ceil(vec4 f) { return vec4( ceil(f.x), ceil(f.y), ceil(f.z), ceil(f.w) ); }

inline ivec2 trunc(vec2 f) { return ivec2( trunc(f.x), trunc(f.y) ); }
inline ivec3 trunc(vec3 f) { return ivec3( trunc(f.x), trunc(f.y), trunc(f.z) ); }
inline ivec4 trunc(vec4 f) { return ivec4( trunc(f.x), trunc(f.y), trunc(f.z), trunc(f.w) ); }

inline float fract(float f) { return f - floor(f); }
inline vec2 fract(vec2 f) { return f - floor(f); }
inline vec3 fract(vec3 f) { return f - floor(f); }
inline vec4 fract(vec4 f) { return f - floor(f); }

inline float mod(float x, float y) { return x - y * floor(x / y); }
inline vec2 mod(vec2 f, float m) { return vec2( mod(f.x, m), mod(f.y, m) ); }
inline vec3 mod(vec3 f, float m) { return vec3( mod(f.x, m), mod(f.y, m), mod(f.z, m) ); }
inline vec4 mod(vec4 f, float m) { return vec4( mod(f.x, m), mod(f.y, m), mod(f.z, m), mod(f.w, m) ); }

inline float modf(float x, float y) { return x - y * floor(x / y); }
inline vec2 modf(vec2 f, vec2 m) { return vec2( mod(f.x, m.x), mod(f.y, m.y) ); }
inline vec3 modf(vec3 f, vec3 m) { return vec3( mod(f.x, m.x), mod(f.y, m.y), mod(f.z, m.z) ); }
inline vec4 modf(vec4 f, vec4 m) { return vec4( mod(f.x, m.x), mod(f.y, m.y), mod(f.z, m.z), mod(f.w, m.w) ); }

inline vec2 min(vec2 a, vec2 b) { return vec2( min(a.x, b.x), min(a.y, b.y) ); }
inline vec2 min(vec2 a, float m) { return vec2( min(a.x, m), min(a.y, m) ); }
inline vec3 min(vec3 a, vec3 b) { return vec3( min(a.x, b.x), min(a.y, b.y), min(a.z, b.z) ); }
inline vec3 min(vec3 a, float m) { return vec3( min(a.x, m), min(a.y, m), min(a.z, m) ); }
inline vec4 min(vec4 a, vec4 b) { return vec4( min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w) ); }
inline vec4 min(vec4 a, float m) { return vec4( min(a.x, m), min(a.y, m), min(a.z, m), min(a.w, m) ); }

inline vec2 max(vec2 a, vec2 b) { return vec2( max(a.x, b.x), max(a.y, b.y) ); }
inline vec2 max(vec2 a, float m) { return vec2( max(a.x, m), max(a.y, m) ); }
inline vec3 max(vec3 a, vec3 b) { return vec3( max(a.x, b.x), max(a.y, b.y), max(a.z, b.z) ); }
inline vec3 max(vec3 a, float m) { return vec3( max(a.x, m), max(a.y, m), max(a.z, m) ); }
inline vec4 max(vec4 a, vec4 b) { return vec4( max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w) ); }
inline vec4 max(vec4 a, float m) { return vec4( max(a.x, m), max(a.y, m), max(a.z, m), max(a.w, m) ); }

inline vec2 clamp(vec2 f, float minVal, float maxVal) { return vec2( clamp(f.x, minVal, maxVal), clamp(f.y, minVal, maxVal) ); }
inline vec2 clamp(vec2 f, vec2 minVal, vec2 maxVal) { return vec2( clamp(f.x, minVal.x, maxVal.x), clamp(f.y, minVal.y, maxVal.y) ); }
inline vec3 clamp(vec3 f, float minVal, float maxVal) { return vec3( clamp(f.x, minVal, maxVal), clamp(f.y, minVal, maxVal), clamp(f.z, minVal, maxVal) ); }
inline vec3 clamp(vec3 f, vec3 minVal, vec3 maxVal) { return vec3( clamp(f.x, minVal.x, maxVal.x), clamp(f.y, minVal.y, maxVal.y), clamp(f.z, minVal.z, maxVal.z) ); }
inline vec4 clamp(vec4 f, float minVal, float maxVal) { return vec4( clamp(f.x, minVal, maxVal), clamp(f.y, minVal, maxVal), clamp(f.z, minVal, maxVal), clamp(f.w, minVal, maxVal) ); }
inline vec4 clamp(vec4 f, vec4 minVal, vec4 maxVal) { return vec4( clamp(f.x, minVal.x, maxVal.x), clamp(f.y, minVal.y, maxVal.y), clamp(f.z, minVal.z, maxVal.z), clamp(f.w, minVal.w, maxVal.w) ); }

inline float mix(float a, float b, float s) { return a + s * (b - a); }
inline vec2 mix(vec2 a, vec2 b, float s) { return vec2( mix(a.x, b.x, s), mix(a.y, b.y, s) ); }
inline vec2 mix(vec2 a, vec2 b, vec2 s) { return a + s * (b - a); }
inline vec3 mix(vec3 a, vec3 b, float s) { return vec3( mix(a.x, b.x, s), mix(a.y, b.y, s), mix(a.z, b.z, s) ); }
inline vec3 mix(vec3 a, vec3 b, vec3 s) { return a + s * (b - a); }
inline vec4 mix(vec4 a, vec4 b, float s) { return vec4( mix(a.x, b.x, s), mix(a.y, b.y, s), mix(a.z, b.z, s), mix(a.w, b.w, s) ); }
inline vec4 mix(vec4 a, vec4 b, vec4 s) { return a + s * (b - a); }

inline float step(float y, float x) { return x >= y ? 1.0 : 0.0; }
inline vec2 step(vec2 y, vec2 x) { return vec2( step(y.x, x.x), step(y.y, x.y) ); }
inline vec3 step(vec3 y, vec3 x) { return vec3( step(y.x, x.x), step(y.y, x.y), step(y.z, x.z) ); }
inline vec4 step(vec4 y, vec4 x) { return vec4( step(y.x, x.x), step(y.y, x.y), step(y.z, x.z), step(x.w, y.w) ); }

inline float smoothstep(float minValue, float maxValue, float x) {
    float t = clamp((x - minValue) / (maxValue - minValue), 0.0, 1.0);
    return t * t * (3.0f - 2.0f * t);
}
inline vec2 smoothstep(vec2 a, vec2 b, float x) { return vec2( smoothstep(a.x, b.x, x), smoothstep(a.y, b.y, x) ); }
inline vec3 smoothstep(vec3 a, vec3 b, float x) { return vec3( smoothstep(a.x, b.x, x), smoothstep(a.y, b.y, x), smoothstep(a.z, b.z, x) ); }
inline vec4 smoothstep(vec4 a, vec4 b, float x) { return vec4( smoothstep(a.x, b.x, x), smoothstep(a.y, b.y, x), smoothstep(a.z, b.z, x), smoothstep(a.w, b.w, x) ); }

// Geometric Functions

inline float dot(vec2 a, vec2 b) { return a.x * b.x + a.y * b.y; }
inline float dot(vec3 a, vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline float dot(vec4 a, vec4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }

inline float length(vec2 p) { return sqrt(dot(p, p)); }
inline float length(vec3 p) { return sqrt(dot(p, p)); }
inline float length(vec4 p) { return sqrt(dot(p, p)); }

inline float distance(float a, float b) { return abs(a - b); }
inline float distance(vec2 a, vec2 b) { return length(a - b); }
inline float distance(vec3 a, vec3 b) { return length(a - b); }
inline float distance(vec4 a, vec4 b) { return length(a - b); }

inline vec3 cross(vec3 a, vec3 b) {
    return vec3( a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x );
}

inline vec2 normalize(vec2 f) { return f / length(f); }
inline vec3 normalize(vec3 f) { return f / length(f); }
inline vec4 normalize(vec4 f) { return f / length(f); }

inline vec2 faceforward(vec2& n, vec2 i, vec2 nref) {
    vec2 r;
    float k = dot(nref, i) < 0;
    if (k < 0.0f)
        r = n;
    else
        r = -n;
    return r;
}
inline vec3 faceforward(vec3& n, vec3 i, vec3 nref) {
    vec3 r;
    float k = dot(nref, i) < 0;
    if (k < 0.0f)
        r = n;
    else
        r = -n;
    return r;
}
inline vec4 faceforward(vec4& n, vec4 i, vec4 nref) {
    vec4 r;
    float k = dot(nref, i) < 0;
    if (k < 0.0f)
        r = n;
    else
        r = -n;
    return r;
}

inline vec2 reflect(vec2 i, vec2 n) { return i - 2.0f * n * dot(n, i); }
inline vec3 reflect(vec3 i, vec3 n) { return i - 2.0f * n * dot(n, i); }
inline vec4 reflect(vec4 i, vec4 n) { return i - 2.0f * n * dot(n, i); }

inline vec2 refract(vec2 i, vec2 n, float rindex) {
    vec2 r;
    float k = 1.0f - rindex * rindex * (1.0f - dot(n, i) * dot(n, i));
    if (k < 0.0f)
        r = 0;
    else
        r = rindex * i - (rindex * dot(n, i) + sqrt(k)) * n;
    return r;
}
inline vec3 refract(vec3 i, vec3 n, float rindex) {
    vec3 r;
    float k = 1.0f - rindex * rindex * (1.0f - dot(n, i) * dot(n, i));
    if (k < 0.0f)
        r = 0;
    else
        r = rindex * i - (rindex * dot(n, i) + sqrt(k)) * n;
    return r;
}
inline vec4 refract(vec4 i, vec4 n, float rindex) {
    vec4 r;
    float k = 1.0f - rindex * rindex * (1.0f - dot(n, i) * dot(n, i));
    if (k < 0.0f)
        r = 0;
    else
        r = rindex * i - (rindex * dot(n, i) + sqrt(k)) * n;
    return r;
}

inline float determinant(mat2& m) {
    return m.m[0][0] * m.m[1][1] - m.m[1][0] * m.m[0][1];
}
inline float determinant(mat3& m) {
    return m.m[0][0] * (m.m[1][1] * m.m[2][2] - m.m[2][1] * m.m[1][2])
         - m.m[1][0] * (m.m[0][1] * m.m[2][2] - m.m[2][1] * m.m[0][2])
         + m.m[2][0] * (m.m[0][1] * m.m[1][2] - m.m[1][1] * m.m[0][2]);
}
inline float determinant(mat4& m) {
    float sf00 = m.m[2][2] * m.m[3][3] - m.m[3][2] * m.m[2][3];
    float sf01 = m.m[2][1] * m.m[3][3] - m.m[3][1] * m.m[2][3];
    float sf02 = m.m[2][1] * m.m[3][2] - m.m[3][1] * m.m[2][2];
    float sf03 = m.m[2][0] * m.m[3][3] - m.m[3][0] * m.m[2][3];
    float sf04 = m.m[2][0] * m.m[3][2] - m.m[3][0] * m.m[2][2];
    float sf05 = m.m[2][0] * m.m[3][1] - m.m[3][0] * m.m[2][1];
    vec4 detcof = vec4((m.m[1][1] * sf00 - m.m[1][2] * sf01 + m.m[1][3] * sf02),
                     - (m.m[1][0] * sf00 - m.m[1][2] * sf03 + m.m[1][3] * sf04),
                       (m.m[1][0] * sf01 - m.m[1][1] * sf03 + m.m[1][3] * sf05),
                     - (m.m[1][0] * sf02 - m.m[1][1] * sf04 + m.m[1][2] * sf05));
    return m.m[0][0] * detcof[0] + m.m[0][1] * detcof[1] + m.m[0][2] * detcof[2] + m.m[0][3] * detcof[3];
}

inline mat2 matrixCompMult(mat2& a, mat2& b) {
    mat2 r;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            r.m[i][j] = a.m[i][j] * b.m[i][j];
        }
    }
    return r;
}
inline mat3 matrixCompMult(mat3& a, mat3& b) {
    mat3 r;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            r.m[i][j] = a.m[i][j] * b.m[i][j];
        }
    }
    return r;
}
inline mat4 matrixCompMult(mat4& a, mat4& b) {
    mat4 r;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            r.m[i][j] = a.m[i][j] * b.m[i][j];
        }
    }
    return r;
}

inline float inverse(float m) { return 1.0 / m; }
inline mat2 inverse(mat2& m) {
    float det = (m.m[0][0]*m.m[1][1] - m.m[0][1]*m.m[1][0]);
    return mat2(vec2(m.m[1][1], -m.m[0][1]) / det, vec2(-m.m[1][0], m.m[0][0]) / det);
}
inline mat3 inverse(mat3& m) {
    float a00 = m.m[0][0], a01 = m.m[0][1], a02 = m.m[0][2];
    float a10 = m.m[1][0], a11 = m.m[1][1], a12 = m.m[1][2];
    float a20 = m.m[2][0], a21 = m.m[2][1], a22 = m.m[2][2];
    float b01 = a22 * a11 - a12 * a21;
    float b11 = -a22 * a10 + a12 * a20;
    float b21 = a21 * a10 - a11 * a20;
    float det = a00 * b01 + a01 * b11 + a02 * b21;
    return mat3(vec3(b01, (-a22 * a01 + a02 * a21), (a12 * a01 - a02 * a11)) / det,
              vec3(b11, (a22 * a00 - a02 * a20), (-a12 * a00 + a02 * a10)) / det,
              vec3(b21, (-a21 * a00 + a01 * a20), (a11 * a00 - a01 * a10)) / det);
}

inline mat4 inverse(mat4& m) {
    float
        a00 = m.m[0][0], a01 = m.m[0][1], a02 = m.m[0][2], a03 = m.m[0][3],
        a10 = m.m[1][0], a11 = m.m[1][1], a12 = m.m[1][2], a13 = m.m[1][3],
        a20 = m.m[2][0], a21 = m.m[2][1], a22 = m.m[2][2], a23 = m.m[2][3],
        a30 = m.m[3][0], a31 = m.m[3][1], a32 = m.m[3][2], a33 = m.m[3][3];
    float
        b00 = a00 * a11 - a01 * a10,
        b01 = a00 * a12 - a02 * a10,
        b02 = a00 * a13 - a03 * a10,
        b03 = a01 * a12 - a02 * a11,
        b04 = a01 * a13 - a03 * a11,
        b05 = a02 * a13 - a03 * a12,
        b06 = a20 * a31 - a21 * a30,
        b07 = a20 * a32 - a22 * a30,
        b08 = a20 * a33 - a23 * a30,
        b09 = a21 * a32 - a22 * a31,
        b10 = a21 * a33 - a23 * a31,
        b11 = a22 * a33 - a23 * a32;
    float det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    return mat4(
        vec4(a11 * b11 - a12 * b10 + a13 * b09, a02 * b10 - a01 * b11 - a03 * b09, a31 * b05 - a32 * b04 + a33 * b03, a22 * b04 - a21 * b05 - a23 * b03) / det,
        vec4(a12 * b08 - a10 * b11 - a13 * b07, a00 * b11 - a02 * b08 + a03 * b07, a32 * b02 - a30 * b05 - a33 * b01, a20 * b05 - a22 * b02 + a23 * b01) / det,
        vec4(a10 * b10 - a11 * b08 + a13 * b06, a01 * b08 - a00 * b10 - a03 * b06, a30 * b04 - a31 * b02 + a33 * b00, a21 * b02 - a20 * b04 - a23 * b00) / det,
        vec4(a11 * b07 - a10 * b09 - a12 * b06, a00 * b09 - a01 * b07 + a02 * b06, a31 * b01 - a30 * b03 - a32 * b00, a20 * b03 - a21 * b01 + a22 * b00) / det);
}

inline mat2 transpose(mat2& m) {
    mat2 r;
    r.m[0][0] = m.m[0][0];
    r.m[0][1] = m.m[1][0];
    r.m[1][0] = m.m[0][1];
    r.m[1][1] = m.m[1][1];
    return r;
}
inline mat3 transpose(mat3& m) {
    mat3 r;
    r.m[0][0] = m.m[0][0];
    r.m[0][1] = m.m[1][0];
    r.m[0][2] = m.m[2][0];
    r.m[1][0] = m.m[0][1];
    r.m[1][1] = m.m[1][1];
    r.m[1][2] = m.m[2][1];
    r.m[2][0] = m.m[0][2];
    r.m[2][1] = m.m[1][2];
    r.m[2][2] = m.m[2][2];
    return r;
}
inline mat4 transpose(mat4& m) {
    mat4 r;
    r.m[0][0] = m.m[0][0];
    r.m[0][1] = m.m[1][0];
    r.m[0][2] = m.m[2][0];
    r.m[0][3] = m.m[3][0];
    r.m[1][0] = m.m[0][1];
    r.m[1][1] = m.m[1][1];
    r.m[1][2] = m.m[2][1];
    r.m[1][3] = m.m[3][1];
    r.m[2][0] = m.m[0][2];
    r.m[2][1] = m.m[1][2];
    r.m[2][2] = m.m[2][2];
    r.m[2][3] = m.m[3][2];
    r.m[3][0] = m.m[0][3];
    r.m[3][1] = m.m[1][3];
    r.m[3][2] = m.m[2][3];
    r.m[3][3] = m.m[3][3];
    return r;
}

// Vector Relational Functions


// sampler
struct CombinedImageSampler {
    uniform int width;
    uniform int height;
    uniform float* data;
};

inline vec4 getSampleColor(uniform CombinedImageSampler& sampler, int u, int v) {
    int index = 4 * (v * sampler.width + u);
    return vec4(sampler.data[index], sampler.data[index + 1], sampler.data[index + 2], sampler.data[index + 3]);
}

vec4 texture(uniform CombinedImageSampler& sampler, const vec2 uv) {
    float u = uv.x;
    float v = uv.y;

#if TEXTURE_WRAPPING == GL_REPEAT
    u = mod(u, 1.0f);
    v = mod(v, 1.0f);
#elif TEXTURE_WRAPPING == GL_MIRRORED_REPEAT
    if (trunc(u) & 1)
        u = frac(u);
    else
        u = 1.0f - frac(u);
    if (trunc(v) & 1)
        v = frac(v);
    else
        v = 1.0f - frac(v);
#elif TEXTURE_WRAPPING == GL_CLAMP_TO_EDGE
    u = clamp(u, 0.0f, 1.0f);
    v = clamp(v, 0.0f, 1.0f);
#else  // TEXTURE_WRAPPING == GL_CLAMP_TO_BORDER
    if (u > 1.0f || u < 0.0f || v > 1.0f || v < 0.0f)
        return vec4(0, 0, 0, 0);
#endif

    u *= (sampler.width - 1);
    v *= (sampler.height - 1);

#if TEXTURE_FILTERING == GL_NEAREST
    return getSampleColor(sampler, round(u), round(v));
#else // TEXTURE_FILTERING == GL_LINEAR
    int u0 = trunc(u);
    int v0 = trunc(v);
    int u1 = u0 == sampler.width - 1 ? u0 : u0 + 1;
    int v1 = v0 == sampler.height - 1 ? v0 : v0 + 1;
    vec4 c00 = getSampleColor(sampler, u0, v0);
    vec4 c10 = getSampleColor(sampler, u1, v0);
    vec4 c01 = getSampleColor(sampler, u0, v1);
    vec4 c11 = getSampleColor(sampler, u1, v1);
    float ufract = fract(u);
    float ufractinv = 1 - ufract;
    float vfract = fract(v);
    float vfractinv = 1 - vfract;
    float w00 = ufractinv * vfractinv;
    float w10 = ufract * vfractinv;
    float w01 = ufractinv * vfract;
    float w11 = ufract * vfract;
    return c00 * w00 + c10 * w10 + c01 * w01 + c11 * w11;
#endif
}