#include "glsl.h"


template <typename T>  T hash32(T x)
{
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

template <typename T> T hash32(T x, T y) { return hash32(x+hash32(y)); }
template <typename T> T hash32(T x, T y, T z) { return hash32(x+hash32(y+hash32(z))); }

// uniform float [0..1]
inline float hashf(uint32 x) { return (float) (int32)(hash32(x) & 0x7FFFFFFF) / (float)0x7FFFFFFF; }
inline float hashf(uint32 x, uint32 y) { return (float) (int32)(hash32(x,y) & 0x7FFFFFFF) / (float)0x7FFFFFFF; }
inline float hashf(uint32 x, uint32 y, uint32 z) { return (float) (int32)(hash32(x,y,z) & 0x7FFFFFFF) / (float)0x7FFFFFFF; }

inline vec2 hashf2(uint32 x) { return vec2(hashf(x^0x34F85A93), hashf(x^0x85F93D5)); }

inline vec2 hashf2(uint32 x, uint32 y) 
{ 
    uint32 h = hash32(x,y);
    return vec2(hashf(h^0x34F85A93), hashf(h^0x85FB93D5)); 
}

inline vec3 hashf3(uint32 x, uint32 y) 
{ 
    uint32 h = hash32(x,y);
    return vec3(hashf(h^0x34F85A93), hashf(h^0x85FB93D5), hashf(h^0x6253DF84));   //for 4th ^0x25FC3625
}
