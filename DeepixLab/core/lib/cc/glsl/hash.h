#ifndef _RANDOM_H_
#define _RANDOM_H_

#include "glsl.h"

uint64_t hash64(uint64_t x)
{
    x ^= x >> 32;
    x *= 0xd6e8feb86659fd93U;
    x ^= x >> 32;
    x *= 0xd6e8feb86659fd93U;
    x ^= x >> 32;
    return x;
}


uint32_t hash32(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

uint32_t hash32(uint32_t x, uint32_t y) { return hash32(x+hash32(y)); }
uint32_t hash32(uint32_t x, uint32_t y, uint32_t z) { return hash32(x+hash32(y+hash32(z))); }

float hashf(uint32_t x) { return (float)hash32(x) / (float)0xFFFFFFFF; }
float hashf(uint32_t x, uint32_t y) { return (float)hash32(x,y) / (float)0xFFFFFFFF; }
float hashf(uint32_t x, uint32_t y, uint32_t z) { return (float)hash32(x,y,z) / (float)0xFFFFFFFF; }

vec2 hashf2(uint32_t x) { return vec2(hashf(x^0x34F85A93), hashf(x^0x85FB93D5)); }

vec2 hashf2(uint32_t x, uint32_t y) 
{ 
    uint32_t h = hash32(x,y);
    return vec2(hashf(h^0x34F85A93), hashf(h^0x85FB93D5)); 
}

vec3 hashf3(uint32_t x, uint32_t y) 
{ 
    uint32_t h = hash32(x,y);
    return vec3(hashf(h^0x34F85A93), hashf(h^0x85FB93D5), hashf(h^0x6253DF84));   //for 4th ^0x25FC3625
}


#endif //_RANDOM_H_