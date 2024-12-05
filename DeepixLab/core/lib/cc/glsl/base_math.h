#ifndef _MATH_H_
#define _MATH_H_

#include "glsl.h"

uint32_t bit_count(uint64_t i)
{
    i = i - ((i >> 1) & 0x5555555555555555UL);
    i = (i & 0x3333333333333333UL) + ((i >> 2) & 0x3333333333333333UL);
    return (uint32_t)( (((i + (i >> 4)) & 0xF0F0F0F0F0F0F0FUL) * 0x101010101010101UL) >> 56 );
}

#endif //_MATH_H_