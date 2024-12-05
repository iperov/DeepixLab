#ifndef _INDEX_H_
#define _INDEX_H_

#include "glsl.h"

#define IDX2D(H,W,y,x) ( (uint32_t)(y) * (uint32_t)(W) + (uint32_t)(x) )

#define IDX3D(H,W,C,y,x,c) ( IDX2D(H,W,y,x) * (uint32_t)(C) + (uint32_t) (c) )

#endif //_INDEX_H_
