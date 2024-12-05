#ifndef _COLOR_H_
#define _COLOR_H_

#include "glsl.h"

void bgr_to_hsv(float b, float g, float r, float &h, float &s, float &v)
{
    float maxc = max(max(r, g), b);
    float minc = min(min(r, g), b);
    float delta = maxc - minc;
    
    float oh;
    if(delta > 0) {
        if(maxc == r) {
            oh = 60 * (mod(((g - b) / delta), 6));
        } else if(maxc == g) {
            oh = 60 * (((b - r) / delta) + 2);
        } else if(maxc == b) {
            oh = 60 * (((r - g) / delta) + 4);
        }
        
        if(maxc > 0) {
            s = delta / maxc;
        } else {
            s = 0;
        }
        
        v = maxc;
    } else {
        oh = 0;
        s = 0;
        v = maxc;
    }
    
    if(oh < 0) {
        oh = 360 + oh;
    }
    
    h = oh / 360.0f;
}

void hsv_to_bgr(float h, float s, float v, float &b, float &g, float &r)
{
    float c = v * s; // Chroma
    float hp = mod(h*360.0 / 60.0, 6);
    float x = c * (1 - abs(mod(hp, 2) - 1));
    float m = v - c;
    
    if(0 <= hp && hp < 1) {
        r = c;
        g = x;
        b = 0;
    } else if(1 <= hp && hp < 2) {
        r = x;
        g = c;
        b = 0;
    } else if(2 <= hp && hp < 3) {
        r = 0;
        g = c;
        b = x;
    } else if(3 <= hp && hp < 4) {
        r = 0;
        g = x;
        b = c;
    } else if(4 <= hp && hp < 5) {
        r = x;
        g = 0;
        b = c;
    } else if(5 <= hp && hp < 6) {
        r = c;
        g = 0;
        b = x;
    } else {
        r = 0;
        g = 0;
        b = 0;
    }
    
    r += m;
    g += m;
    b += m;
}

#endif //_COLOR_H_