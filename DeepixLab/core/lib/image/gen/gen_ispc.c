#include "glsl.h"
#include "index.h"
#include "macro.h"
#include "sd.h"
#include "hash.h"
#include "spline.h"

export void noise(uniform float img[], uniform uint32 size, uniform uint32 seed)
{
    foreach (i = 0 ... size)
        img[i] = hashf(i+seed);
}


export void circle_faded(uniform float img[], uniform uint32 W, uniform uint32 H, uniform float cx, uniform float cy, uniform float fs, uniform float fe)
{
    uniform float d = max(fs, fe) - fs;
    if (d == 0.0f) 
        d = 1.0f;
    
    uniform vec2 c = vec2(cx, cy);
    foreach(y = 0 ... H, x = 0 ... W)
    { 
        vec2 pt = vec2(x, y);
        float dist = length(pt - c);
        img[ IDX2D(H,W,y,x) ] = min(1.0f, max(0.0f, 1.0f - (dist - fs) / d));
    }
}


export void cut_edges_mask( uniform float img[], uniform uint32 W, uniform uint32 H,
                            uniform float angle_deg, uniform float edge_dist, uniform float cx, uniform float cy, uniform float cw, uniform float ch, uniform bool init)
{
    // Ellipse center point
    vec2 c_pt = vec2(cx *W, cy * H);
    // Ellipse radius x,y
    float cwr = cw * W / 2.0f;
    float chr = ch * H / 2.0f;
    
    // Corner points
    vec2 c0_pt = vec2(0.0, 0.0);
    vec2 c1_pt = vec2(W  , 0.0);
    vec2 c2_pt = vec2(W  , H  );
    vec2 c3_pt = vec2(0.0, H  );
    
    // Point on ellipse -0.1 deg
    float angle_rad_p0 = (angle_deg - 0.1) * PI / 180.0;
    vec2 e0_pt = c_pt + vec2( cwr*cos(angle_rad_p0), chr*-sin(angle_rad_p0) );
    
    // Point on ellipse +0.1 deg
    float angle_rad_p1 = (angle_deg + 0.1) * PI / 180.0;
    vec2 e1_pt = c_pt + vec2( cwr*cos(angle_rad_p1), chr*-sin(angle_rad_p1) );
    
    // Ellipse tangent now is e0x, e0y -> e1x, e1y
    // Calculate max dist from ellipse tanget to any corner
    float corner_dist = max(max(max(sd_line(c0_pt, e0_pt, e1_pt),
                                    sd_line(c1_pt, e0_pt, e1_pt)),
                                    sd_line(c2_pt, e0_pt, e1_pt)),
                                    sd_line(c3_pt, e0_pt, e1_pt));
    
    foreach(y = 0 ... H, x = 0 ... W)
    {
        vec2 pt = vec2(x, y) + 0.5f;
        
        // Dist to ellipse tangent
        float d;
        if (sd_line(pt, e0_pt, e1_pt) >= edge_dist*corner_dist)
            d = 0; else d = 1;
        
        if (init)
            img[ IDX2D(H,W,y,x) ] = d;
        else 
            img[ IDX2D(H,W,y,x) ] *= d;
    }
}

export void bezier(uniform float img[], uniform uint32 W, uniform uint32 H, 
                   uniform float ax, uniform float ay, uniform float bx, uniform float by, uniform float cx, uniform float cy, uniform float width)
{
    vec2 a = vec2(ax, ay);
    vec2 b = vec2(bx, by);
    vec2 c = vec2(cx, cy);
    foreach(y = 0 ... H, x = 0 ... W)
    { 
        vec2 pt = vec2(x, y);
        
        float dist = sd_bezier(pt, a, b, c);
        
        img[ IDX2D(H,W,y,x) ] = abs(dist) <= width/2 ? 1.0f : 0.0f;
    }
}

export void icon_loading(uniform float img[], 
                            uniform uint32 H, uniform uint32 W, uniform uint32 C, 
                            uniform float R_inner, uniform float R_outter, uniform float edge_smooth, 
                            uniform float bg_color[], uniform float fg_color[], uniform float u_time)
{    
    vec2 center = vec2(W / 2.0f, H / 2.0f);
    
    foreach(y = 0 ... H, x = 0 ... W)
    { 
        vec2 pix_coord = vec2(x, y) + vec2(0.5f, 0.5f);
        
        float a = sd_loading_circle(pix_coord, center, u_time, R_inner, R_outter, edge_smooth);
        
        for (uniform uint32 c=0; c<C; ++c)
            img[IDX3D(H,W,C,y,x,c)] = bg_color[c]*(1-a) + fg_color[c]*a;
    }
}




/*
struct TestStruct
{
    varying float v;
};

float func(float* v)
{
    return v[0];
}
*/
/* uniform TestStruct* uniform values = uniform new uniform TestStruct[1];
values[0].v = 1.0f;

foreach(y=0 ... H, x=0 ... W)
{
    varying TestStruct* asd = (varying TestStruct*) &values[0];
    
    out[ IDX2D(H,W,y,x) ] = func( &asd->v);        
}*/

#define Hair_pts_max 100
struct Hair
{
    int32 pts_count;
    vec2 pts[Hair_pts_max];
};

export void test_gen(uniform float out[], uniform uint32 W, uniform uint32 H, uniform uint32 seed)
{
    // amount of hair per pixel for full opaque
    uniform int32 hair_density = 3;
    uniform float hair_density_frac = 1.0f / hair_density;
    
    // Generate random hair
    
    // Mode 1. Linear.
    
    uniform int32 hair_count = 30;//W*hair_density;   //W*hair_density;// hash32(seed) % 600;
    
    uniform Hair* uniform hairs = uniform new uniform Hair[hair_count];
    
    foreach(hid=0 ... hair_count)
    {
        vec2 pt_start = vec2(hid*3 + 0.5f,0);
        vec2 pt_dir = vec2(0,1);
        
        int32 pts_count = H / 4;
        
        hairs[hid].pts_count = pts_count;
        
        float hid_length = H;
        
        for (uniform int32 pid=0; pid < pts_count; ++pid)
        {
            hairs[hid].pts[pid] = pt_start + pt_dir*pid*4;
        }
    }
    
    seed = hash32(seed+1);
    
    foreach(hid=0 ... hair_count)
    {
        uint32 id = hash32(seed+hid) % hairs[hid].pts_count;
        
        vec2 pt = hairs[hid].pts[id];
        hairs[hid].pts[id] = pt; 
        
        //float dist = FLT_MAX;
        //vec2 o_pt;
        /*
        for (uniform int32 s_hid=0; s_hid<hair_count, s_hid != hid; ++s_hid)
        {
            for (uniform int32 pid=0; pid<hairs[s_hid].pts_count; ++pid)
            {
                vec2 s_pt = hairs[s_hid].pts[pid];
                
                float s_dist = length(s_pt-pt);
                if (s_dist < dist)
                {
                    o_pt = s_pt;
                    dist = s_dist;
                }
            }
        }*/
        
        //vec2 d_pt = o_pt-pt;
        
//        hairs[hid].pts[id] = pt; 
        
        //vec2(0,0);//pt + 0.1f;//normalize(d_pt);//*length(d_pt)*0.1f;
    }
    
    foreach(y=0 ... H, x=0 ... W)
    {
        vec2 pt = vec2(x,y) + vec2(0.5f,0.5f);
        
        float v = 0;
        
        for (uniform int32 hid=0; hid<hair_count; ++hid)
        {
            float spline_w=0;
            float s = sd_catmull_spline(pt, hairs[hid].pts, hairs[hid].pts_count, 8, spline_w);
            
            if (s < 0.5f)
            {
                v = clamp(v+hair_density_frac, 0, 1);
            }
            //v = s;//mix(1.0f, 0.0f, s );  //*spline_w;
        }

        
        out[ IDX2D(H,W,y,x) ] = v;
        
    }
    
    //delete[] hls;
    
}


/*
        for (int32 i=0; i<hair_count; ++i)
        {
           
            
            sd_catmull_spline(pt, (varying vec2*) &hls[i].pts[0], hls[i].pts_count, (varying int32)32, spline_w);
        }*/
        
        /*vec2 pts[6] = {
            vec2(0,0), 
            vec2(50,0), 
            vec2(50,50), 
            vec2(0,50),
            vec2(50,100),
            vec2(50,200),
        };
        float spline_w=0;
        
        float v = sd_catmull_spline(pt, pts, 6, 32, spline_w);
        
        v = mix(1.0f, 0.0f, v )*spline_w;

        
        out[ IDX2D(H,W,y,x) ] = v;*/