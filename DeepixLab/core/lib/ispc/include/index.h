#define IDX2D(H,W,y,x)(                       \
                               (    (uint32) (y)    \
                                  * (uint32) (W)    \
                               )                    \
                               + (uint32) (x)       \
                            )                       \


#define IDX3D(H,W,C,y,x,c) (                          \
                                    IDX2D(H,W,y,x)    \
                                    * (uint32) (C)          \
                                    + (uint32) (c)          \
                                 )                          \
                  

#define IDX3D_2(WC,C,y,x,c) (y*WC + C*x + c)
                                                
/*

(y * W * C) + (x*C) + c

((y*W) + x)*C + c


(z * NyNx) + (y * Nx) + x


*/