import cv2
import numpy as np
import numpy.random as nprnd

from .. import gen as lib_gen
from ..FImage import FImage
from .Geo import Geo


def patch_dropout_mask( W : int, H : int,
                        h_patch_count_range=[2,32],
                        v_patch_count_range=[2,32],
                        prob : float = 0.5,
                        seed : int|None = None,
                        ) -> FImage:
    """"""
    rnd_state = np.random.RandomState(seed)
    
    return lib_gen.patch_dropout_mask(W, H, 
                                      h_patch_count=rnd_state.randint(h_patch_count_range[0], h_patch_count_range[1]+1),
                                      v_patch_count=rnd_state.randint(v_patch_count_range[0], v_patch_count_range[1]+1),
                                      prob=prob,
                                      seed=rnd_state.randint(2**31))
    
    
    
def cut_edges_mask(W : int, H : int = None, cuts_range=[1,4], seed=None) -> FImage:
    """returns (H,W,1) f32 mask image"""
    if H is None:
        H = W
    img = np.empty( (H,W,1), np.float32 )
    rnd_state = np.random.RandomState(seed)
    for i in range( rnd_state.randint(cuts_range[0], cuts_range[1]+1) ):
        lib_gen.cut_edges_mask(img, angle_deg = rnd_state.uniform()*360, edge_dist=rnd_state.uniform(), init=i==0)
    return FImage.from_numpy(img)


def circle_faded_mask(W, H=None, cx_range=[-1.0, 2.0], cy_range=[-1.0, 2.0], f_range=[0.0, 1.5], seed=None) -> FImage:
    if H is None:
        H = W
    HW_max = max(H,W)
    rnd_state = np.random.RandomState(seed)

    return lib_gen.circle_faded(W, H,   rnd_state.uniform(cx_range[0]*W, cx_range[1]*W),
                                        rnd_state.uniform(cy_range[0]*H, cy_range[1]*H),
                                        fs := rnd_state.uniform(f_range[0]*HW_max, f_range[1]*HW_max),
                                        fe = rnd_state.uniform(fs, f_range[1]*HW_max)   )

def noise_clouds(W, H=None) -> FImage:
    if H is None:
        H = W

    img = lib_gen.noise(1,1).resize(W, H)

    d = 1
    while True:
        dW = W // d
        dH = H // d
        if dW <= 1 or dH <= 1:
            break
        if np.random.randint(2) == 0:

            x = lib_gen.noise(W // d, H // d).resize(W, H, interp=FImage.Interp.CUBIC  if np.random.randint(2) == 0 else
                                                                  FImage.Interp.LANCZOS4)
            if np.random.randint(2) == 0:
                img = img * x
            else:
                img = img + x
        d *= 2

    return img.satushift()

def binary_clouds(W, H=None, density_range=[0.33, 0.66]) -> FImage:
    return lib_gen.binary_clouds(W,H, density=np.random.uniform(*density_range),)

def binary_stripes(W, H=None, line_width_range=[2,8], density_range=[0.33, 0.66]) -> FImage:
    return lib_gen.binary_stripes(W,H,  line_width=nprnd.randint(line_width_range[0], line_width_range[1]+1),
                                        density=np.random.uniform(*density_range), )

def rgb_exposure(img : FImage, r_exposure_var=1.0, g_exposure_var=1.0, b_exposure_var=1.0):
    C = img.shape[-1]
    if C != 3:
        raise ValueError('img C must == 3')

    r = nprnd.uniform(-r_exposure_var, r_exposure_var)
    g = nprnd.uniform(-g_exposure_var, g_exposure_var)
    b = nprnd.uniform(-b_exposure_var, b_exposure_var)

    return img.channel_exposure(exposure = (b, g, r))

def channels_deform(img : FImage, alpha=0.25, seed : int|None = None) -> FImage:
    rnd_state = np.random.RandomState(seed)

    if img.shape[-1] == 3:
        x = FImage.from_b_g_r(  Geo(seed=rnd_state.randint(2**31)).transform(img.ch1_from_b(), deform_intensity=1.0),
                                Geo(seed=rnd_state.randint(2**31)).transform(img.ch1_from_g(), deform_intensity=1.0),
                                Geo(seed=rnd_state.randint(2**31)).transform(img.ch1_from_r(), deform_intensity=1.0) )
    else:
        x = Geo(seed=rnd_state.randint(2**31)).transform(img.ch1(), deform_intensity=1.0)

    return img.blend(x, FImage.ones_f32_like(img), alpha=0.25)


def levels(img : FImage, in_b_range=[0.0, 0.25], in_w_range=[0.75, 1.0], in_g_range=[0.8, 1.25],
                         out_b_range=[0.0, 0.25], out_w_range=[0.75, 1.0], seed : int|None = None)-> FImage:
    C = img.shape[-1]
    rnd_state = np.random.RandomState(seed)
    return img.levels(  in_b = ([ rnd_state.uniform(*in_b_range) for _ in range(C) ]),
                        in_w = ([ rnd_state.uniform(*in_w_range) for _ in range(C) ]),
                        in_g = ([ rnd_state.uniform(*in_g_range) for _ in range(C) ]),

                        out_b = ([ rnd_state.uniform(*out_b_range) for _ in range(C) ]),
                        out_w = ([ rnd_state.uniform(*out_w_range) for _ in range(C) ]))


def hsv_shift(img : FImage, h_offset_range=[-1.0,1.0], s_offset_range=[-0.25,0.25], v_offset_range=[-0.25,0.25], seed : int|None = None) -> FImage:
    rnd_state = np.random.RandomState(seed)
    return img.hsv_shift(rnd_state.uniform(*h_offset_range),rnd_state.uniform(*s_offset_range),rnd_state.uniform(*v_offset_range))

def box_sharpen(img : FImage, kernel_range=None, power_range=[0.0, 2.0], seed : int|None = None) -> FImage:
    if kernel_range is None:
        kernel_range = [1, max(1, max(img.shape[0:2]) // 32) ]

    rnd_state = np.random.RandomState(seed)
    return img.box_sharpen(kernel_size=int(rnd_state.uniform(*kernel_range)),
                           power=rnd_state.uniform(*power_range),
                           )

def gaussian_sharpen(img : FImage, sigma_range=None, power_range=[0.0, 2.0], seed : int|None = None) -> FImage:
    if sigma_range is None:
        sigma_range = [0, max(img.shape[0:2]) / 32.0 ]
    rnd_state = np.random.RandomState(seed)
    return img.gaussian_sharpen(sigma=rnd_state.uniform(*sigma_range), power=rnd_state.uniform(*power_range))

def gaussian_blur(img : FImage, sigma_range=None, seed : int|None = None) -> FImage:
    if sigma_range is None:
        sigma_range = [0, max(img.shape[0:2]) / 32.0 ]
    rnd_state = np.random.RandomState(seed)
    return img.gaussian_blur(sigma=rnd_state.uniform(*sigma_range))

def median_blur(img : FImage, kernel_range=None, seed : int|None = None) -> FImage:
    if kernel_range is None:
        kernel_range = [1, max(1, max(img.shape[0:2]) // 32) ]
    rnd_state = np.random.RandomState(seed)
    return img.median_blur(kernel_size=rnd_state.randint(kernel_range[0], kernel_range[1]+1))

def motion_blur(img : FImage, kernel_range=None, angle_range=[0,360], seed : int|None = None) -> FImage:
    if kernel_range is None:
        kernel_range = [1, max(1, max(img.shape[0:2]) // 8) ]
    rnd_state = np.random.RandomState(seed)
    return img.motion_blur(kernel_size=rnd_state.randint(kernel_range[0], kernel_range[1]+1), angle=rnd_state.randint(*angle_range))

def glow_shade(img : FImage, mask : FImage, inner=True, glow=False) -> FImage:
    """"""
    H,W,_ = img.shape
    HW_max = max(H,W)

    img = img.f32()
    mask = mask.f32()

    if inner:
        halo = img*mask
    else:
        halo = img*(1-mask)
    halo = halo.gaussian_blur(sigma=np.random.uniform(HW_max/16, HW_max/4))

    if glow:
        img = img + halo
    else:
        img = img - halo
    img = img.clip()

    return img

def resize(img : FImage, size_range=[0.25, 1.0], interp=FImage.Interp.LINEAR, seed : int|None = None) -> FImage:
    H,W,C = img.shape

    rnd_state = np.random.RandomState(seed)
    s = rnd_state.uniform(*size_range)

    img = img.resize (int(s*W), int(s*H), interp=interp )
    img = img.resize (W, H, interp=interp )

    return img


def jpeg_artifacts(img : FImage, quality_range = [10,100], seed : int|None = None ) -> FImage:
    """
     quality    0-100
    """
    rnd_state = np.random.RandomState(seed)
    quality = rnd_state.randint(quality_range[0],quality_range[1]+1)

    dtype = img.dtype
    img = img.u8().HWC()

    ret, result = cv2.imencode('.jpg', img, params=[cv2.IMWRITE_JPEG_QUALITY, quality] )
    if not ret:
        raise Exception('unable to compress jpeg')
    img = cv2.imdecode(result, flags=cv2.IMREAD_UNCHANGED)

    return FImage.from_numpy(img).to_dtype(dtype)


