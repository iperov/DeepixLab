import random
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from core.lib import cc, ispc

from ..FImage import FImage


def patch_dropout_mask( W : int, H : int,
                        h_patch_count : int,
                        v_patch_count : int,
                        prob : float = 0.5,
                        seed : int|None = None,
                        ) -> FImage:
    """"""
    x = np.random.RandomState(seed).binomial(1, prob, size=(v_patch_count,h_patch_count,1)).astype(np.uint8)
    x *= 255
    x = cv2.resize(x, (W,H), interpolation=cv2.INTER_NEAREST)
    return FImage.from_numpy(x)

def clouds(W, H=None, scales=[64, 32, 16, 8]) -> FImage:
    if H is None:
        H = W

    img = FImage.zeros(H,W,1)

    mod = 1.0
    for scale in scales:
        img = img + noise(W // scale, H // scale).resize(W, H, interp=FImage.Interp.CUBIC).apply(lambda img: img*mod)
        mod /= 2.0

    return img.satushift()

def binary_clouds(W, H=None, density=0.5, scales=[64, 32, 16, 8]) -> FImage:
    img = clouds(W,H, scales=scales)
    img = img.apply(lambda x: np.where(x >= density, np.float32(0.0), np.float32(1.0)) )

    return img

def stripes(W, H=None, line_width=1, density=0.5, scales=[64, 32, 16, 8]) -> FImage:
    img = binary_clouds(W, H, density=density, scales=scales)
    g_blur = line_width / 2.0
    img = img.gaussian_blur(g_blur) * img.invert().gaussian_blur(g_blur)
    return img.satushift()

def binary_stripes(W, H=None, line_width=2, density=0.5, scales=[64, 32, 16, 8]) -> FImage:
    img = stripes(W,H, line_width=line_width, density=density, scales=scales)
    img = img.apply(lambda x: np.where(x > 0.5, np.float32(1.0), np.float32(0.0)) )
    return img

lib_path = cc.get_lib_path(Path(__file__).parent, Path(__file__).stem)

@cc.lib_import(lib_path)
def c_cut_edges_mask(img : cc.c_float32_p, W : cc.c_uint32, H : cc.c_uint32, angle_deg : cc.c_float32, edge_dist : cc.c_float32, cx : cc.c_float32 = 0.5, cy : cc.c_float32 = 0.5, cw : cc.c_float32 = 0.5, ch : cc.c_float32 = 0.5, init : cc.c_bool = True) -> cc.c_void: ...
def cut_edges_mask(in_out : np.ndarray, angle_deg : float, edge_dist : float, cx : float = 0.5, cy : float = 0.5, cw : float = 0.5, ch : float = 0.5, init=True):
    """in_out float32"""
    in_out = np.ascontiguousarray(in_out)
    H,W,C = in_out.shape
    c_cut_edges_mask(in_out.ctypes.data_as(cc.c_float32_p), W, H, angle_deg, edge_dist, cx, cy, cw, ch, init)

@cc.lib_import(lib_path)
def c_noise(img : cc.c_float32_p, size : cc.c_uint32, seed : cc.c_uint32) -> cc.c_void: ...
def noise(W : int, H : int, seed : int|None = None) -> FImage:
    """H,W,1, float32"""
    if seed is None:
        seed = random.getrandbits(32)

    img = np.empty( (H,W,1), np.float32 )
    c_noise(img.ctypes.data_as(cc.c_float32_p), W*H, seed)
    return FImage.from_numpy(img)


@cc.lib_import(lib_path)
def c_bezier(img : cc.c_void_p, W : cc.c_uint32, H : cc.c_uint32, ax : cc.c_float32, ay : cc.c_float32, bx : cc.c_float32, by : cc.c_float32, cx : cc.c_float32, cy : cc.c_float32, width : cc.c_float32 ) -> None: ...
def bezier(W : int, H : int, ax : float, ay : float, bx : float, by : float, cx : float, cy : float, width : float) -> FImage:
    """H,W,1, float32"""
    img = np.empty( (H,W,1), np.float32 )
    c_bezier(img.ctypes.data_as(cc.c_float32_p), W, H, ax, ay, bx, by, cx, cy, width)
    return FImage.from_numpy(img)

@cc.lib_import(lib_path)
def c_circle_faded(img : cc.c_float32_p, W : cc.c_uint32, H : cc.c_uint32, cx : cc.c_float32, cy : cc.c_float32, fs : cc.c_float32, fe : cc.c_float32) -> cc.c_void: ...
def circle_faded(W : int, H : int, cx : float, cy : float, fs : float, fe : float) -> FImage:
    """H,W,1, float32"""
    img = np.empty( (H,W,1), np.float32 )
    c_circle_faded(img.ctypes.data_as(cc.c_float32_p), W, H, cx, cy, fs, fe)
    return FImage.from_numpy(img)

@cc.lib_import(lib_path)
def c_icon_loading(img : cc.c_float32_p, W : cc.c_uint32, H : cc.c_uint32, C : cc.c_uint32, R_inner : cc.c_float32, R_outter : cc.c_float32, edge_smooth : cc.c_float32, bg_color : cc.c_float32_p, fg_color : cc.c_float32_p, u_time : cc.c_float32) -> cc.c_void: ...

def icon_loading(size : int, R_inner : float, R_outter : float, edge_smooth : float,
                    bg_color : Sequence[float], fg_color : Sequence[float], u_time : float) -> FImage:
    """H,W,4, float32"""
    bg_color = np.float32(bg_color)
    fg_color = np.float32(fg_color)
    if bg_color.shape != fg_color.shape:
        raise ValueError()
    C = bg_color.shape[0]
    img = np.empty( (size,size,C), np.float32 )

    c_icon_loading(img.ctypes.data_as(cc.c_float32_p), size, size, C, R_inner, R_outter, edge_smooth, bg_color.ctypes.data_as(cc.c_float32_p), fg_color.ctypes.data_as(cc.c_float32_p), u_time)
    return FImage.from_numpy(img)


@cc.lib_import(lib_path)
def c_test_gen(img : cc.c_float32_p, W : cc.c_uint32, H : cc.c_uint32) -> cc.c_void: ...
def test_gen(W : int, H : int, seed : int = None) -> FImage:
    """H,W,1, float32"""
    if seed is None:
        seed = random.getrandbits(32)

    img = np.empty( (H,W,1), np.float32 )
    c_test_gen(img.ctypes.data_as(cc.c_float32_p), W, H, seed)
    return FImage.from_numpy(img)


def setup_compile():
    ispc.compile_o(Path(__file__).parent / 'gen_ispc.c')

    return cc.Compiler(Path(__file__).parent).shared_library(lib_path).minimal_c().optimize().include_glsl().include('gen.cpp').include('gen_ispc.o').compile()

#setup_compile()
