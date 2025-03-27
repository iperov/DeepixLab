from __future__ import annotations

import colorsys
import functools
import os
import struct
import time
from enum import StrEnum
from pathlib import Path
from typing import Callable, Self, Sequence, Tuple, overload

import cv2
import numpy as np

from .. import cc, ispc
from ..collections import FDict
from ..hash import Hash64
from ..math import FAffMat2, FRectf, FVec2i
from .ImageFormat import ImageFormatType, get_image_format_by_type


class FImage:
    """
    Immutable image.
    Represented as np.ndarray HWC image (u8/f32).
    Provides image processing methods.

    Saves lines of code, because you don't need to check image channels, dtype or format.
    You just get FImage as arg, and then transform it to your needs.

    Example
    ```
    npi.ch1().u8().CHW()
    ```
    """
    __slots__ = ['_img']

    class Interp(StrEnum):
        NEAREST = '@(FImage.Interp.NEAREST)'
        LINEAR = '@(FImage.Interp.LINEAR)'
        CUBIC = '@(FImage.Interp.CUBIC)'
        LANCZOS4 = '@(FImage.Interp.LANCZOS4)'

    class Border(StrEnum):
        CONSTANT = '@(FImage.Border.CONSTANT)'
        REFLECT = '@(FImage.Border.REFLECT)'
        REPLICATE = '@(FImage.Border.REPLICATE)'

    @staticmethod
    def _border_to_cv(border : Border):
        return _cv_border[border]

    @staticmethod
    def full_u8_like(img : FImage, color) -> FImage:
        """full like img.HW but with color """
        H,W,_ = img.shape

        return FImage.from_numpy(np.full( (H,W,len(color)), color, np.uint8))

    @staticmethod
    def full_f32_like(img : FImage, color) -> FImage:
        """full like img.HW but with color """
        H,W,_ = img.shape

        return FImage.from_numpy(np.full( (H,W,len(color)), color, np.float32))

    @staticmethod
    def full_u8(W, H, color) -> FImage:
        """full of u8 (H,W, color )"""
        return FImage.from_numpy(np.full( (H,W,len(color)), color, np.uint8))

    @staticmethod
    def full_f32(W, H, color) -> FImage:
        """full of f32 (H,W, color )"""
        return FImage.from_numpy(np.full( (H,W,len(color)), color, np.float32))

    @staticmethod
    def ones(H,W,C) -> FImage:
        """ones of f32 (H,W,C)"""
        return FImage.from_numpy(np.ones( (H,W,C), np.float32))

    @staticmethod
    def ones_f32_like(img : FImage) -> FImage:
        """ones of f32 of img's shape"""
        self = FImage.__new__(FImage)
        self._img = np.ones(img.shape, np.float32)
        return self

    @staticmethod
    def zeros(H,W,C) -> FImage:
        """zeros of f32 (H,W,C)"""
        return FImage.from_numpy(np.zeros((H,W,C), np.float32))

    @staticmethod
    def zeros_f32_like(img : FImage) -> FImage:
        """zeros of f32 of img's shape"""
        self = FImage.__new__(FImage)
        self._img = np.zeros(img.shape, np.float32)
        return self

    @staticmethod
    def from_b_g_r(b : FImage, g : FImage,  r : FImage) -> FImage:
        """"""
        self = FImage.__new__(FImage)
        self._img = np.concatenate([b.ch1().HWC(), g.ch1().HWC(), r.ch1().HWC()], -1)
        return self


    @staticmethod
    def from_bgr_a(bgr : FImage, a : FImage) -> FImage:
        """"""
        self = FImage.__new__(FImage)
        self._img = np.concatenate([bgr.bgr().HWC(), a.ch(1).HWC()], -1)
        return self

    @staticmethod
    def from_file(path : Path|str) -> FImage:
        """raises on error"""
        if not isinstance(path, Path):
            path = Path(path)

        with open(path, "rb") as stream:
            b = memoryview(stream.read())

        if path.suffix == '.rgb8':
            s = struct.calcsize('III')
            H,W,C = struct.unpack('III', b[:s])

            img = np.frombuffer(b[s:s+H*W*C*1], np.uint8).reshape(H,W,C)
        else:
            buf = np.frombuffer(b, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise Exception(f'Error loading {path}')
            dtype = img.dtype

            if dtype == np.uint16:
                img = img.astype(np.float32) / 65535.0

            if img.dtype not in [np.uint8, np.float32]:
                raise ValueError('unsupported dtype')

        return FImage.from_numpy(img)



    @staticmethod
    def from_state(state : FDict|None) -> FImage|None:
        state = FDict(state)
        if (dtype := state.get('dtype', None)) is not None and \
           (shape := state.get('shape', None)) is not None and \
           (data := state.get('data', None)) is not None and \
           (compressed := state.get('compressed', None)) is not None:
            if compressed:
                return FImage.from_numpy(cv2.imdecode(np.asarray(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED))
            else:
                return FImage.from_numpy( np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape))
        return None

    @staticmethod
    def from_numpy(img : np.ndarray, channels_last = True):
        """```
            img     np.ndarray

            dtype must be uint8[0..255] or float32[0..1]

            acceptable format:
            HW[C] channels_last(True)
            [C]HW channels_last(False)

            C must be
                0: assume 1 ch
                1: assume grayscale
                3: assume BGR   (opencv & qt compliant)
                4: assume BGRA  (opencv & qt compliant)
        ```"""
        if channels_last:
            if (shape_len := len(img.shape)) != 3:
                if shape_len == 2:
                    img = img[:,:,None]
                else:
                    raise ValueError(f'Wrong shape len {shape_len}')
        else:
            if (shape_len := len(img.shape)) != 3:
                if shape_len == 2:
                    img = img[None,:,:]
                else:
                    raise ValueError(f'Wrong shape len {shape_len}')
            img = img.transpose(1,2,0)

        if (C := img.shape[-1]) not in [1,2,3,4]:
            raise ValueError(f'img.C must be 0,1,2,3,4 but {C} provided.')

        if np.issubdtype(img.dtype, np.floating):
            img = img.astype(np.float32, copy=False)
        elif img.dtype != np.uint8:
            raise ValueError(f'img.dtype must be np.uint8 or np.floating, but passed {img.dtype}.')

        r = FImage.__new__(FImage)
        r._img = img
        return r

    def __init__(self):
        self._img : np.ndarray
        raise Exception('to create FImage use .from_ methods')

    def clone(self) -> Self:
        new_self = (c := self.__class__).__new__(c)
        new_self._img = self._img
        return new_self

    def copy(self) -> Self:
        self = self.clone()
        self._img = self._img.copy()
        return self

    def get_state(self, compressed = True) -> FDict:
        """Represent FImage as dependency-free data

            compressed(True)    u8:  PNG compression
                                f32: no compression
        """
        img = self._img

        c = False
        data = img.tobytes()
        if compressed and img.dtype == np.uint8:
            ret, buf = cv2.imencode('.png', img)
            if ret:
                c = True
                data = buf

        return FDict({  'compressed' : c,
                        'data'  : data,
                        'shape' : img.shape,
                        'dtype' : img.dtype.name})

    @property
    def width(self) -> int: return self._img.shape[1]
    @property
    def height(self) -> int: return self._img.shape[0]

    @property
    def shape(self) -> Tuple[int, int, int]:
        """get (H,W,C) dims"""
        return self._img.shape
    @property
    def size(self) -> FVec2i:
        """ (W,H) as FVec2i"""
        return FVec2i(self._img.shape[1], self._img.shape[0])
    @property
    def dtype(self) -> np.dtype:
        """np.uint8 / np.float32"""
        return self._img.dtype

    def HWC(self) -> np.ndarray:
        return self._img

    def CHW(self) -> np.ndarray:
        return self._img.transpose(2,0,1)

    def HW(self) -> np.ndarray:
        return self.HWC()[:,:,0]

    def ch(self, ch : int) -> FImage:
        """
            same as call ch1()/bgr()/bgra()
            ch should be 1/3/4
        """
        if ch == 1:
            return self.ch1()
        elif ch == 3:
            return self.bgr()
        elif ch == 4:
            return self.bgra()
        raise ValueError('ch must be 1/3/4')

    def ch1(self) -> FImage:
        """reduce to single ch"""
        C = (img := self._img).shape[-1]
        if C == 1:
            return self

        self = self.__class__.__new__(self.__class__)
        self._img = np.dot(img[...,0:3], np.array([0.1140, 0.5870, 0.299], np.float32))[...,None].astype(img.dtype, copy=False)
        return self

    def ch1_from_b(self) -> FImage:
        """gray from blue if possible or ones """
        H,W,C = (img := self._img).shape
        if C >= 3:
            self = self.__class__.__new__(self.__class__)
            self._img = img[...,0:1]
            return self
        return FImage.ones(H,W,1)

    def ch1_from_g(self) -> FImage:
        """gray from green if possible or ones """
        H,W,C = (img := self._img).shape
        if C >= 3:
            self = self.__class__.__new__(self.__class__)
            self._img = img[...,1:2]
            return self
        return FImage.ones(H,W,1)

    def ch1_from_r(self) -> FImage:
        """gray from red if possible or ones """
        H,W,C = (img := self._img).shape
        if C >= 3:
            self = self.__class__.__new__(self.__class__)
            self._img = img[...,2:3]
            return self
        return FImage.ones(H,W,1)

    def ch1_from_a(self) -> FImage:
        """gray from alpha if possible or ones """
        H,W,C = (img := self._img).shape
        if C == 4:
            self = self.__class__.__new__(self.__class__)
            self._img = img[...,3:4]
            return self
        return FImage.ones(H,W,1)

    def discard_a(self) -> FImage:
        """if ch==4 sets bgr, otherwise do nothing"""
        if self._img.shape[-1] == 4:
            self = self.bgr()
        return self

    def bgr(self) -> FImage:
        """"""
        C = (img := self._img).shape[-1]
        if C == 3:
            return self
        if C == 1:
            img = np.repeat(img, 3, -1)
        if C == 4:
            img = img[...,:3]

        self = self.__class__.__new__(self.__class__)
        self._img = img
        return self

    def bgra(self) -> FImage:
        """"""
        C = (img := self._img).shape[-1]
        if C == 4:
            return self

        if C == 1:
            img = np.repeat(img, 3, -1)
            C = 3
        if C == 3:
            img = np.pad(img, ( (0,0), (0,0), (0,1) ), mode='constant', constant_values=255 if img.dtype == np.uint8 else 1.0 )

        self = self.__class__.__new__(self.__class__)
        self._img = img
        return self

    def swap_rb(self) -> FImage:
        C = (img := self._img).shape[-1]
        if C == 3:
            img = img[..., [2,1,0]]
        if C == 1:
            return self
        if C == 4:
            img = img[..., [2,1,0,3]]

        self = self.__class__.__new__(self.__class__)
        self._img = img
        return self


    def to_dtype(self, dtype) -> FImage:
        """
        convert to dtype.
            f32 / 255 -> u8
            u8 * 255 -> f32

        allowed dtypes: np.uint8, np.float32
        """
        if dtype == np.uint8:     return self.u8()
        elif dtype == np.float32: return self.f32()
        else: raise ValueError('unsupported dtype')

    def f32(self) -> FImage:
        """
        Convert to float32

            convert(True)   if current image dtype uint8, then image will be divided by / 255.0
        """
        dtype = (img := self._img).dtype
        if dtype == np.uint8:
            img = np.ascontiguousarray(img)
            img_out = np.empty_like(img, dtype=np.float32)

            c_u8_to_f32(img_out.ctypes.data_as(cc.c_float32_p), img.ctypes.data_as(cc.c_uint8_p), np.prod(img.shape))
            self = self.__class__.__new__(self.__class__)
            self._img = img_out

        return self

    def u8(self) -> FImage:
        """
        convert to uint8

        if current image dtype is f32, then image will be multiplied by *255
        """
        img = self._img
        if img.dtype == np.float32:
            img = np.ascontiguousarray(img)
            img_out = np.empty_like(img, dtype=np.uint8)

            c_f32_to_u8(img_out.ctypes.data_as(cc.c_uint8_p), img.ctypes.data_as(cc.c_float32_p), np.prod(img.shape))

            self = self.__class__.__new__(self.__class__)
            self._img = img_out

        return self

    def apply(self, func : Callable[ [np.ndarray], np.ndarray]) -> FImage:
        """
        apply your own function on image.

        image has HWC format. Do not change format, keep dtype either u8 or float, dims can be changed.

        ```
        example:
        .apply( lambda img: img-[102,127,63] )
        ```
        """
        img = func(self._img)
        if img.dtype not in [np.uint8, np.float32]:
            raise Exception('dtype result of apply() must be u8/f32')
        self = self.__class__.__new__(self.__class__)
        self._img = img

        return self


    def clip(self, min=None, max=None) -> FImage:
        """clip to min,max.

        if min/max not specified and dtype is f32, clips to 0..1
        """
        img = self._img
        if min is None and max is None:
            if img.dtype == np.float32:
                min = 0.0
                max = 1.0
        if min is not None and max is not None:
            self = self.__class__.__new__(self.__class__)
            self._img = np.clip(img, min, max)

        return self

    def satushift(self) -> FImage:
        """fastest version of ```.apply(lambda x: x - x.min()).apply(lambda x: x / x.max())``` """
        img = np.ascontiguousarray(self._img)
        img_out = np.empty_like(img)
        H,W,C = img.shape
        args = ( img_out.ctypes.data_as(cc.c_void_p), img.ctypes.data_as(cc.c_void_p), H*W*C )

        if img.dtype == np.uint8:
            c_satushift_u8(*args)
        else:
            c_satushift_f32(*args)

        self = self.__class__.__new__(self.__class__)
        self._img = img_out

        return self

    def draw_key_points(self, pts : Sequence[ Tuple[int|float, int|float] ], ) -> FImage:
        """
        draw key points

            pts list of [x,y] int in range of W,H
        """
        img = self.f32().HWC().copy()
        H,W,C = img.shape
        pts_len = len(pts)

        if C == 1:
            get_color=lambda i: ( i / pts_len, )
        elif C == 3:
            get_color=lambda i: colorsys.hsv_to_rgb(i / pts_len, 1, 1)
        elif C == 4:
            get_color=lambda i: colorsys.hsv_to_rgb(i / pts_len, 1, 1) + (1,)

        for i, (x,y) in enumerate(pts):
            cv2.circle(img, (int(x), int(y)), radius=max(1, max(W,H)//64), color=get_color(i), lineType=cv2.LINE_AA)

        self = self.__class__.__new__(self.__class__)
        self._img = img
        return self

    def draw_rect(self, rect : FRectf, color, thickness=1) -> FImage:
        """
        draw rect

         color  tuple of values      should be the same as img color channels and dtype
        """
        img = self.HWC().copy()

        pts = [ np.int32(pt.as_np()) for pt in rect.as_4pts()]

        cv2.line(img, pts[0], pts[1], color, thickness, lineType=cv2.LINE_AA)
        cv2.line(img, pts[1], pts[2], color, thickness, lineType=cv2.LINE_AA)
        cv2.line(img, pts[2], pts[3], color, thickness, lineType=cv2.LINE_AA)
        cv2.line(img, pts[3], pts[0], color, thickness, lineType=cv2.LINE_AA)
        self = self.__class__.__new__(self.__class__)
        self._img = img

        return self

    def dilate(self, iterations : int, k_size=3) -> FImage:
        """"""
        img = self._img
        el = np.asarray(cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k_size,k_size)))

        self = self.__class__.__new__(self.__class__)
        img = cv2.dilate(img, el, iterations = iterations )
        if len(img.shape) == 2:
            img = img[...,None]
        self._img = img

        return self

    def invert(self) -> FImage:
        img = self._img

        if self.dtype == np.uint8:
            img = 255 - img
        else:
            img = 1.0 - img

        self = self.__class__.__new__(self.__class__)
        self._img = img

        return self

    def gaussian_blur(self, sigma : float) -> FImage:
        """
        Spatial gaussian blur.

            sigma  float
        """
        sigma = max(0, sigma)
        if sigma == 0:
            return self

        H,W,C = (img := self._img).shape

        self = self.__class__.__new__(self.__class__)
        self._img = cv2.GaussianBlur(img, (0,0), sigma).reshape(H,W,C)

        return self

    def blend(self, other : FImage, mask : FImage, alpha = 1.0) -> FImage:
        """
        Pixel-wise blending `self*(1-mask*alpha) + other*mask*alpha`

        Image will be forced to f32.

            alpha  [0.0 ... 1.0]
        """
        #self_dtype = self.dtype
        #other_dtype = other.dtype

        if self.size != other.size:
            raise ValueError('self.size != other.size')

        img = np.ascontiguousarray(self._img)
        img_out = np.empty_like(img)
        other = np.ascontiguousarray(other.f32()._img)
        mask = np.ascontiguousarray(mask.f32()._img)

        args = (img_out.ctypes.data_as(cc.c_void_p), *img_out.shape,
                img.ctypes.data_as(cc.c_void_p),
                other.ctypes.data_as(cc.c_float32_p), *other.shape,
                mask.ctypes.data_as(cc.c_float32_p), *mask.shape,
                alpha)

        if img.dtype == np.uint8:
            c_blend_u8(*args)
        else:
            c_blend_f32(*args)

        self = self.__class__.__new__(self.__class__)
        self._img = img_out

        return self

    def box_sharpen(self, kernel_size : int, power : float) -> FImage:
        """
         kernel_size   int     kernel size

         power  float   0 .. 1.0 (or higher)

        Image will be forced to f32.
        """
        power = max(0, power)
        if power == 0:
            return self

        H,W,C = (img := self.f32()._img).shape

        img = cv2.filter2D(img, -1, _box_sharpen_kernel(kernel_size, power))
        img = np.clip(img, 0, 1, out=img)
        img = img.reshape(H,W,C)

        self = self.__class__.__new__(self.__class__)
        self._img = img

        return self

    def channel_exposure(self, exposure) -> FImage:
        """```
        Exposure applied per every channel.

            exposure    (C,) float

        Image will be forced to f32.
        ```"""
        H,W,C = (img := self._img).shape

        exposure = np.float32(exposure)
        if exposure.shape[0] == 1:
            exposure = np.tile(exposure, (C,))
        if C != exposure.shape[0]:
            raise Exception('exposure must match C dims')

        img = np.ascontiguousarray(self._img)
        img_out = np.empty_like(img)

        args = (img_out.ctypes.data_as(cc.c_void_p), img.ctypes.data_as(cc.c_void_p), H, W, C, exposure.ctypes.data_as(cc.c_float32_p))
        if img.dtype==np.uint8:
            c_channel_exposure_u8(*args)
        else:
            c_channel_exposure_f32(*args)

        self = self.__class__.__new__(self.__class__)
        self._img = img_out
        return self


    def gaussian_sharpen(self, sigma : float, power : float) -> FImage:
        """
         sigma  float

         power  float   0 .. 1.0 and higher

        Image will be forced to f32.
        """
        sigma = max(0, sigma)
        if sigma == 0:
            return self

        H,W,C = (img := self.f32()._img).shape

        img = cv2.addWeighted(img, 1.0 + power,
                              cv2.GaussianBlur(img, (0, 0), sigma), -power, 0)
        img = np.clip(img, 0, 1, out=img)
        img = img.reshape(H,W,C)
        self = self.__class__.__new__(self.__class__)
        self._img = img

        return self

    def gaussian_blur(self, sigma : float) -> FImage:
        """
         sigma  float

        Image will be forced to f32.
        """
        sigma = max(0, sigma)
        if sigma == 0:
            return self

        H,W,C = (img := self.f32()._img).shape

        img = cv2.GaussianBlur(img, (0,0), sigma)
        img = img.reshape(H,W,C)
        self = self.__class__.__new__(self.__class__)
        self._img = img

        return self

    def histogram(self, normalized=False) -> np.ndarray:
        """
        Compute u8 histogram.

            normalized

        returns [C, 256] np.uint32 or float32(normalized)
        """
        img = np.ascontiguousarray(self.u8()._img)

        H,W,C = img.shape

        if normalized:
            hist_out = np.empty( (C,256), np.float32)
            c_histogram_u8_f32(hist_out.ctypes.data_as(cc.c_float32_p), img.ctypes.data_as(cc.c_uint8_p), H*W, C)
        else:
            hist_out = np.empty( (C,256), np.uint32)
            c_histogram_u8_uint32(hist_out.ctypes.data_as(cc.c_uint32_p), img.ctypes.data_as(cc.c_uint8_p), H*W, C)

        return hist_out

    # def fit_in (self, TW = None, TH = None, pad_to_target : bool = False, allow_upscale : bool = False, interp : FImage.Interp = Interp.LINEAR) -> FImage:
    #     """
    #     fit image in w,h keeping aspect ratio

    #         TW,TH           int/None     target width,height

    #         pad_to_target   bool    pad remain area with zeros

    #         allow_upscale   bool    if image smaller than TW,TH, img will be upscaled

    #         interp          PImage.Interp. value

    #     returns scale float value
    #     """
    #     H,W,C = (img := self._img).shape

    #     if TW is not None and TH is None:
    #         scale = TW / W
    #     elif TW is None and TH is not None:
    #         scale = TH / H
    #     elif TW is not None and TH is not None:
    #         SW = W / TW
    #         SH = H / TH
    #         scale = 1.0
    #         if SW > 1.0 or SH > 1.0 or (SW < 1.0 and SH < 1.0):
    #             scale /= max(SW, SH)
    #     else:
    #         raise ValueError('TW or TH should be specified')

    #     if not allow_upscale and scale > 1.0:
    #         scale = 1.0

    #     if scale != 1.0:
    #         img = cv2.resize (img, ( int(W*scale), int(H*scale) ), interpolation=_cv_inter[interp])
    #         H,W = img.shape[0:2]
    #         img = img.reshape( (H,W,C) )

    #     if pad_to_target:
    #         w_pad = (TW-W) if TW is not None else 0
    #         h_pad = (TH-H) if TH is not None else 0
    #         if w_pad != 0 or h_pad != 0:
    #             img = np.pad(img, ( (0,h_pad), (0,w_pad), (0,0) ))

    #     return FImage(img), scale


    def h_flip(self) -> FImage:
        img = self._img
        self = self.__class__.__new__(self.__class__)
        self._img = img[:,::-1,:]
        return self
    def v_flip(self) -> FImage:
        img = self._img
        self = self.__class__.__new__(self.__class__)
        self._img = img[::-1,:,:]
        return self

    def levels(self, in_b, in_w, in_g, out_b, out_w) -> FImage:
        """```
            in_b
            in_w
            in_g
            out_b
            out_w     (1,) or (C,) float

        ```"""
        H,W,C = (img := self._img).shape

        in_b = np.float32(in_b)
        if in_b.shape[0] == 1:
            in_b = np.tile(in_b, (C,))
        in_w = np.float32(in_w)
        if in_w.shape[0] == 1:
            in_w = np.tile(in_w, (C,))
        in_g = np.float32(in_g)
        if in_g.shape[0] == 1:
            in_g = np.tile(in_g, (C,))
        out_b = np.float32(out_b)
        if out_b.shape[0] == 1:
            out_b = np.tile(out_b, (C,))
        out_w = np.float32(out_w)
        if out_w.shape[0] == 1:
            out_w = np.tile(out_w, (C,))

        img = np.ascontiguousarray(self._img)
        img_out = np.empty_like(img)

        if img.dtype==np.uint8:
            c_levels_u8(img_out.ctypes.data_as(cc.c_uint8_p), img.ctypes.data_as(cc.c_uint8_p), H, W, C, in_b.ctypes.data_as(cc.c_float32_p), in_w.ctypes.data_as(cc.c_float32_p), in_g.ctypes.data_as(cc.c_float32_p), out_b.ctypes.data_as(cc.c_float32_p), out_w.ctypes.data_as(cc.c_float32_p))
        else:
            c_levels_f32(img_out.ctypes.data_as(cc.c_float32_p), img.ctypes.data_as(cc.c_float32_p), H, W, C, in_b.ctypes.data_as(cc.c_float32_p), in_w.ctypes.data_as(cc.c_float32_p), in_g.ctypes.data_as(cc.c_float32_p), out_b.ctypes.data_as(cc.c_float32_p), out_w.ctypes.data_as(cc.c_float32_p))

        self = self.__class__.__new__(self.__class__)
        self._img = img_out

        return self

    def bilateral_filter(self, sigma : float) -> FImage:
        H,W,C = (img := self._img).shape

        self = self.__class__.__new__(self.__class__)
        self._img = cv2.bilateralFilter(img, 0, sigma, sigma).reshape(H,W,C)

        return self

    # def median_blur(self, kernel_size : int) -> FImage:
    #     """
    #      kernel_size   int     median kernel size

    #     Image will be forced to f32.
    #     """
    #     if kernel_size % 2 == 0:
    #         kernel_size += 1
    #     kernel_size = max(1, kernel_size)

    #     H,W,C = (img := self.f32()._img).shape

    #     img = cv2.medianBlur(img, kernel_size)
    #     img = img.reshape(H,W,C)
    #     return FImage(img)

    def motion_blur( self, kernel_size : int, angle : float):
        """
            kernel_size    >= 1

            angle   degrees

        Image will be forced to f32.
        """
        if kernel_size % 2 == 0:
            kernel_size += 1

        H,W,C = (img := self.f32()._img).shape

        self = self.__class__.__new__(self.__class__)
        self._img = cv2.filter2D(img, -1, _motion_blur_kernel(kernel_size, angle)).reshape(H,W,C)

        return self

    def hsv_shift(self, h_offset : float, s_offset : float, v_offset : float) -> FImage:
        """```
            H,S,V in [-1.0..1.0]
        ```"""
        H,W,C = (img := np.ascontiguousarray(self._img)).shape
        if C != 3:
            raise Exception('C must be == 3')

        img_out = np.empty_like(img)

        if img.dtype==np.uint8:
            c_hsv_shift_u8(img_out.ctypes.data_as(cc.c_uint8_p), img.ctypes.data_as(cc.c_uint8_p), *img.shape, h_offset, s_offset, v_offset)
        else:
            c_hsv_shift_f32(img_out.ctypes.data_as(cc.c_float32_p), img.ctypes.data_as(cc.c_float32_p), *img.shape, h_offset, s_offset, v_offset)

        self = self.__class__.__new__(self.__class__)
        self._img = img_out

        return self


    def pad(self, PADL : int = 0, PADT : int = None, PADR : int = None, PADB : int = None) -> FImage:
        """```
        pad with zeros
        ```"""
        if PADT is None:
            PADT = PADL
        if PADR is None:
            PADR = PADT
        if PADB is None:
            PADB = PADR
        img = self._img
        self = self.__class__.__new__(self.__class__)
        self._img = np.pad(img, ((PADT,PADB), (PADL, PADR), (0,0)))

        return self

    def pad_to_next_divisor(self, dw=None, dh=None) -> FImage:
        """
        pad image to next divisor of width/height

         dw,dh  int
        """
        H,W,_ = self._img.shape

        w_pad = 0
        if dw is not None:
            w_pad = W % dw
            if w_pad != 0:
                w_pad = dw - w_pad

        h_pad = 0
        if dh is not None:
            h_pad = H % dh
            if h_pad != 0:
                h_pad = dh - h_pad

        if w_pad != 0 or h_pad != 0:
            return self.pad(0, 0, w_pad, h_pad)
        return self

    def resize(self, OW : int, OH : int, interp : Interp = Interp.LINEAR, smooth=False) -> FImage:
        """resize to (OW,OH)"""
        H,W,C = (img := self._img).shape

        if smooth:
            while W != OW or H != OH:
                W = min(W*2.0, OW) if W < OW else max(OW, W/2.0)
                H = min(H*2.0, OH) if H < OH else max(OH, H/2.0)
                self = self.resize(int(W), int(H), interp=interp)

            return self

        if OW != W or OH != H:
            img = cv2.resize (img, (OW, OH), interpolation=_cv_inter[interp])
            img = img.reshape(OH,OW,C)
            self = self.__class__.__new__(self.__class__)
            self._img = img
        return self

    def remap(self, grid : np.ndarray, interp : Interp = Interp.LINEAR, border : Border = Border.CONSTANT) -> FImage:
        """```
            grid    HW2
        ```"""
        OH,OW,_ = grid.shape

        H,W,C = (img := self._img).shape

        self = self.__class__.__new__(self.__class__)
        self._img = cv2.remap(img, grid, None, interpolation=_cv_inter[interp], borderMode=_cv_border[border] ).reshape(OH,OW,C)
        return self

    @overload
    def warp_affine(self, mat : FAffMat2, size : FVec2i, interp : Interp = Interp.LINEAR, border : Border = Border.CONSTANT) -> Self:
        """"""
    @overload
    def warp_affine(self, mat : FAffMat2, OW : int, OH : int, interp : Interp = Interp.LINEAR, border : Border = Border.CONSTANT) -> Self:
        """"""
    def warp_affine(self, *args, **kwargs) -> Self:
        mat = kwargs.get('mat', ...)
        size = kwargs.get('size', ...)
        OW = kwargs.get('OW', ...)
        OH = kwargs.get('OH', ...)
        interp = kwargs.get('interp', FImage.Interp.LINEAR)
        border = kwargs.get('border', FImage.Border.CONSTANT)

        args_len = len(args)
        if args_len >= 1:
            mat = args[0]
        if args_len >= 2:
            arg1 = args[1]
            if isinstance(arg1, FVec2i):
                size = arg1

                if args_len >= 3:
                    interp = args[2]
                if args_len >= 4:
                    border = args[3]
            else:
                OW = arg1

                if args_len >= 3:
                    OH = args[2]

                if args_len >= 4:
                    interp = args[3]
                if args_len >= 5:
                    border = args[4]

        if not (size is Ellipsis):
            OW, OH = size

        H,W,C = (img := self._img).shape

        self = self.__class__.__new__(self.__class__)
        self._img = cv2.warpAffine(img, mat.as_np(), (OW, OH), flags=_cv_inter[interp], borderMode=_cv_border[border] ).reshape(OH,OW,C)
        return self


    def save(self, path : Path, fmt_type : ImageFormatType|None=None, quality : int = 100) -> Path:
        """```
        Save as JPEG 8   (.jpg / .jpeg) + quality
                JPEG2000 8/16 (.jp2) + quality
                PNG 8/16 (.png)  (quality ignored)
                WEBP 8   (.webp) + quality
                TIFF 8/16 (.tif / .tiff) (quality ignored)

                RGBA8   (.rgba8) (quality ignored)

        if fmt_type is specified, suffix will be replaced according the format

        if fmt_type is None, it will be determined from suffix and image dtype

        returns Path

        raise on error
        ```"""
        suffix = None
        if fmt_type is None:
            # Determine ImageFormatType from suffix and dtype
            suffix = path.suffix

            is_f32 = self.dtype == np.float32

            if suffix in ['.jpg','.jpeg']:
                fmt_type = ImageFormatType.JPEG
            elif suffix in ['.jp2']:
                fmt_type = ImageFormatType.JPEG2000_16 if is_f32 else ImageFormatType.JPEG2000
            elif suffix in ['.png']:
                fmt_type = ImageFormatType.PNG_16 if is_f32 else ImageFormatType.PNG
            elif suffix in ['.tif','.tiff']:
                fmt_type = ImageFormatType.TIFF_16
            elif suffix in ['.webp']:
                fmt_type = ImageFormatType.WEBP
            elif suffix in ['.rgb8']:
                fmt_type = ImageFormatType.RGB8
            else:
                raise Exception('unsupported format')
        fmt = get_image_format_by_type(fmt_type)

        if suffix is None:
            suffix = fmt.suffix

        parent = path.parent
        parent.mkdir(parents=True, exist_ok=True)
        out_path = parent / (path.stem+suffix)

        if out_path.exists():
            out_path.unlink()
        while os.path.exists(out_path):
            time.sleep(0.016)
                        
        if fmt_type == ImageFormatType.RGB8:
            img = np.ascontiguousarray(self.ch(3).u8().HWC())
            H,W,C = img.shape
                        
            with open(out_path, "wb") as stream:
                stream.write(struct.pack('III', H,W,C))
                stream.write(img.data)

        else:
            if fmt.bits_per_ch == 8:
                img = self.u8().HWC()
            else:
                img = np.clip(self.f32().HWC()*65535.0, 0, 65535.0).astype(np.uint16)

            if fmt_type in [ImageFormatType.JPEG]:
                imencode_args = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            elif fmt_type in [ImageFormatType.JPEG2000, ImageFormatType.JPEG2000_16]:
                imencode_args = [int(cv2.IMWRITE_JPEG2000_COMPRESSION_X1000), quality]
            elif fmt_type in [ImageFormatType.WEBP]:
                imencode_args = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
            else:
                imencode_args = []

            ret, buf = cv2.imencode(suffix, img, imencode_args)
            if not ret:
                raise Exception(f'Unable to encode image to {suffix}')
                            
            with open(out_path, "wb") as stream:
                stream.write( buf )

        return out_path


    def get_perc_hash(self) -> Hash64:
        """
        Calculates perceptual local-sensitive 64-bit hash of image

        based on http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html

        returns Hash64
        """
        hash_size, highfreq_factor = 8, 4

        img = self.ch1().resize(hash_size * highfreq_factor, hash_size * highfreq_factor).f32().HW()

        dct = cv2.dct(img)

        dct_low_freq = dct[:hash_size, :hash_size]
        bits = ( dct_low_freq > np.median(dct_low_freq) ).reshape( (hash_size*hash_size,)).astype(np.uint64)
        bits = bits << np.arange(len(bits), dtype=np.uint64)

        return Hash64(bits.sum())

    def __radd__(self, other) -> Self: return other.__add__(self)
    def __add__(self, other) -> Self:
        if isinstance(other, FImage):
            if self.shape != other.shape:
                raise Exception('shape mismatch', self.shape, other.shape)
            H,W,C = self.shape

            img = np.ascontiguousarray(self._img)
            img_out = np.empty_like(img)
            other_img = np.ascontiguousarray(other._img)

            _c_add_func[img.dtype.type][other.dtype.type](img_out.ctypes.data_as(cc.c_void_p), img.ctypes.data_as(cc.c_void_p), other_img.ctypes.data_as(cc.c_void_p), H*W*C)

            self = self.__class__.__new__(self.__class__)
            self._img = img_out
            return self
        else:
            raise Exception('FImage allowed only. Use .apply() to operate with ndarray')



    def __rsub__(self, other) -> Self: return other.__sub__(self)
    def __sub__(self, other) -> Self:
        if isinstance(other, FImage):
            if self.shape != other.shape:
                raise Exception('shape mismatch', self.shape, other.shape)
            H,W,C = self.shape

            img = np.ascontiguousarray(self._img)
            img_out = np.empty_like(img)
            other_img = np.ascontiguousarray(other._img)

            _c_sub_func[img.dtype.type][other.dtype.type](img_out.ctypes.data_as(cc.c_void_p), img.ctypes.data_as(cc.c_void_p), other_img.ctypes.data_as(cc.c_void_p), H*W*C)

            self = self.__class__.__new__(self.__class__)
            self._img = img_out
            return self
        else:
            raise Exception('FImage allowed only. Use .apply() to operate with ndarray')

    def __rmul__(self, other) -> Self: return other.__mul__(self)
    def __mul__(self, other) -> Self:
        if isinstance(other, FImage):

            if self.shape != other.shape:
                raise Exception('shape mismatch', self.shape, other.shape)
            H,W,C = self.shape



            img = np.ascontiguousarray(self._img)
            img_out = np.empty_like(img)
            other_img = np.ascontiguousarray(other._img)

            _c_mul_func[img.dtype.type][other.dtype.type](img_out.ctypes.data_as(cc.c_void_p), img.ctypes.data_as(cc.c_void_p), other_img.ctypes.data_as(cc.c_void_p), H*W*C)

            self = self.__class__.__new__(self.__class__)
            self._img = img_out
            return self
        else:
            raise Exception('FImage allowed only. Use .apply() to operate with ndarray')

    def __truediv__(self, value) -> Self: raise Exception('divide operation with FImage is not allowed. Use .apply()')
    def __rtruediv__(self, value) -> Self: raise Exception('divide operation with FImage is not allowed. Use .apply()')
    def __iadd__(self, value) -> Self: raise Exception('in-place operation with FImage is not allowed')
    def __isub__(self, value) -> Self: raise Exception('in-place operation with FImage is not allowed')
    def __imul__(self, value) -> Self: raise Exception('in-place operation with FImage is not allowed')
    def __idiv__(self, value) -> Self: raise Exception('in-place operation with FImage is not allowed')
    def __itruediv__(self, value) -> Self: raise Exception('in-place operation with FImage is not allowed')

    def __repr__(self): return self.__str__()
    def __str__(self): return f'FImage {self._img.shape} {self._img.dtype}'


_cv_inter = { FImage.Interp.NEAREST : cv2.INTER_NEAREST,
              FImage.Interp.LINEAR : cv2.INTER_LINEAR,
              FImage.Interp.CUBIC : cv2.INTER_CUBIC,
              FImage.Interp.LANCZOS4 : cv2.INTER_LANCZOS4,
               }

_cv_border = {FImage.Border.CONSTANT : cv2.BORDER_CONSTANT,
              FImage.Border.REFLECT : cv2.BORDER_REFLECT,
              FImage.Border.REPLICATE : cv2.BORDER_REPLICATE,
            }

@functools.cache
def _box_sharpen_kernel(kernel_size : int, power : float) -> np.ndarray:
    if kernel_size % 2 == 0:
        kernel_size += 1

    k = np.zeros( (kernel_size, kernel_size), dtype=np.float32)
    k[ kernel_size//2, kernel_size//2] = 1.0
    b = np.ones( (kernel_size, kernel_size), dtype=np.float32) / (kernel_size**2)
    k = k + (k - b) * power
    return k


@functools.cache
def _motion_blur_kernel(kernel_size, angle) -> np.ndarray:
    k = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    k[ (kernel_size-1)// 2 , :] = np.ones(kernel_size, dtype=np.float32)
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D( (kernel_size / 2 -0.5 , kernel_size / 2 -0.5 ) , angle, 1.0), (kernel_size, kernel_size) )
    k = k * ( 1.0 / np.sum(k) )
    return k

lib_path = cc.get_lib_path(Path(__file__).parent, Path(__file__).stem)


@cc.lib_import(lib_path)
def c_mul_u8_u8(O : cc.c_void_p, A : cc.c_void_p, B : cc.c_void_p, size : cc.c_uint32) -> cc.c_void: ...
@cc.lib_import(lib_path)
def c_mul_u8_f32(O : cc.c_void_p, A : cc.c_void_p, B : cc.c_void_p, size : cc.c_uint32) -> cc.c_void: ...
@cc.lib_import(lib_path)
def c_mul_f32_u8(O : cc.c_void_p, A : cc.c_void_p, B : cc.c_void_p, size : cc.c_uint32) -> cc.c_void: ...
@cc.lib_import(lib_path)
def c_mul_f32_f32(O : cc.c_void_p, A : cc.c_void_p, B : cc.c_void_p, size : cc.c_uint32) -> cc.c_void: ...
@cc.lib_import(lib_path)
def c_sub_u8_u8(O : cc.c_void_p, A : cc.c_void_p, B : cc.c_void_p, size : cc.c_uint32) -> cc.c_void: ...
@cc.lib_import(lib_path)
def c_sub_u8_f32(O : cc.c_void_p, A : cc.c_void_p, B : cc.c_void_p, size : cc.c_uint32) -> cc.c_void: ...
@cc.lib_import(lib_path)
def c_sub_f32_u8(O : cc.c_void_p, A : cc.c_void_p, B : cc.c_void_p, size : cc.c_uint32) -> cc.c_void: ...
@cc.lib_import(lib_path)
def c_sub_f32_f32(O : cc.c_void_p, A : cc.c_void_p, B : cc.c_void_p, size : cc.c_uint32) -> cc.c_void: ...
@cc.lib_import(lib_path)
def c_add_u8_u8(O : cc.c_void_p, A : cc.c_void_p, B : cc.c_void_p, size : cc.c_uint32) -> cc.c_void: ...
@cc.lib_import(lib_path)
def c_add_u8_f32(O : cc.c_void_p, A : cc.c_void_p, B : cc.c_void_p, size : cc.c_uint32) -> cc.c_void: ...
@cc.lib_import(lib_path)
def c_add_f32_u8(O : cc.c_void_p, A : cc.c_void_p, B : cc.c_void_p, size : cc.c_uint32) -> cc.c_void: ...
@cc.lib_import(lib_path)
def c_add_f32_f32(O : cc.c_void_p, A : cc.c_void_p, B : cc.c_void_p, size : cc.c_uint32) -> cc.c_void: ...

_c_mul_func = { np.uint8   : { np.uint8 : c_mul_u8_u8,  np.float32 : c_mul_u8_f32, },
                np.float32 : { np.uint8 : c_mul_f32_u8, np.float32 : c_mul_f32_f32}, }
_c_sub_func = { np.uint8   : { np.uint8 : c_sub_u8_u8,  np.float32 : c_sub_u8_f32, },
                np.float32 : { np.uint8 : c_sub_f32_u8, np.float32 : c_sub_f32_f32}, }
_c_add_func = { np.uint8   : { np.uint8 : c_add_u8_u8,  np.float32 : c_add_u8_f32, },
                np.float32 : { np.uint8 : c_add_f32_u8, np.float32 : c_add_f32_f32}, }

@cc.lib_import(lib_path)
def c_u8_to_f32(img_out : cc.c_float32_p, img_in : cc.c_uint8_p, size : cc.c_int32) -> cc.c_void: ...
@cc.lib_import(lib_path)
def c_f32_to_u8(img_out : cc.c_uint8_p, img_in : cc.c_float32_p, size : cc.c_int32) -> cc.c_void: ...
@cc.lib_import(lib_path)
def c_histogram_u8_uint32(hist_out : cc.c_uint32_p, img_in : cc.c_uint8_p, HW : cc.c_int32, C : cc.c_int32) -> cc.c_void: ...
@cc.lib_import(lib_path)
def c_histogram_u8_f32(hist_out : cc.c_float32_p, img_in : cc.c_uint8_p, HW : cc.c_int32, C : cc.c_int32) -> cc.c_void: ...



@cc.lib_import(lib_path)
def c_channel_exposure_u8(img_out : cc.c_void_p, img_in : cc.c_void_p, H : cc.c_uint32, W : cc.c_uint32, C : cc.c_uint32, exposure : cc.c_float32_p) -> cc.c_void: ...
@cc.lib_import(lib_path)
def c_channel_exposure_f32(img_out : cc.c_void_p, img_in : cc.c_void_p, H : cc.c_uint32, W : cc.c_uint32, C : cc.c_uint32, exposure : cc.c_float32_p) -> cc.c_void: ...


@cc.lib_import(lib_path)
def c_levels_u8(img_out : cc.c_uint8_p, img_in : cc.c_uint8_p, H : cc.c_uint32, W : cc.c_uint32, C : cc.c_uint32, in_b : cc.c_float32_p, in_w : cc.c_float32_p, in_g : cc.c_float32_p, out_b : cc.c_float32_p, out_w : cc.c_float32_p) -> cc.c_void: ...
@cc.lib_import(lib_path)
def c_levels_f32(img_out : cc.c_float32_p, img_in : cc.c_float32_p, H : cc.c_uint32, W : cc.c_uint32, C : cc.c_uint32, in_b : cc.c_float32_p, in_w : cc.c_float32_p, in_g : cc.c_float32_p, out_b : cc.c_float32_p, out_w : cc.c_float32_p) -> cc.c_void: ...

@cc.lib_import(lib_path)
def c_hsv_shift_u8(O : cc.c_uint8_p, I : cc.c_uint8_p, H : cc.c_uint32, W : cc.c_uint32, C : cc.c_uint32,
                    h_offset : cc.c_float32, s_offset : cc.c_float32, v_offset : cc.c_float32) -> cc.c_void: ...
@cc.lib_import(lib_path)
def c_hsv_shift_f32(O : cc.c_float32_p, I : cc.c_float32_p, H : cc.c_uint32, W : cc.c_uint32, C : cc.c_uint32,
                    h_offset : cc.c_float32, s_offset : cc.c_float32, v_offset : cc.c_float32) -> cc.c_void: ...

@cc.lib_import(lib_path)
def c_blend_u8(O : cc.c_void_p, OH : cc.c_int32, OW : cc.c_int32, OC : cc.c_int32,
               A : cc.c_void_p,
               B : cc.c_float32_p, BH : cc.c_int32, BW : cc.c_int32, BC : cc.c_int32,
               M : cc.c_float32_p, MH : cc.c_int32, MW : cc.c_int32, MC : cc.c_int32,
               alpha : cc.c_float32) -> cc.c_void: ...
@cc.lib_import(lib_path)
def c_blend_f32(O : cc.c_void_p, OH : cc.c_int32, OW : cc.c_int32, OC : cc.c_int32,
                A : cc.c_void_p,
                B : cc.c_float32_p, BH : cc.c_int32, BW : cc.c_int32, BC : cc.c_int32,
                M : cc.c_float32_p, MH : cc.c_int32, MW : cc.c_int32, MC : cc.c_int32,
                alpha : cc.c_float32) -> cc.c_void: ...

@cc.lib_import(lib_path)
def c_satushift_u8(O : cc.c_void_p, I : cc.c_void_p, size : cc.c_uint32) -> cc.c_void: ...
@cc.lib_import(lib_path)
def c_satushift_f32(O : cc.c_void_p, I : cc.c_void_p, size : cc.c_uint32) -> cc.c_void: ...


def setup_compile():
    ispc.compile_o(Path(__file__).parent / 'FImage_ispc.c')
    return cc.Compiler(Path(__file__).parent).shared_library(lib_path).minimal_c().optimize().include_glsl().include('FImage.cpp').include('FImage_ispc.o').compile()

#setup_compile()
