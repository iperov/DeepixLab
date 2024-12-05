from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ... import cc
from ...functools import cached_method
from ...math import FAffMat2
from ..FImage import FImage


class TransformParams:
    def __init__(self,  tx : float = 0.0,
                        ty : float = 0.0,
                        scale : float = 0,
                        rot_deg : float = 0.0,
                        ):
        """"""
        self._tx = tx
        self._ty = ty
        self._scale = scale
        self._rot_deg = rot_deg

    @property
    def _affine_scale(self) -> float:
        return (1 / (1 - scale)) if (scale := self._scale) < 0 else 1 + scale


    def copy(self) -> TransformParams:
        return TransformParams( tx = self._tx,
                                ty = self._ty,
                                scale = self._scale,
                                rot_deg = self._rot_deg)

    @staticmethod
    def generate(   tx_var : float = 0.3,
                    ty_var : float = 0.3,
                    scale_var : float = 0.4,
                    rot_deg_var : float = 15.0,
                    seed : int|None = None,) -> TransformParams:
        rnd_state = np.random.RandomState(seed)

        return TransformParams( tx = rnd_state.uniform(-tx_var, tx_var),
                                ty = rnd_state.uniform(-ty_var, ty_var),
                                scale = rnd_state.uniform(-scale_var, scale_var),
                                rot_deg = rnd_state.uniform(-rot_deg_var, rot_deg_var) )

    def added(self, tx : float = 0.0,
                    ty : float = 0.0,
                    scale : float = 0,
                    rot_deg : float = 0.0,) -> TransformParams:
        return TransformParams( tx = self._tx + tx,
                                ty = self._ty + ty,
                                scale = self._scale + scale,
                                rot_deg = self._rot_deg + rot_deg)

    def scaled(self, value : float) -> TransformParams:
        return TransformParams( tx = self._tx*value,
                                ty = self._ty*value,
                                scale = self._scale*value,
                                rot_deg = self._rot_deg*value)

    def __add__(self, value : TransformParams) -> TransformParams:
        return  self.added( tx=value._tx,
                            ty=value._ty,
                            scale=value._scale,
                            rot_deg=value._rot_deg )

class Geo:
    """Max quality one-pass image augmentation using geometric transformations."""

    def __init__(self,  offset_transform_params : TransformParams|None = None,
                        transform_params : TransformParams|None = None,
                        deform_transform_params : TransformParams|None = None,
                        seed : int|None = None,
                ):
        rnd_state = np.random.RandomState(seed)

        if offset_transform_params is None:
            offset_transform_params = TransformParams()
        if transform_params is None:
            transform_params = TransformParams()

        if deform_transform_params is None:
            deform_transform_params = TransformParams.generate( tx_var = 0.0,
                                                                ty_var = 0.0,
                                                                scale_var = 0.0,
                                                                rot_deg_var = 0.0,
                                                                seed=rnd_state.randint(2**31))


        self._offset_transform_params = offset_transform_params
        self._transform_params = transform_params
        self._deform_transform_params = deform_transform_params
        self._deform_grid_cell_count = rnd_state.randint(5,10)
        self._seed = rnd_state.randint(2**31)

    def transform(self, img : FImage,
                        OW : int|None = None,
                        OH : int|None = None,
                        center_fit = True,
                        transform_intensity : float = 1.0,
                        deform_intensity : float = 1.0,
                        interp : FImage.Interp = FImage.Interp.LANCZOS4,
                        border : FImage.Border = FImage.Border.CONSTANT,
                        ) -> FImage:
        """
        transform an image.

        Subsequent calls will output the same result for any img shape and out_res.

        """
        H,W,_ = img.shape

        if OW is None:
            OW = W
        if OH is None:
            OH = H

        rnd_state = np.random.RandomState(self._seed)

        remap_grid, mask = self._gen_remap_grid(W, H, OW, OH,
                                            center_fit=center_fit,
                                            tr_params=self._offset_transform_params + self._transform_params.scaled(transform_intensity),
                                            deform_tr_params=self._deform_transform_params,
                                            deform_cell_count=self._deform_grid_cell_count,
                                            deform_intensity=deform_intensity,
                                            border=border,
                                            seed=rnd_state.randint(2**31),
                                            )

        img = img.remap(remap_grid, interp=interp, border=border).HWC()

        if border == FImage.Border.CONSTANT:
            img *= mask

        return FImage.from_numpy(img).clip()

    @cached_method
    def _get_cached(self,   SW, SH, TW, TH,
                    center_fit : bool,
                    tr_params : TransformParams,
                    deform_tr_params : TransformParams,
                    deform_cell_count : int,
                    deform_intensity : float,
                    grid_seed : int,
                    ) :
        # Make mat to transform source space to target space
        s2t_mat = (FAffMat2()   .translate( (-0.5+tr_params._tx)*SW, (-0.5+tr_params._ty)*SH )
                                .rotate_deg(tr_params._rot_deg)
                                .scale(tr_params._affine_scale)
                                .scale( ( ( TW if SW < SH else TH ) / min(SW,SH) ) if center_fit else 1.0 )
                                .translate( 0.5*TW, 0.5*TH )     )

        # Make identity remap coord grid in source space
        s_remap_grid = np.stack(np.meshgrid(np.arange(SW, dtype=np.float32),
                                            np.arange(SH, dtype=np.float32), copy=False), -1)

        if deform_intensity != 0.0:
            # Apply random deformations of target space in s_remap_grid

            # Make random coord shifts in target space
            # t_deform_grid = _gen_rw_coord_diff_grid(TW, TH, deform_cell_count, rnd_state)
            # # Scale with 0..1 intensity
            # t_deform_grid *= deform_intensity
            # # Merge with identity
            # t_deform_grid += np.stack(np.meshgrid(np.arange(TW, dtype=np.float32),
            #                                       np.arange(TH, dtype=np.float32), ), -1)

            t_deform_grid = gen_grid(TW, TH, deform_cell_count, deform_intensity, seed=grid_seed ) #


            # Make transform mat for deform_grid from deform_tr_params.
            # this mat translates deform_grid in target space
            t2t_deform_mat = (FAffMat2().translate( (-0.5+deform_tr_params._tx)*TW, (-0.5+deform_tr_params._ty)*TH )
                                        .rotate_deg(deform_tr_params._rot_deg)
                                        .scale(deform_tr_params._affine_scale)
                                        .translate(0.5*TW, 0.5*TH)    )

            s2t_mat_t2t_deform_mat = s2t_mat*t2t_deform_mat

            # Warp border-reflected s_remap_grid to target space with s2t_mat+t2t_deform_mat
            t_remap_grid = cv2.warpAffine(s_remap_grid, s2t_mat_t2t_deform_mat.as_np(), (TW,TH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

            # Remap t_remap_grid with t_deform_grid and get diffs
            t_diff_deform_grid = t_remap_grid - cv2.remap(t_remap_grid, t_deform_grid, None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

            # Fade to zero at borders
            w_border_size = TW // deform_cell_count
            w_dumper = np.linspace(0, 1, w_border_size, dtype=np.float32)
            t_diff_deform_grid[:,:w_border_size ,:] *= w_dumper[None,:,None]
            t_diff_deform_grid[:,-w_border_size:,:] *= w_dumper[None,::-1,None]

            h_border_size = TH // deform_cell_count
            h_dumper = np.linspace(0, 1, h_border_size, dtype=np.float32)
            t_diff_deform_grid[:h_border_size, :,:] *= h_dumper[:,None,None]
            t_diff_deform_grid[-h_border_size:,:,:] *= h_dumper[::-1,None,None]

            # Warp t_diff_deform_grid to source space. BORDER_CONSTANT ensures zero coord diff outside.
            s_diff_deform_grid = cv2.warpAffine(t_diff_deform_grid, s2t_mat_t2t_deform_mat.inverted.as_np(), (SW,SH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

            # Merge s_remap_grid with s_diff_deform_grid
            s_remap_grid += s_diff_deform_grid

        # make binary mask to refine image-boundary
        mask = cv2.warpAffine( np.ones((SH,SW), dtype=np.uint8), s2t_mat.as_np(), (TW,TH), flags=cv2.INTER_NEAREST)[...,None]

        return s2t_mat, s_remap_grid, mask

    def _gen_remap_grid(self,   SW, SH, TW, TH,
                                center_fit : bool,
                                tr_params : TransformParams,
                                deform_tr_params : TransformParams,
                                deform_cell_count : int,
                                deform_intensity : float,
                                border,
                                seed : int|None,
                                ):
        rnd_state = np.random.RandomState(seed)

        s2t_mat, s_remap_grid, mask = self._get_cached(SW, SH, TW, TH, center_fit, tr_params, deform_tr_params, deform_cell_count, deform_intensity, rnd_state.randint(2**31))

        # Warp s_remap_grid to target space using s2t_mat
        t_remap_grid = cv2.warpAffine(s_remap_grid, s2t_mat.as_np(), (TW,TH), flags=cv2.INTER_LINEAR,
                                      borderMode=FImage._border_to_cv(FImage.Border.REPLICATE if border == FImage.Border.CONSTANT else border))

        return t_remap_grid, mask


lib_path = cc.get_lib_path(Path(__file__).parent, Path(__file__).stem)

@cc.lib_import(lib_path)
def c_gen_grid(out_grid : cc.c_float32_p, OW : cc.c_int32, OH : cc.c_int32, cell_count : cc.c_int32, intensity : cc.c_float32, seed : cc.c_uint32) -> cc.c_void: ...
def gen_grid(W : int, H : int, cell_count : int, intensity : float, seed : int) -> np.ndarray:
    """bilinear interpolation with align corners."""
    out_grid = np.empty( (H,W,2), np.float32)
    c_gen_grid( out_grid.ctypes.data_as(cc.c_float32_p), W, H, cell_count, intensity, seed)
    return out_grid


def setup_compile():
    return cc.Compiler(Path(__file__).parent).shared_library(lib_path).minimal_c().optimize().include_glsl().include('Geo.cpp').compile()

#setup_compile()