from __future__ import annotations

from typing import Self

import cv2
import numpy as np

from ..collections import FDict
from ..image import FImage
from ..math import FAffMat2, FVec2fArray, FVec2i
from .FAnno2D import FAnno2D
from .FAnnoPose import FAnnoPose


class FAnnoLmrk2D(FAnno2D):
    """base class for 2D landmarks annotations"""

    @staticmethod
    def from_state(state : FDict|None) -> FAnnoLmrk2D|None:
        state = FDict(state)
        if (lmrks := FVec2fArray.from_state(state.get('lmrks', None))) is not None:
            return FAnnoLmrk2D(lmrks)
        return None

    def __init__(self, lmrks : FVec2fArray):
        super().__init__()
        self._lmrks = lmrks

    def clone(self) -> Self:
        f = super().clone()
        f._lmrks = self._lmrks
        return f

    def get_state(self) -> FDict: return FDict({'lmrks' : self._lmrks.get_state(),})

    @property
    def lmrks(self) -> FVec2fArray: return self._lmrks

    def get_align_mat(self, coverage : float = 1.0, ux_offset = 0.0, uy_offset = 0.0, output_size = FVec2i(1,1) ) -> FAffMat2:
        """align mat to transform landmarks space to uniform space of standard coverage 1.0"""
        raise NotImplementedError()

    def get_pose(self) -> FAnnoPose:
        """roughly estimate"""
        lmrks_np = self._lmrks.as_np()
        return FAnnoPose( float(-lmrks_np[:,1].sum()), float(-lmrks_np[:,0].sum()), 0)

    def transform(self, mat : FAffMat2) -> Self:
        """tranforms using affine mat"""
        f = self.clone()
        f._lmrks = mat.map(f.lmrks)
        return f

    def draw(self, img : FImage, color, radius=1) -> FImage:
        """
        draw landmarks on the img

         color  tuple of values      should be the same as img color channels
        """
        img = img.HWC().copy()

        pts = self.lmrks.as_np().round().astype(np.int32)

        for i,(x, y) in enumerate(pts):
            cv2.circle(img, (x, y), radius, color)#, lineType=cv2.LINE_AA)
            #cv2.putText(img, str(i), (x,y), cv2.FONT_HERSHEY_PLAIN, 0.5, color)

        return FImage.from_numpy(img)

    def generate_mask(self, W, H) -> FImage:
        raise NotImplementedError()