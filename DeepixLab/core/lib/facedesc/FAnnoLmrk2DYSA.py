from __future__ import annotations

import cv2
import numpy as np

from ..collections import FDict
from ..image import FImage
from ..math import FAffMat2, FRectf, FVec2fArray, FVec2i
from .FAnnoLmrk2D import FAnnoLmrk2D


class FAnnoLmrk2DYSA(FAnnoLmrk2D):
    """Immutable class. Describes two 2D landmarks of the face projected from Yaw Sphere Axis"""

    @staticmethod
    def from_state(state : FDict|None) -> FAnnoLmrk2DYSA|None:
        if (anno := FAnnoLmrk2D.from_state(state)) is not None:
            return FAnnoLmrk2DYSA(anno.lmrks)
        return None

    def __init__(self, lmrks : FVec2fArray):
        if len(lmrks) != 2:
            raise ValueError('lmrks count must be == 2')
        super().__init__(lmrks)

    def get_align_mat(self, coverage : float = 1, ux_offset = 0.0, uy_offset = 0.0, output_size = FVec2i(1,1) ) -> FAffMat2:
        """Calculates Affine2D to translate landmarks space to uniform 0..1 space"""
        mat = FAffMat2.estimate(self._lmrks, uni_landmarks_L2V)

        u_rect = FRectf(1,1).transform( FAffMat2()  .translate(-0.5, -0.5)
                                                    .scale(coverage)
                                                    .translate(0.5+ux_offset, 0.5+uy_offset) )
        g_rect = u_rect.transform( mat.inverted )
        return FAffMat2.estimate(g_rect, FRectf(output_size))

    def draw(self, img : FImage, color, radius=1) -> FImage:
        img = img.HWC().copy()

        pts = self.lmrks.as_np().round().astype(np.int32)
        cv2.line(img, pts[0], pts[1], color, thickness=radius)

        return FImage.from_numpy(img)

uni_landmarks_L2V = np.array([
    [0.5, 0.0],
    [0.5, 1.0],
    ], dtype=np.float32)
