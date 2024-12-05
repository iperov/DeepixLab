from __future__ import annotations

import cv2
import numpy as np

from ..collections import FDict
from ..image import FImage
from ..math import FAffMat2, FRectf, FVec2fArray, FVec2i
from .FAnnoLmrk2D import FAnnoLmrk2D


class FAnnoLmrk2D68(FAnnoLmrk2D):
    """Immutable class. Describes 2D 68 landmarks of the face"""

    @staticmethod
    def from_state(state : FDict|None) -> FAnnoLmrk2D68|None:
        if (anno := FAnnoLmrk2D.from_state(state)) is not None:
            return FAnnoLmrk2D68(anno.lmrks)
        return None

    def __init__(self, lmrks : FVec2fArray):
        if len(lmrks) != 68:
            raise ValueError('lmrks count must be == 68')
        super().__init__(lmrks)


    def get_align_mat(self, coverage : float = 1.0, ux_offset = 0.0, uy_offset = 0.0, output_size = FVec2i(1,1) ) -> FAffMat2:
        """
        Calculates Affine2D to translate landmarks space to uniform 0..1 space

        raise on error
        """
        # estimate landmarks transform from global space to local aligned space with bounds [0..1]
        lmrks = self._lmrks.as_np()

        # Base align using all landmarks. Has right rotation, but moving parts of the face seriously affect scale of align.
        #b_mat = FAffMat2.estimate(lmrks, uni_landmarks_2d_68)

        # Align that excludes highly moving parts, such as chin, mouth, but side faces can be rotated more than usual.
        mat = FAffMat2.estimate(lmrks[_non_moving_idxs], uni_landmarks_2d_68[_non_moving_idxs])

        # Get diff rotation between both alignments in global space
        #a_v0, a_v1 = (mat * FAffMat2().translate(-0.5,-0.5)).inverted.map([FVec2f(0,0), FVec2f(1,0)])
        #b_v0, b_v1 = (b_mat * FAffMat2().translate(-0.5,-0.5)).inverted.map([FVec2f(0,0), FVec2f(1,0)])

        #angle_diff = (a_v1 - a_v0).angle(b_v1 - b_v0)

        # Rotate final align at the center
        u_rect = FRectf(1,1).transform( FAffMat2()  .translate(-0.5,-0.5)
                                                    #.rotate(-angle_diff)
                                                    .scale(coverage)
                                                    .translate(0.5+ux_offset, 0.5+uy_offset) )
        g_rect = u_rect.transform( mat.inverted )
        return FAffMat2.estimate(g_rect, FRectf(output_size))


    def generate_mask(self, W, H) -> FImage:
        """"""
        out = np.zeros((H,W,1), dtype=np.float32)

        lmrks = self.lmrks.as_np().astype(np.int32)

        merged = lmrks[17:]
        cv2.fillConvexPoly(out, cv2.convexHull(merged), (1,) )

        return FImage.from_numpy(out)


_right_eyebrow = [17,18,19,20,21]
_left_eyebrow = [22,23,24,25,26]
_right_eye_corners = [36,39]
_left_eye_corners = [42,45]

_nose = [27,28,29,30]
_under_nose = [31,32,33,34,35]
_mouth_corners = [48,54]

_jaw_idxs = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

_non_moving_idxs = (_right_eyebrow + _left_eyebrow
                    + _nose
                    + _under_nose
                    + _right_eye_corners
                    + _left_eye_corners
                    )

uni_landmarks_2d_68 = np.array([
        [0.        , 0.17856914],  # 00
        [0.00412831, 0.31259227],  # 01
        [0.0196793 , 0.44770938],  # 02
        [0.04809872, 0.5800727 ],  # 03
        [0.10028344, 0.70349526],  # 04
        [0.17999782, 0.81208664],  # 05
        [0.27627307, 0.90467805],  # 06
        [0.38463727, 0.98006284],  # 07
        [0.5073561 , 1.        ],  # 08
        [0.61536276, 0.98006284],  # 09
        [0.7237269 , 0.90467805],  # 10
        [0.8200022 , 0.81208664],  # 11
        [0.89971656, 0.70349526],  # 12
        [0.95190126, 0.5800727 ],  # 13
        [0.9803207 , 0.44770938],  # 14
        [0.99587166, 0.31259227],  # 15
        [1.        , 0.17856914],  # 16

        [0.09485531, 0.06025707],  # 17
        [0.15534875, 0.01191543],  # 18
        [0.2377474 , 0.        ],  # 19
        [0.32313403, 0.01543965],  # 20
        [0.4036699 , 0.0521157],   # 21

        [0.56864655, 0.0521157 ],  # 22
        [0.65128165, 0.01543965],  # 23
        [0.7379608 , 0.        ],  # 24
        [0.82290924, 0.01191543],  # 25
        [0.88739765, 0.06025707],  # 26

        [0.5073561, 0.15513189],  # 27
        [0.5073561, 0.24343018],  # 28
        [0.5073561, 0.33176517],  # 29
        [0.5073561, 0.422107  ],  # 30

        [0.397399  , 0.48004663],  # 31
        [0.4442625 , 0.49906778],  # 32
        [0.5073561 , 0.5144414 ],  # 33
        [0.54558265, 0.49906778],  # 34
        [0.59175086, 0.48004663],  # 35

        [0.194157  , 0.17277813],  # 36
        [0.24600308, 0.12988105],  # 37
        [0.31000495, 0.12790850],  # 38
        [0.36378494, 0.15817115],  # 39
        [0.3063696 , 0.18155812],  # 40
        [0.24390514, 0.18370388],  # 41

        [0.6189632 , 0.17277813],  # 42
        [0.67249435, 0.12988105],  # 43
        [0.7362857 , 0.12790850],  # 44
        [0.7888591 , 0.15817115],  # 45
        [0.74115133, 0.18155812],  # 46
        [0.6791372 , 0.18370388],  # 47

        [0.30711025, 0.6418497 ],  # 48
        [0.3759703 , 0.6109595 ],  # 49
        [0.44670257, 0.5970508 ],  # 50
        [0.5073561,  0.60872644],  # 51
        [0.5500201 , 0.5970508 ],  # 52
        [0.6233016 , 0.6109595 ],  # 53
        [0.69541407, 0.6418497 ],  # 54
        [0.628068  , 0.7145425],   # 55
        [0.5573954 , 0.74580276],  # 56
        [0.5073561,  0.7505844 ],  # 57
        [0.44528747, 0.74580276],  # 58
        [0.37508208, 0.7145425 ],  # 59
        [0.3372878 , 0.64616466],  # 60
        [0.44701463, 0.64064664],  # 61
        [0.5073561,  0.6449633 ],  # 62
        [0.5513943 , 0.64064664],  # 63
        [0.6650228 , 0.64616466],  # 64
        [0.5530556 , 0.6786047],   # 65
        [0.5073561 , 0.68417645],  # 66
        [0.44657204, 0.6786047 ]], # 67
        dtype=np.float32)
