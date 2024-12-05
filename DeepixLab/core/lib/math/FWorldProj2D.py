from __future__ import annotations

from functools import cached_property
from typing import Self

from ..collections import FDict
from .FAffMat2 import FAffMat2
from .FBBox import FBBox
from .FVec2 import FVec2f


class FWorldProj2D:
    """
    Immutable 2D world view projection
    """
    @staticmethod
    def from_state(state : FDict|None) -> FVec2f|None:
        state = FDict(state)
        if (vp_size := FVec2f.from_state(state.get('vp_size', None))) is not None and \
           (w_view_pos := FVec2f.from_state(state.get('w_view_pos', None))) is not None and \
           (scale := state.get('scale', None)) is not None:
            proj = FWorldProj2D()
            proj._vp_size = vp_size
            proj._w_view_pos = w_view_pos
            proj._scale = scale
            return proj
        return None

    def __init__(self):
        self._vp_size = FVec2f(320,240)
        self._w_view_pos = FVec2f(0,0)
        self._scale = 1.0

    def clone(self) -> Self:
        m = FWorldProj2D.__new__(self.__class__)
        m._vp_size = self._vp_size
        m._w_view_pos = self._w_view_pos
        m._scale = self._scale
        return m

    def get_state(self) -> FDict: return FDict({'vp_size': self._vp_size.get_state(),
                                                'w_view_pos': self._w_view_pos.get_state(),
                                                'scale': self._scale,
                                                  })

    @property
    def vp_size(self) -> FVec2f:
        """viewport size in origin coord space"""
        return self._vp_size
    def set_vp_size(self, vp_size : FVec2f) -> Self:
        self = self.clone()
        self._vp_size = vp_size
        return self


    @property
    def w_view_pos(self) -> FVec2f:
        """view pos in world space"""
        return self._w_view_pos
    def set_w_view_pos(self, w_view_pos : FVec2f) -> Self:
        self = self.clone()
        self._w_view_pos = w_view_pos
        return self

    @cached_property
    def vp_view_pos(self) -> FVec2f:
        """w_view_pos in viewport space"""
        return self.vp2w_mat.inverted.map(self._w_view_pos)
    def set_vp_view_pos(self, vp_view_pos : FVec2f) -> Self:
        """set world view pos from viewport coord space"""
        return self.set_w_view_pos(self.vp2w_mat.map(vp_view_pos))

    @property
    def scale(self) -> float: return self._scale
    def set_scale(self, scale : float) -> Self:
        """scale viewport projection. 1.0 is default"""
        self = self.clone()
        self._scale = scale
        return self

    @cached_property
    def vp2w_mat(self) -> FAffMat2:
        """projection matrix from viewport space to world space"""
        vp_size = self._vp_size
        w_view_pos = self._w_view_pos
        scale = self._scale
        return FAffMat2().translate(-vp_size.x/2, -vp_size.y/2).scale(scale).translate(w_view_pos.x, w_view_pos.y)

    @property
    def w2vp_mat(self) -> FAffMat2:
        """projection matrix from world to viewport space"""
        return self.vp2w_mat.inverted

    def center_fit(self, box : FBBox, coverage : float = 1.0) -> Self:
        """centers world FBBox in viewport"""
        vp_size = self.vp_size
        if vp_size.x != 0 and vp_size.y != 0:
            scale = max(box.width/vp_size.x, box.height/vp_size.y)*coverage
        else:
            scale = 1

        return self.set_scale(scale).set_w_view_pos(box.pc)

    def scale_at(self, vp_pt : FVec2f, scale_diff : float) -> Self:
        """scale projection by diff at specific viewport point."""
        mat = self.w2vp_mat.translate(-vp_pt.x,-vp_pt.y).scale(1/scale_diff).translate(vp_pt.x, vp_pt.y)
        new_world_pos = mat.inverted.map(self.vp_view_pos)
        return self.set_scale(self.scale*scale_diff).set_w_view_pos(new_world_pos)



    def __hash__(self): return hash((self._vp_size, self._w_view_pos, self._scale))
    def __eq__(self, other) -> bool:
        if self is other:
            return True
        if isinstance(other, FWorldProj2D):
            return  (self._vp_size == other._vp_size) & \
                    (self._w_view_pos == other._w_view_pos) & \
                    (self._scale == other._scale)

        return False

    def __repr__(self): return self.__str__()
    def __str__(self): return f"FWorldProj2D {self._vp_size} {self._w_view_pos} {self._scale}"