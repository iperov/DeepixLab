from __future__ import annotations

from ..collections import FDict
from ..math import FVec2fArray
from .FAnnoLmrk2D import FAnnoLmrk2D
from .FAnnoLmrk2DYSA import FAnnoLmrk2DYSA


class FAnnoLmrk2DYSARange(FAnnoLmrk2D):
    """Immutable class. Allows to get FAnnoLmrk2DYSA from specific offset"""

    @staticmethod
    def from_state(state : FDict|None) -> FAnnoLmrk2DYSARange|None:
        if (anno := FAnnoLmrk2D.from_state(state)) is not None:
            return FAnnoLmrk2DYSARange(anno.lmrks)
        return None

    def __init__(self, lmrks : FVec2fArray):
        if len(lmrks) != 4:
            raise ValueError('lmrks count must be == 4')
        super().__init__(lmrks)

    def to_2DYSA(self, y_axis_offset : float = 0.0) -> FAnnoLmrk2DYSA:
        p0u, p0d, p1u, p1d = self.lmrks

        p01u = p1u - p0u
        p01d = p1d - p0d

        return FAnnoLmrk2DYSA(FVec2fArray([p0u + p01u.normalize() * p01u.length*y_axis_offset,
                                           p0d + p01d.normalize() * p01d.length*y_axis_offset]))