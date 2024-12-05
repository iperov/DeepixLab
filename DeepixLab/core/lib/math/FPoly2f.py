from __future__ import annotations

from functools import cached_property
from typing import Sequence, Tuple

import cv2
import numpy as np

from ..collections import FDict
from ..math import sd
from .FLine2f import FLine2f
from .FVec2 import FVec2f
from .FVec2Array import FVec2fArray


class FPoly2f(FVec2fArray):
    """Immutable polygon as array of 2D points"""

    @staticmethod
    def from_state(state : FDict|None) -> FPoly2f|None: return FVec2fArray._from_state(state, FPoly2f)

    @cached_property
    def points(self) -> Sequence[FVec2f]: return tuple(x for x in self)
    @cached_property
    def points_count(self) -> int: return len(self.points)

    @cached_property
    def edges(self) -> Sequence[FLine2f]:
        """edges including closing edge"""
        pts = self.points
        if len(pts) >= 3:
            pts = pts + (pts[0],)

        return tuple( FLine2f(p0, p1) for p0, p1 in zip(pts[:-1], pts[1:]) )

    @cached_property
    def edges_count(self) -> int: return len(self.edges)

    def dist(self, pt : FVec2f) -> float|None:
        if self.points_count >= 3:
            return float(-cv2.pointPolygonTest(self.as_np()[None,...], pt, True))
        return None

    def dists_to_edges(self, pt : FVec2f) -> Tuple[ Sequence[float], Sequence[FVec2f] ] | None:
        if self.points_count >= 2:
            dists, projs = sd.dist_to_edges(self.as_np(), pt.as_np(), is_closed=True)

            return dists.tolist(), [ FVec2f(*x) for x in projs]
        return None

    def nearest_edge_id_pt(self, pt) -> Tuple[int, FVec2f]|None:
        if (dists_projs := self.dists_to_edges(pt)) is not None:
            dists, projs = dists_projs
            if len(dists) > 0:
                edge_id = int(np.argmin(dists))
                return edge_id, projs[edge_id]
        return None

