from __future__ import annotations

import math
from functools import cached_property
from typing import Self

from ..collections import FDict
from .FAffMat2 import FAffMat2
from .FVec2 import FVec2f


class FCirclef:
    """
    Immutable 2D Circle.

    Represented as position, radius.
    """
    @staticmethod
    def from_state(state : FDict|None) -> FCirclef|None:
        state = FDict(state)
        if (pos := FVec2f.from_state(state.get('pos', None))) is not None and \
           (radius := state.get('radius', None)) is not None:
            return FCirclef(pos, radius)
        return None

    def __init__(self, pos : FVec2f = FVec2f(0,0), radius : float = 1):
        self._pos = pos
        self._radius = radius

    def clone(self) -> Self:
        f = self.__class__.__new__(self.__class__)
        f._pos = self._pos
        f._radius = self._radius
        return f

    @property
    def pos(self) -> FVec2f: return self._pos
    @property
    def radius(self) -> float: return self._radius
    @property
    def area(self) -> float: return math.pi*self._radius**2
    @property
    def length(self) -> float: return 2*math.pi*self._radius

    @cached_property
    def mat(self) -> FAffMat2:
        """mat to transform FCirclef center space to origin space"""
        return FAffMat2().translate(self._pos)

    def get_state(self) -> FDict: return FDict({'pos': self._pos.get_state(),
                                                'radius': self._radius })

    def is_point_inside(self, pt : FVec2f) -> bool: return (self._pos-pt).length <= self._radius

    def is_point_on_edge(self, pt : FVec2f, inner_edge_width : float, outter_edge_width : float) -> float:
        l = (self._pos-pt).length
        r = self._radius
        return (l >= r-inner_edge_width) & (l <= r+outter_edge_width)

    def set_radius(self, radius : float) -> Self:
        self = self.clone()
        self._radius = radius
        return self

    def transform(self, mat : FAffMat2) -> Self:
        """Tranforms FCirclef using FAffMat2"""
        p0 = self._pos
        p1 = p0 + FVec2f(self._radius,0)
        p0, p1 = mat.map([p0,p1])
        return FCirclef(p0, (p1-p0).length)

    def __hash__(self): return hash((self._pos, self._radius))
    def __eq__(self, other) -> bool:
        if self is other:
            return True
        if isinstance(other, FCirclef):
            return (self._pos == other._pos) & (self._radius == other._radius)
        return False

    def __repr__(self): return self.__str__()
    def __str__(self): return f'FCirclef: {self._pos} {self._radius}'
