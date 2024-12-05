from __future__ import annotations

import math
from functools import cached_property
from typing import Self, Tuple

from ..collections import FDict
from .FAffMat2 import FAffMat2
from .FVec2 import FVec2f


class FEllipsef:
    """
    Immutable 2D ellipse.

    Represented as position, rotation, x-radius, y-radius
    """

    @staticmethod
    def from_state(state : FDict|None) -> FEllipsef|None:
        state = FDict(state)
        if (pos := FVec2f.from_state(state.get('pos', None))) is not None and \
           (x_radius := state.get('x_radius', None)) is not None and \
           (y_radius := state.get('y_radius', None)) is not None and \
           (rotation := state.get('rotation', None)) is not None:
            return FEllipsef(pos, x_radius, y_radius, rotation)
        return None

    @staticmethod
    def from_3pts(p0 : FVec2f, p1 : FVec2f, p2 : FVec2f) -> FEllipsef:
        """
        Construct FEllipsef from 3 pts
            0 -- 1  (x_rad)
            |
            2 (y_rad)
        """
        x_radius = (p1-p0).length
        y_radius = (p2-p0).length
        rotation = math.atan2(p1.y-p0.y, p1.x-p0.x)
        return FEllipsef(p0, x_radius, y_radius, rotation)

    def __init__(self, pos : FVec2f = FVec2f(0,0), x_radius : float = 1, y_radius : float = 1, rotation : float = 0):
        self._pos = pos
        self._x_radius = x_radius
        self._y_radius = y_radius
        self._rotation = rotation

    def clone(self) -> Self:
        f = self.__class__.__new__(self.__class__)
        f._pos = self._pos
        f._x_radius = self._x_radius
        f._y_radius = self._y_radius
        f._rotation = self._rotation
        return f

    @property
    def pos(self) -> FVec2f: return self._pos
    @property
    def rotation(self) -> float: return self._rotation
    @property
    def x_radius(self) -> float: return self._x_radius
    @property
    def y_radius(self) -> float: return self._y_radius

    @cached_property
    def mat(self) -> FAffMat2:
        """mat to transform FEllipsef center space to origin space"""
        return FAffMat2().translate(self._pos)

    def get_state(self) -> FDict: return FDict({'pos': self._pos.get_state(),
                                                'rotation': self._rotation,
                                                'x_radius': self._x_radius,
                                                'y_radius': self._y_radius, })

    def set_pos(self, pos : FVec2f) -> Self:
        self = self.clone()
        self._pos = pos
        return self

    def set_x_radius(self, x_radius : float) -> Self:
        self = self.clone()
        self._x_radius = x_radius
        return self

    def set_y_radius(self, y_radius : float) -> Self:
        self = self.clone()
        self._y_radius = y_radius
        return self

    def set_rotation(self, rotation : float) -> Self:
        self = self.clone()
        self._rotation = rotation
        return self

    def set_rotation_deg(self, rotation_deg : float) -> Self:
        self = self.clone()
        self._rotation = rotation_deg*math.pi/180
        return self

    def translate(self, diff : FVec2f) -> Self:
        """translate FEllipsef by diff"""
        self = self.clone()
        self._pos = self._pos + diff
        return self

    def as_3pts(self) -> Tuple[FVec2f, FVec2f, FVec2f]:
        """```
        get ellipse as 3 pts

            0 -- 1  (x_rad)
            |
            2 (y_rad)

        ```"""
        p0x=self._pos.x; p0y=self._pos.y; r=self._rotation; xr=self._x_radius; yr=self._y_radius

        xv = math.cos(r)
        yv = math.sin(r)

        p1x = p0x+xv*xr
        p1y = p0y+yv*xr

        p2x = p0x-yv*yr
        p2y = p0y+xv*yr

        return self._pos, FVec2f(p1x, p1y), FVec2f(p2x, p2y)


    def transform(self, mat : FAffMat2) -> Self:
        """Tranforms FEllipsef using FAffMat2"""
        return FEllipsef.from_3pts( *mat.map(self.as_3pts()) )

    def __hash__(self): return hash((self._pos, self._width, self._height, self._rotation))
    def __eq__(self, other) -> bool:
        if self is other:
            return True
        if isinstance(other, FEllipsef):
            return (self._pos == other._pos) & (self._x_radius == other._x_radius) & (self._y_radius == other._y_radius) & (self._rotation == other._rotation)
        return False

    def __repr__(self): return self.__str__()
    def __str__(self): return f'FEllipsef: {self._pos} {self._rotation} {self._x_radius} {self._y_radius} {self._rotation}'
