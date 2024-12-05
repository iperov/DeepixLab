from __future__ import annotations

import math
from typing import Iterable, Self, Tuple, overload

import numpy as np

from ..collections import FDict
from .FVec import FVec

FVec2_like = Tuple[int|float, int|float]

class FVec2(FVec):
    """base class for 2 component vectors"""

    @overload
    def __init__(self, x : int|float, y : int|float): ...
    @overload
    def __init__(self, vec : FVec2): ...
    @overload
    def __init__(self, values : FVec2_like): ...
    @overload
    def __init__(self, values : Iterable[int|float]): ...
    @overload
    def __init__(self, values : np.ndarray):
        """from (2,) np.ndarray"""

    def __init__(self, *args):
        args_len = len(args)
        if args_len == 1:
            arg0 = args[0]
            if isinstance(arg0, FVec2):
                values = arg0._values
            elif isinstance(arg0, np.ndarray):
                values = arg0
            elif isinstance(arg0, Iterable):
                values = np.array(arg0, dtype=self._dtype)
            else:
                raise ValueError()
        elif args_len == 2:
            values = np.array(args, self._dtype)
        else:
            raise ValueError()

        if values.shape != (2,):
            raise ValueError()

        super().__init__(values)

    @property
    def x(self) -> int|float: return self._python_type(self._values[0])
    @property
    def y(self) -> int|float: return self._python_type(self._values[1])
    @property
    def xy(self) -> Tuple[int|float, int|float]: return (self.x,self.y)
    @property
    def yx(self) -> Tuple[int|float, int|float]: return (self.y,self.x)

    @property
    def cross(self) -> Self:
        self = self.clone()
        self._values = np.array([self._values[1], -self._values[0]], self._dtype)
        return self

    def atan2(self) -> float: return math.atan2(self._values[1], self._values[0])
    def angle(self, other : FVec2) -> float: return math.atan2(self.x * other.y - self.y*other.x, self.x*other.x + self.y*other.y )

    def rotate_around(self, pivot : FVec2, angle : float) -> Self:
        s = math.sin(angle)
        c = math.cos(angle)
        x = self.x - pivot.x
        y = self.y - pivot.y

        self = self.clone()
        self._values = np.array([x*c - y*s + pivot.x, x*s + y*c + pivot.y], self._dtype)
        return self

    def rotate_around_deg(self, pivot : FVec2, angle_deg : float) -> Self:
        return self.rotate_around(pivot, angle_deg*math.pi/180)




class FVec2f(FVec2):
    """Immutable float Vector 2D"""
    @staticmethod
    def from_state(state : FDict|None) -> FVec2f|None: return FVec2._from_state(state, FVec2f)
    @property
    def _dtype(self) -> np.dtype: return np.float32
    @property
    def _python_type(self): return float



class FVec2i(FVec2):
    """Immutable Int Vector 2D"""
    @staticmethod
    def from_state(state : FDict|None) -> FVec2i|None: return FVec2._from_state(state, FVec2i)
    @property
    def _dtype(self) -> np.dtype: return np.int32
    @property
    def _python_type(self): return int



