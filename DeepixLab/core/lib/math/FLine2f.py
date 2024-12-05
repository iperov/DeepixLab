from __future__ import annotations

from functools import cached_property
from typing import Iterable, Tuple, overload

import numpy as np

from ..collections import FDict
from .FLine import FLine
from .FNDArray import FNDArray
from .FVec2 import FVec2, FVec2_like, FVec2f

FLine2_like = Tuple[ FVec2_like, FVec2_like ]

class FLine2f(FLine):
    """
    Immutable 2D Line.
    """
    @staticmethod
    def from_state(state : FDict|None) -> FLine2f|None: return FNDArray._from_state(state, FLine2f)

    @overload
    def __init__(self, values : Tuple[FVec2_like, FVec2_like] ): ...
    @overload
    def __init__(self, p0 : FVec2, p1 : FVec2): ...
    @overload
    def __init__(self, x0 : float, y0 : float, x1 : float, y1 : float): ...
    @overload
    def __init__(self, values : np.ndarray):
        """from (2,2) array"""

    def __init__(self, *args, **kwargs):

        args_len = len(args)
        if args_len == 1:
            arg0 = args[0]
            if isinstance(arg0, np.ndarray):
                values = arg0.astype(np.float32, copy=False)
            elif isinstance(arg0, Iterable):
                values = np.array(arg0, dtype=np.float32)
            else:
                raise ValueError('wrong type')
        elif args_len == 2:
            values = np.array(args, dtype=np.float32)
        elif args_len == 4:
            values = np.array([ [args[0], args[1]],
                                [args[2], args[3]], ], dtype=np.float32)

        super().__init__(values)

    @property
    def _dtype(self) -> np.dtype: return np.float32
    @property
    def _python_type(self): return float

    @cached_property
    def p0(self) -> FVec2f: return FVec2f(self._values[0])
    @cached_property
    def p1(self) -> FVec2f: return FVec2f(self._values[1])
    @cached_property
    def pm(self) -> FVec2f:
        """mid point"""
        return self.get_line_pt(0.5)


    def get_line_pt(self, u : float) -> FVec2f:
        """u = [0.0 ... 1.0]"""
        p0p1 = self.p1 - self.p0
        return self.p0 + p0p1.normalize()*(p0p1.length*u)

    @cached_property
    def length(self) -> float: return (self.p1-self.p0).length


    def __getitem__(self, key : int) -> FVec2f: return FVec2f(*self._values[key])