from __future__ import annotations

from typing import Iterable, Tuple, overload

import numpy as np

from ..collections import FDict
from .FVec import FVec

FVec3_like = Tuple[int|float, int|float, int|float]

class FVec3(FVec):
    """base class for 3 component vectors"""

    @overload
    def __init__(self, x : int|float, y : int|float, z : int|float): ...
    @overload
    def __init__(self, vec : FVec3): ...
    @overload
    def __init__(self, values : FVec3_like): ...
    @overload
    def __init__(self, values : Iterable[int|float]): ...
    @overload
    def __init__(self, values : np.ndarray):
        """from (3,) np.ndarray"""

    def __init__(self, *args):
        args_len = len(args)
        if args_len == 1:
            arg0 = args[0]
            if isinstance(arg0, FVec3):
                values = arg0._values
            elif isinstance(arg0, np.ndarray):
                values = arg0
            elif isinstance(arg0, Iterable):
                values = np.array(arg0, dtype=self._dtype)
            else:
                raise ValueError()
        elif args_len == 3:
            values = np.array(args, self._dtype)
        else:
            raise ValueError()

        if values.shape != (3,):
            raise ValueError()

        super().__init__(values)

    @property
    def x(self) -> int|float: return self._python_type(self._values[0])
    @property
    def y(self) -> int|float: return self._python_type(self._values[1])
    @property
    def z(self) -> int|float: return self._python_type(self._values[2])
    @property
    def xy(self) -> Tuple[int|float, int|float]: return (self.x,self.y)
    @property
    def xyz(self) -> Tuple[int|float, int|float, int|float]: return (self.x,self.y,self.z)




class FVec3f(FVec3):
    """Immutable float Vector 3D"""
    @staticmethod
    def from_state(state : FDict|None) -> FVec3f|None: return FVec3._from_state(state, FVec3f)
    @property
    def _dtype(self) -> np.dtype: return np.float32
    @property
    def _python_type(self): return float



class FVec3i(FVec3):
    """Immutable Int Vector 2D"""
    @staticmethod
    def from_state(state : FDict|None) -> FVec3i|None: return FVec3._from_state(state, FVec3i)
    @property
    def _dtype(self) -> np.dtype: return np.int32
    @property
    def _python_type(self): return int



