from __future__ import annotations

from functools import cached_property
from typing import Iterable, Self, overload

import numpy as np

from ..collections import FDict
from .FVec3 import FVec3, FVec3_like, FVec3f
from .FVec3Array import FVec3Array, FVec3Array_like, FVec3fArray


class FAffMat3:
    """Immutable Affine 3D transformation matrix"""

    @staticmethod
    def from_state(state : FDict|None):
        state = FDict(state)
        if (values := state.get('values', None)) is not None:
            return FAffMat3(values)
        return None

    @overload
    @staticmethod
    def estimate(src : Iterable[FVec3_like], dst : Iterable[FVec3_like]) -> FAffMat3:
        """"""
    @staticmethod
    def estimate(*args, **kwargs) -> FAffMat3: raise

    @overload
    def __init__(self):
        """identity mat"""
    @overload
    def __init__(self, vec : FAffMat3): ...
    @overload
    def __init__(self, values : Iterable[int|float]):
        """from (12,) or (3,4) iterable values"""
    @overload
    def __init__(self, values : np.ndarray):
        """from (12,) or (3,4) np.ndarray"""

    def __init__(self, *args, **kwargs):
        args_len = len(args)
        if args_len == 0:
            values = np.array([[1,0,0,0],
                               [0,1,0,0],
                               [0,0,1,0], ], np.float32)
        elif args_len == 1:
            arg0 = args[0]
            if isinstance(arg0, FAffMat3):
                values = arg0._values
            elif isinstance(arg0, np.ndarray):
                values = arg0.astype(np.float32, copy=False).reshape(3,4)
            elif isinstance(arg0, Iterable):
                values = np.array(arg0, np.float32).reshape(3,4)

        if values.shape != (3,4):
            raise ValueError('wrong shape')

        values.setflags(write=False)
        self._values = values

    def clone(self) -> Self:
        f = self.__class__.__new__(self.__class__)
        f._values = self._values
        return f

    def get_state(self) -> FDict: return FDict({'values': self._values})

    @overload
    def translate(self, v : FVec3) -> Self: ...
    @overload
    def translate(self, tx : int|float, ty : int|float, tz : int|float) -> Self: ...
    def translate(self, *args) -> Self:
        if len(args) == 1:
            v = args[0]
            tx = v.x
            ty = v.y
            tz = v.z
        else:
            tx = args[0]
            ty = args[1]
            tz = args[2]

        return self*FAffMat3((1,0,0,tx,
                              0,1,0,ty,
                              0,0,1,tz,))

    @overload
    def scale(self, sv : FVec3) -> Self: ...
    @overload
    def scale(self, sx : float, sy : float = None, sz : float = None) -> Self: ...
    def scale(self, *args, **kwargs) -> Self:
        if len(args) == 1:
            v = args[0]
            if isinstance(v, FVec3):
                sx = v.x
                sy = v.y
                sz = v.z
            else:
                sx = v
                sy = None
                sz = None
        else:
            sx = args[0]
            sy = args[1]
            sz = args[1]

        return self*FAffMat3((sx,0,0,0,
                              0, sy if sy is not None else sx,0,0,
                              0, 0, sz if sz is not None else sx,0, ))

    @cached_property
    def inverted(self) -> Self:
        return FAffMat3(np.linalg.inv(self._values))

    def as_np(self) -> np.ndarray:
        """as (3,4) np.ndarray"""
        return self._values

    @overload
    def map(self, point : FVec3_like) -> FVec3f: ...
    @overload
    def map(self, points : FVec3Array_like) -> FVec3fArray: ...
    @overload
    def map(self, points : Iterable) -> FVec3fArray: ...
    @overload
    def map(self, points : np.ndarray) -> np.ndarray: ...

    def map(self, *args):
        arg0 = args[0]
        if isinstance(arg0, np.ndarray):
            return (self._values @ np.pad(arg0, ((0,0), (0,1)), constant_values=(1,), mode='constant').T).T.astype(np.float32, copy=False)
        elif isinstance(arg0, FVec3Array):
            return FVec3fArray(self.map(arg0.as_np()))
        elif isinstance(arg0, FVec3):
            return FVec3f(self.map([arg0])[0])
        elif isinstance(arg0, Iterable):
            return FVec3fArray(self.map(np.float32(arg0)))
        else:
            raise ValueError()


    def __hash__(self): return hash(self._values)
    def __eq__(self, other) -> bool:
        if self is other:
            return True
        if isinstance(other, FAffMat3):
            return np.all(self._values == other._values)
        return False

    def __mul__(self, other) -> FAffMat3:
        return FAffMat3( (np.concatenate([self._values, np.float32([[0,0,0,1]])], 0)
                          @
                          np.concatenate([other._values, np.float32([[0,0,0,1]])], 0))[:3] )


    def __repr__(self): return self.__str__()
    def __str__(self): return f"{self._values}"
