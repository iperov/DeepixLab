from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Self, TypeVar, overload

import numpy as np

from ..collections import FDict
from .FArray import FArray
from .FNDArrayArif import FNDArrayArif
from .FVec3 import FVec3_like, FVec3f, FVec3i

if TYPE_CHECKING:
    from .FAffMat2 import FAffMat2

FVec3Array_like = Iterable[FVec3_like]

T = TypeVar('T')
class FVec3Array(FArray[T], FNDArrayArif):
    """Immutable array of 3D points"""

    @overload
    def __init__(self): ...
    @overload
    def __init__(self, values : Iterable[FVec3_like] ): ...

    def __init__(self, *args, **kwargs):
        args_len = len(args)
        if args_len == 0:
            values = np.ndarray( (0, 2), self._dtype )
        elif args_len == 1:
            arg0 = args[0]
            if isinstance(arg0, FVec3Array):
                values = arg0._values
            elif isinstance(arg0, np.ndarray):
                values = arg0
            elif isinstance(arg0, Iterable):
                values = np.array(arg0, dtype=self._dtype)
            else:
                raise ValueError()

        super().__init__(values)

    def transform(self, mat : FAffMat2) -> Self:
        self = self.clone()
        self._values = mat.map(self._values)
        return self

class FVec3fArray(FVec3Array[FVec3f]):
    """Immutable float array of 3D points"""

    @staticmethod
    def from_state(state : FDict|None) -> FVec3fArray|None: return FVec3Array._from_state(state, FVec3fArray)
    @property
    def _dtype(self) -> np.dtype: return np.float32
    @property
    def _python_type(self): return float
    @property
    def _elem_type(self): return FVec3f


class FVec3iArray(FVec3Array[FVec3i]):
    """Immutable int array of 3D points"""

    @staticmethod
    def from_state(state : FDict|None) -> FVec3iArray|None: return FVec3Array._from_state(state, FVec3iArray)
    @property
    def _dtype(self) -> np.dtype: return np.int32
    @property
    def _python_type(self): return int
    @property
    def _elem_type(self): return FVec3i

