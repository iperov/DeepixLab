from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Self, TypeVar, overload

import numpy as np

from ..collections import FDict
from .FArray import FArray
from .FNDArrayArif import FNDArrayArif
from .FVec2 import FVec2_like, FVec2f, FVec2i

if TYPE_CHECKING:
    from .FAffMat2 import FAffMat2

FVec2Array_like = Iterable[FVec2_like]

T = TypeVar('T')
class FVec2Array(FArray[T], FNDArrayArif):
    """Immutable array of 2D points"""

    @overload
    def __init__(self): ...
    @overload
    def __init__(self, values : Iterable[FVec2_like] ): ...

    def __init__(self, *args, **kwargs):
        args_len = len(args)
        if args_len == 0:
            values = np.ndarray( (0, 2), self._dtype )
        elif args_len == 1:
            arg0 = args[0]
            if isinstance(arg0, FVec2Array):
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

class FVec2fArray(FVec2Array[FVec2f]):
    """Immutable float array of 2D points"""

    @staticmethod
    def from_state(state : FDict|None) -> FVec2fArray|None: return FVec2Array._from_state(state, FVec2fArray)
    @property
    def _dtype(self) -> np.dtype: return np.float32
    @property
    def _python_type(self): return float
    @property
    def _elem_type(self): return FVec2f


class FVec2iArray(FVec2Array[FVec2i]):
    """Immutable int array of 2D points"""

    @staticmethod
    def from_state(state : FDict|None) -> FVec2iArray|None: return FVec2Array._from_state(state, FVec2iArray)
    @property
    def _dtype(self) -> np.dtype: return np.int32
    @property
    def _python_type(self): return int
    @property
    def _elem_type(self): return FVec2i

