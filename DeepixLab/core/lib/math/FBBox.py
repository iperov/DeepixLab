from __future__ import annotations

from functools import cached_property
from typing import Generic, Iterable, Iterator, Self, Tuple, TypeVar, overload

import numpy as np

from ..collections import FDict
from .FAffMat2 import FAffMat2
from .FNDArray import FNDArray
from .FVec2 import FVec2_like, FVec2f, FVec2i

T = TypeVar('T')
class FBBox(FNDArray, Generic[T], Iterable):
    """base class for FBBox"""

    @overload
    def __init__(self):
        """box in 0,0 with zero size"""
    @overload
    def __init__(self, box : FBBox): ...
    @overload
    def __init__(self, points : Iterable[FVec2_like]):
        """compute bbox of multiple FVec2 or ndarray (N,2)"""
    @overload
    def __init__(self, lt : FVec2_like, size : FVec2_like):
        """pos `lt` with `size` size"""
    @overload
    def __init__(self, l : int|float, t : int|float, r : int|float, b : int|float):
        """
        from l,t,r,b bbox
           t
         l-|-r
           b
        """

    def __init__(self, *args):
        args_len = len(args)
        if args_len == 0:
            values = np.array([0,0,0,0], dtype=self._dtype)
        elif args_len == 1:
            arg0 = args[0]
            if isinstance(arg0, FBBox):
                values = arg0._values
            elif isinstance(arg0, Iterable):
                if not isinstance(arg0, np.ndarray):
                    arg0 = np.array(arg0, dtype=self._dtype)
                values = np.array([arg0[:,0].min(),
                                   arg0[:,1].min(),
                                   arg0[:,0].max(),
                                   arg0[:,1].max()], dtype=self._dtype)
            else:
                raise ValueError()
        elif args_len == 2:
            (l,t), (w,h) = args
            values = np.array([l,t,l+w,t+h], self._dtype)
        elif args_len == 4:
            values = np.array(args, self._dtype)

        if values.shape != (4,):
            raise ValueError()

        super().__init__(values)

    @property
    def _T_type(self): raise NotImplementedError()

    @property
    def x(self) -> int|float: return self._python_type(self._values[0])
    @property
    def y(self) -> int|float: return self._python_type(self._values[1])
    @cached_property
    def width(self) -> int|float: return self._python_type(self._values[2]-self._values[0])
    @cached_property
    def height(self) -> int|float: return self._python_type(self._values[3]-self._values[1])

    @cached_property
    def pos(self) -> T: return self._T_type(self._values[0:2])
    @cached_property
    def size(self) -> T: return self._T_type(self.width, self.height)
    @cached_property
    def area(self) -> float: return self.width*self.height
    @cached_property
    def xy(self) -> T: return self.pos
    @cached_property
    def wh(self) -> T: return self.size

    @property
    def lt(self) -> T: return self.pos
    @property
    def rb(self) -> T: return self._T_type(self._values[2:4])

    @property
    def p0(self) -> T:
        """```
         0--x
         |  |
         x--x
        ```"""
        return self.as_4pts()[0]
    @property
    def p1(self) -> T:
        """```
         x--1
         |  |
         x--x
        ```"""
        return self.as_4pts()[1]
    @property
    def p2(self) -> T:
        """```
         x--x
         |  |
         x--2
        ```"""
        return self.as_4pts()[2]
    @property
    def p3(self) -> T:
        """```
         x--x
         |  |
         3--x
        ```"""
        return self.as_4pts()[3]

    @cached_property
    def pc(self) -> T:
        """center point"""
        return self.pos + self.size/2

    def as_4pts(self) -> Tuple[FVec2f, FVec2f, FVec2f, FVec2f]:
        """```
        get rect as 4 pts

         0--1
         |  |
         3--2
        ```"""
        pos = self.pos
        size = self.size
        T_type = self._T_type

        return pos, T_type(pos.x+size.x, pos.y), T_type(pos.x+size.x, pos.y+size.y), T_type(pos.x, pos.y+size.y)

    # Iterable
    def __iter__(self) -> Iterator[T]:
        for i in range(len(self)):
            yield self[i]

    @overload
    def __getitem__(self, key : int) -> T: ...
    @overload
    def __getitem__(self, key : Iterable|slice) -> Self: ...
    def __getitem__(self, key):
        result = self._values[key]
        if isinstance(key, int):
            return self._python_type(result)
        else:
            raise ValueError()

    def transform(self, mat : FAffMat2) -> Self:
        """
        Tranforms FBBox using FAffMat2

            mat     FAffMat2

        result is bounding box of transformed box
        """

        pts = mat.map(self.as_4pts())

        pts = pts.as_np()
        self = self.clone()
        self._values = np.array((pts[:,0].min(), pts[:,1].min(), pts[:,0].max(), pts[:,1].max()), self._dtype)
        return self




class FBBoxf(FBBox[FVec2f]):
    """Immutable float non rotatable bounding box"""
    @staticmethod
    def from_state(state : FDict|None) -> FBBoxf|None: return FBBox._from_state(state, FBBoxf)
    @property
    def _dtype(self) -> np.dtype: return np.float32
    @property
    def _python_type(self): return float
    @property
    def _T_type(self): return FVec2f


class FBBoxi(FBBox[FVec2i]):
    """Immutable int non rotatable bounding box"""
    @staticmethod
    def from_state(state : FDict|None) -> FBBoxi|None: return FBBox._from_state(state, FBBoxi)
    @property
    def _dtype(self) -> np.dtype: return np.int32
    @property
    def _python_type(self): return int
    @property
    def _T_type(self): return FVec2i