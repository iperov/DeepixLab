from __future__ import annotations

from functools import cached_property
from typing import Self, overload

from ..collections import FDict
from ..functools import cached_method
from .FAffMat2 import FAffMat2
from .FBBox import FBBoxf
from .FPoly2f import FPoly2f
from .FVec2 import FVec2, FVec2f
from .FVec2Array import FVec2Array


class FRectf:
    """
    Immutable 2D Rectangle.

    """

    @staticmethod
    def from_state(state : FDict|None) -> FRectf|None:
        state = FDict(state)
        if (mat := FAffMat2.from_state(state.get('mat', None))) is not None:
            return FRectf(mat)
        return None

    @overload
    def __init__(self, size : FVec2):
        """
        as size in (0,0) pos
        """
    @overload
    def __init__(self, pos : FVec2, size : FVec2):
        """"""
    @overload
    def __init__(self, w : int|float, h : int|float):
        """
        as (w,h) in (0,0) pos
        """
    @overload
    def __init__(self, l : int|float, t : int|float, r : int|float, b : int|float):
        """
        from l,t,r,b bbox
           t
         l-|-r
           b
        """
    @overload
    def __init__(self, p0 : FVec2f, p1 : FVec2f, p2 : FVec2f):
        """```
        from 3 pts
           p0--p1
          /   /
         p2--
        ```"""

    def __init__(self, *args, **kwargs):
        args_len = len(args)
        if args_len == 1:
            arg0, = args
            if isinstance(arg0, FVec2):
                mat = FAffMat2().scale(arg0.x, arg0.y)
            elif isinstance(arg0, FAffMat2):
                mat = arg0
            else:
                raise ValueError()
        elif args_len == 2:
            arg0, arg1 = args
            if isinstance(arg0, FVec2) and isinstance(arg1, FVec2):
                mat = FAffMat2().scale(arg1).translate(arg0)
            else:
                mat = FAffMat2().scale(arg0, arg1)
        elif args_len == 3:
            arg0, arg1, arg2 = args

            p0p1 = arg1-arg0

            width = p0p1.length
            height = (arg2-arg0).length

            mat = FAffMat2().rotate(p0p1.atan2()).scale(width, height).translate(arg0)

        elif args_len == 4:
            l, t, r, b = args

            mat = FAffMat2().scale(r-l,b-t).translate(l, t)
        else:
            raise ValueError()

        self._mat = mat

    def clone(self) -> Self:
        f = self.__class__.__new__(self.__class__)
        f._mat = self._mat
        return f

    def get_state(self) -> FDict: return FDict({'mat': self._mat.get_state(), })

    # @cached_property
    # def mat(self) -> FAffMat2:
    #     """mat to transform FRectf space to origin space"""
    #     return FAffMat2.estimate(self, FRectf(0,0,self.width,self.height))

    @property
    def size(self) -> FVec2f:
        """width,height"""
        return FVec2f(self.width, self.height)

    @cached_property
    def width(self) -> float:
        p0, p1 = self._mat.map([FVec2f(0,0), FVec2f(1,0)])
        return (p1-p0).length
    @cached_property
    def height(self) -> float:
        p0, p1 = self._mat.map([FVec2f(0,0), FVec2f(0,1)])
        return (p1-p0).length
    @cached_property
    def area(self) -> float:
        return self.width*self.height

    @property
    def p0(self) -> FVec2f:
        """```
         0--x
         |  |
         x--x
        ```"""
        return self.as_4pts()[0]
    @property
    def p1(self) -> FVec2f:
        """```
         x--1
         |  |
         x--x
        ```"""
        return self.as_4pts()[1]
    @property
    def p2(self) -> FVec2f:
        """```
         x--x
         |  |
         x--2
        ```"""
        return self.as_4pts()[2]
    @property
    def p3(self) -> FVec2f:
        """```
         x--x
         |  |
         3--x
        ```"""
        return self.as_4pts()[3]

    @cached_property
    def pc(self) -> FVec2f:
        """center point"""
        return self._mat.map([FVec2f(0.5, 0.5)])[0]


    @cached_method
    def as_3pts(self) -> FVec2Array:
        """```
        get rect as 3 pts

         0--1
         |  |
         2--
        ```"""
        return self._mat.map(((0,0),(1,0),(0,1)))

    @cached_method
    def as_4pts(self) -> FVec2Array:
        """```
        get rect as 4 pts

         0--1
         |  |
         3--2
        ```"""
        return self._mat.map(((0,0),(1,0),(1,1),(0,1)))

    @cached_method
    def to_poly(self, closed=True) -> FPoly2f:
        pts = self.as_4pts()
        if closed:
            pts = pts+(pts[0],)
        return FPoly2f(pts)

    def to_bbox(self) -> FBBoxf:
        """ltrb bbox"""
        pts = self.as_4pts().as_np()
        return FBBoxf(pts[:,0].min(), pts[:,1].min(), pts[:,0].max(), pts[:,1].max())


    def inflate_to_square(self) -> Self:
        """"inflate" from center rect to nearest square, i.e w==h"""
        w = self.width
        h = self.height
        if w == h:
            return self

        if w > h:
            d=w/h/2
            p0, p1, p2 = self._mat.map([FVec2f(0, 0.5-d),
                                        FVec2f(1, 0.5-d),
                                        FVec2f(0, 0.5+d) ])

            return FRectf(p0,p1,p2)
        else:
            d=h/w/2
            p0, p1, p2 = self._mat.map([FVec2f(0.5-d, 0),
                                        FVec2f(0.5+d, 0),
                                        FVec2f(0.5-d, 1) ])

            return FRectf(p0,p1,p2)


    def transform(self, mat : FAffMat2) -> Self:
        """Tranform FRectf using FAffMat2"""
        self = self.clone()
        self._mat = self._mat * mat
        return self


    def __hash__(self): return hash(self._mat)
    def __eq__(self, other) -> bool:
        if self is other:
            return True
        if isinstance(other, FRectf):
            return self._mat == other._mat
        return False

    def __repr__(self): return self.__str__()
    def __str__(self): return f'FRectf: W:{self.width} H:{self.height} {self.as_4pts()}'
