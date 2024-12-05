from __future__ import annotations

import math
from functools import cached_property
from typing import TYPE_CHECKING, Iterable, Self, Sequence, overload

import numpy as np

from ..collections import FDict
from .FLine2f import FLine2f
from .FVec2 import FVec2, FVec2_like, FVec2f
from .FVec2Array import FVec2Array, FVec2Array_like, FVec2fArray

if TYPE_CHECKING:
    from .FRectf import FRectf


class FAffMat2(Sequence):
    """Immutable Affine 2D transformation matrix"""

    @staticmethod
    def from_state(state : FDict|None):
        state = FDict(state)
        if (values := state.get('values', None)) is not None:
            return FAffMat2(values)
        return None


    @overload
    @staticmethod
    def estimate(src : FRectf, dst : FRectf) -> FAffMat2:
        """"""
    @overload
    @staticmethod
    def estimate(src : Iterable[FVec2_like], dst : Iterable[FVec2_like]) -> FAffMat2:
        """"""
    @staticmethod
    def estimate(*args, **kwargs) -> FAffMat2: raise

    @overload
    def __init__(self):
        """identity mat"""
    @overload
    def __init__(self, vec : FAffMat2): ...
    @overload
    def __init__(self, values : Iterable[int|float]):
        """from (6,) or (2,3) iterable values"""
    @overload
    def __init__(self, values : np.ndarray):
        """from (6,) or (2,3) np.ndarray"""

    def __init__(self, *args, **kwargs):
        args_len = len(args)
        if args_len == 0:
            self._values = _identity
        elif args_len == 1:
            arg0 = args[0]
            if isinstance(arg0, FAffMat2):
                values = arg0._values
            elif isinstance(arg0, np.ndarray):
                values = arg0.astype(np.float32, copy=False).reshape(2,3)
            elif isinstance(arg0, Iterable):
                values = np.array(arg0, np.float32).reshape(2,3)

            if values.shape != (2,3):
                raise ValueError('wrong shape')

            values.setflags(write=False)
            self._values = values

    def clone(self) -> Self:
        f = self.__class__.__new__(self.__class__)
        f._values = self._values
        return f

    def get_state(self) -> FDict: return FDict({'values': self._values})

    @overload
    def translate(self, v : FVec2) -> Self: ...
    @overload
    def translate(self, tx : int|float, ty : int|float) -> Self: ...
    def translate(self, *args) -> Self:
        if len(args) == 1:
            v = args[0]
            tx = v.x
            ty = v.y
        else:
            tx = args[0]
            ty = args[1]

        return self*np.float32((1,0,tx,0,1,ty))

    @overload
    def scale(self, sv : FVec2) -> Self: ...
    @overload
    def scale(self, sx : float, sy : float = None) -> Self: ...
    def scale(self, *args, **kwargs) -> Self:
        if len(args) == 1:
            v = args[0]
            if isinstance(v, FVec2):
                sx = v.x
                sy = v.y
            else:
                sx = v
                sy = None
        else:
            sx = args[0]
            sy = args[1]

        return self*np.float32((sx,0,0,0,sy if sy is not None else sx,0))

    @overload
    def scale_space(self, sv : FVec2) -> Self:
        """scale space under mat"""
    @overload
    def scale_space(self, sv : int|float) -> Self:
        """scale space under mat"""
    @overload
    def scale_space(self, sx : int|float, sy : int|float) -> Self:
        """scale space under mat"""
    def scale_space(self, *args) -> Self:
        if len(args) == 1:
            arg0 = args[0]
            if isinstance(arg0, FVec2):
                sx = arg0.x
                sy = arg0.y
            else:
                sx = sy = arg0
        else:
            sx = args[0]
            sy = args[1]

        s0,s1,s2 = FVec2fArray([(0,0), (sx,0), (0,sy)])
        t0,t1,t2 = self.map([(0,0), (1,0), (0,1)]) * FVec2f(sx,sy)

        s0s1 = s1-s0
        s0s2 = s2-s0
        t0t1 = t1-t0
        sx = (t1-t0).length / s0s1.length
        sy = (t2-t0).length / s0s2.length

        rot = t0t1.atan2() - s0s1.atan2()
        tr = t0-s0
        return FAffMat2().rotate(rot).scale(sx,sy).translate(tr)


    def rotate_deg(self, deg : int|float) -> Self: return self.rotate(deg * math.pi / 180.0)
    def rotate(self, rad : int|float) -> Self:
        alpha = math.cos(rad)
        beta = math.sin(rad)
        return self*np.float32((alpha, -beta, 0,
                                beta, alpha, 0))

    @cached_property
    def inverted(self) -> Self:
        ( (a, b, c), (d, e, f) ) = self._values
        D = a*e - b*d
        D = 1.0 / D if D != 0.0 else 0.0

        result = FAffMat2.__new__(FAffMat2)
        result._values = np.float32( ( ( e*D, -b*D, (b*f-e*c)*D),
                                       (-d*D,  a*D, (d*c-a*f)*D))   )


        return result

    def as_np(self) -> np.ndarray:
        """as (2,3) np.ndarray"""
        return self._values

    @overload
    def map(self, point : FVec2_like) -> FVec2f: ...
    @overload
    def map(self, point : FVec2f) -> FVec2f: ...
    @overload
    def map(self, line : FLine2f) -> FLine2f: ...
    @overload
    def map(self, lines : Iterable[FLine2f]) -> Sequence[FLine2f]: ...
    @overload
    def map(self, points : FVec2Array_like) -> FVec2fArray: ...
    @overload
    def map(self, points : Iterable) -> FVec2fArray: ...
    @overload
    def map(self, points : np.ndarray) -> np.ndarray: ...

    def map(self, *args):
        arg0 = args[0]
        if isinstance(arg0, np.ndarray):
            return (self._values @ np.pad(arg0, ((0,0), (0,1)), constant_values=(1,), mode='constant').T).T.astype(np.float32, copy=False)
        elif isinstance(arg0, FVec2Array):
            return FVec2fArray(self.map(arg0.as_np()))
        elif isinstance(arg0, FVec2):
            return FVec2f(self.map([arg0])[0])
        elif isinstance(arg0, FLine2f):
            return self.map([arg0])[0]
        elif isinstance(arg0, Iterable):
            try:
                el0 = next(iter(arg0))
                if isinstance(el0, FLine2f):
                    return tuple( FLine2f(x) for x in
                                  self.map(np.float32(arg0).reshape(-1,2)).reshape(-1,2,2)  )
                return FVec2fArray(self.map(np.float32(arg0)))
            except:
                return ()
        else:
            raise ValueError()


    # Collection
    def __len__(self) -> int: return 6
    # Sequence
    def __getitem__(self, key) -> float:
        return float(self._values.reshape(-1)[key])


    def __hash__(self): return hash(self._values)
    def __eq__(self, other) -> bool:
        if self is other:
            return True
        if isinstance(other, FAffMat2):
            return np.all(self._values == other._values)
        return False

    def __mul__(self, other) -> FAffMat2:
        ( (Aa, Ab, Ac), (Ad, Ae, Af) ) = self._values
        if isinstance(other, FAffMat2):
           ( (Ba, Bb, Bc), (Bd, Be, Bf) ) = other._values
        else:
           Ba, Bb, Bc, Bd, Be, Bf = other

        result = self.__class__.__new__(self.__class__)
        result._values = np.float32( (  ( Aa * Ba + Ad * Bb,
                                          Ab * Ba + Ae * Bb,
                                          Ac * Ba + Af * Bb + Bc),
                                        ( Aa * Bd + Ad * Be,
                                          Ab * Bd + Ae * Be,
                                          Ac * Bd + Af * Be + Bf) ))

        return result

    def __repr__(self): return self.__str__()
    def __str__(self): return f"{self._values}"

_identity = np.array([[1,0,0],[0,1,0]], np.float32)
_identity.setflags(write=False)