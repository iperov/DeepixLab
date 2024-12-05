from __future__ import annotations

from typing import Iterable, Self

from .FNDArray import FNDArray


class FNDArrayArif:
    """FNDArray arithmetics, adds class methods"""


    def __radd__(self, other) -> Self:
        self = self.clone()
        if isinstance(other, FNDArray):
            values = other._values + self._values
        elif isinstance(other, Iterable) or type(other) in (int, float):
            values = other + self._values
        else:
            raise ValueError()
        if values.shape != self._values.shape:
            raise ValueError()
        self._values = values.astype(self._dtype, copy=False)
        return self

    def __add__(self, other) -> Self:
        self = self.clone()
        if isinstance(other, FNDArray):
            values = self._values + other._values
        elif isinstance(other, Iterable) or type(other) in (int, float):
            values = self._values + other
        else:
            raise ValueError()
        if values.shape != self._values.shape:
            raise ValueError()
        self._values = values.astype(self._dtype, copy=False)
        return self

    def __rsub__(self, other) -> Self:
        self = self.clone()
        if isinstance(other, FNDArray):
            values = other._values - self._values
        elif isinstance(other, Iterable) or type(other) in (int, float):
            values = other - self._values
        else:
            raise ValueError()
        if values.shape != self._values.shape:
            raise ValueError()
        self._values = values.astype(self._dtype, copy=False)
        return self

    def __sub__(self, other) -> Self:
        self = self.clone()
        if isinstance(other, FNDArray):
            values = self._values - other._values
        elif isinstance(other, Iterable) or type(other) in (int, float):
            values = self._values - other
        else:
            raise ValueError()
        if values.shape != self._values.shape:
            raise ValueError()
        self._values = values.astype(self._dtype, copy=False)
        return self


    def __rmul__(self, other) -> Self:
        self = self.clone()
        if isinstance(other, FNDArray):
            values = other._values * self._values
        elif isinstance(other, Iterable) or type(other) in (int, float):
            values = other * self._values
        else:
            raise ValueError()
        if values.shape != self._values.shape:
            raise ValueError()
        self._values = values.astype(self._dtype, copy=False)
        return self

    def __mul__(self, other) -> Self:
        self = self.clone()
        if isinstance(other, FNDArrayArif):
            values = self._values * other._values
        elif isinstance(other, Iterable) or type(other) in (int, float):
            values = self._values * other
        else:
            raise ValueError()
        if values.shape != self._values.shape:
            raise ValueError()
        self._values = values.astype(self._dtype, copy=False)
        return self

    def __pow__(self, other) -> Self:
        self = self.clone()
        if  type(other) in (int, float):
            values = self._values ** other
        else:
            raise ValueError()
        self._values = values.astype(self._dtype, copy=False)
        return self

    def __rdiv__(self, other) -> Self: return FNDArrayArif.__rtruediv__(self, other)
    def __div__(self, other) -> Self: return FNDArrayArif.__truediv__(self, other)

    def __rtruediv__(self, other) -> Self:
        self = self.clone()
        if isinstance(other, FNDArray):
            values = other._values / self._values
        elif isinstance(other, Iterable) or type(other) in (int, float):
            values = other / self._values
        else:
            raise ValueError()

        self._values = values.astype(self._dtype, copy=False)
        return self

    def __truediv__(self, other) -> Self:
        self = self.clone()
        if isinstance(other, FNDArray):
            values = self._values / other._values
        elif isinstance(other, Iterable) or type(other) in (int, float):
            values = self._values / other
        else:
            raise ValueError()

        self._values = values.astype(self._dtype, copy=False)
        return self

    def __neg__(self):
        self = self.clone()
        self._values = -self._values
        return self

    def __mod__(self, other) -> Self:
        self = self.clone()
        if isinstance(other, FNDArray):
            values = self._values % other._values
        elif isinstance(other, Iterable) or type(other) in (int, float):
            values = self._values % other
        else:
            raise ValueError()

        self._values = values.astype(self._dtype, copy=False)
        return self

    def __and__(self, other) -> Self:
        self = self.clone()
        if isinstance(other, FNDArray):
            values = self._values & other._values
        elif isinstance(other, Iterable) or type(other) in (int, float):
            values = self._values & other
        else:
            raise ValueError()

        self._values = values.astype(self._dtype, copy=False)
        return self