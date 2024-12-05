from __future__ import annotations

from typing import Self, Type

import numpy as np

from ..collections import FDict


class FNDArray:
    """base class for F-classes that use np.ndarray internally"""

    @staticmethod
    def _from_state[T](state : FDict|None, cls : Type[T]) -> T|None:
        state = FDict(state)
        if (values := state.get('values', None)) is not None:
            return cls(values)
        return None

    def __init__(self, values : np.ndarray):
        self._values = values.astype(self._dtype, copy=False)

    def clone(self) -> Self:
        f = self.__class__.__new__(self.__class__)
        f._values = self._values
        return f

    def get_state(self) -> FDict: return FDict({'values': self._values})

    @property
    def _dtype(self) -> np.dtype: raise NotImplementedError()

    def as_np(self) -> np.ndarray: return self._values

    def __hash__(self): return hash(self._values)
    def __eq__(self, other) -> bool:
        if self is other:
            return True
        if isinstance(other, FNDArray):
            values = self._values
            o_values = other._values
            return values.shape == o_values.shape and \
                   np.all(values == o_values)
        return False

    def __repr__(self): return self.__str__()
    def __str__(self): return f'{self.__class__.__name__} {self._values}'