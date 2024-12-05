from __future__ import annotations

from typing import Generic, Iterable, Iterator, Self, TypeVar, overload

import numpy as np

from .FNDArray import FNDArray

ElemFType = TypeVar('ElemFType', bound=FNDArray)

class FArray(FNDArray, Generic[ElemFType], Iterable):
    """base class for F-arrays of any shape, for example (N,2,2) array of 2d lines"""

    @property
    def _elem_type(self): raise NotImplementedError()

    def add(self, elem : ElemFType) -> Self:
        self = self.clone()
        self._values = np.append(self._values, [elem.as_np().astype(self._dtype, copy=False)], axis=0)
        return self

    def insert(self, idx : int, elem : ElemFType) -> Self:
        self = self.clone()
        self._values = np.insert(self._values, idx, [elem.as_np().astype(self._dtype, copy=False)], axis=0)
        return self

    def remove(self, idx : int) -> Self:
        self = self.clone()
        values = self._values
        self._values = np.concatenate([values[:idx], values[idx+1:]], 0)
        return self

    def replace(self, idx : int, elem : ElemFType) -> Self:
        self = self.clone()
        values = self._values = self._values.copy()
        values[idx] = elem.as_np().astype(self._dtype, copy=False)
        return self

    # Iterable
    def __iter__(self) -> Iterator[ElemFType]:
        for i in range(len(self)):
            yield self[i]

    def __len__(self): return self._values.shape[0]
    @overload
    def __getitem__(self, key : int) -> ElemFType: ...
    @overload
    def __getitem__(self, key : Iterable|slice) -> Self: ...
    def __getitem__(self, key):
        result = self._values[key]
        if isinstance(key, int):
            return self._elem_type(result)
        elif isinstance(key, (Iterable,slice) ):
            self = self.clone()
            self._values = result
            return self
        else:
            raise ValueError()


