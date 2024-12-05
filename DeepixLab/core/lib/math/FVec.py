from __future__ import annotations

import math
from typing import Self, Sequence

from .FNDArray import FNDArray
from .FNDArrayArif import FNDArrayArif


class FVec(FNDArray, FNDArrayArif, Sequence):
    """base class for N component vectors"""

    @property
    def length(self) -> float: return float(math.sqrt((self._values**2).sum()))

    def dot(self, other : FVec) -> float: return float((self._values * other._values).sum())

    def normalize(self) -> Self:
        self = self.clone()
        if (d := self.length) != 0:
            return self / d
        return self

    # Sequence
    def __getitem__(self, key : int) -> int|float: return self._python_type(self._values[key])

    # Collection
    def __len__(self): return self._values.shape[0]




