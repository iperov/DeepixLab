from __future__ import annotations

from typing import Self

import numpy as np

from ..collections import FDict
from .FAnno import FAnno


class FAnnoID(FAnno):
    """describes Face ID as vector"""

    @staticmethod
    def from_state(state : FDict|None) -> FAnnoID|None:
        state = FDict(state)
        if (vector := state.get('vector', None)) is not None:
            return FAnnoID(vector)
        return None

    def __init__(self, vector : np.ndarray):
        """(N,)"""
        super().__init__()

        if len(vector.shape) != 1:
            raise ValueError()

        self._vector = vector.astype(np.float32, copy=False)

    def clone(self) -> Self:
        f = super().clone()
        f._vector = self._vector
        return f

    def get_state(self) -> FDict: return FDict({'vector' : self._vector, })

    @property
    def vector(self) -> np.ndarray:
        """(N,) nd array f32"""
        return self._vector
