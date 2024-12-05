from __future__ import annotations

from typing import Self

from ..math import FAffMat2
from .FAnno import FAnno


class FAnno2D(FAnno):
    """base class for 2D annotations that can be transformed using FAffMat2"""

    def transform(self, mat : FAffMat2) -> Self:
        raise NotImplementedError()