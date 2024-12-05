from __future__ import annotations

from .FNDArray import FNDArray
from .FNDArrayArif import FNDArrayArif


class FLine(FNDArray, FNDArrayArif):
    """base class for lines"""

    def __len__(self): return self._values.shape[0]





