from typing import Iterable

import numpy as np

from .FAffMat2 import FAffMat2
from .FRectf import FRectf
from .FVec2Array import FVec2Array
from .math_ import umeyama


def FAffMat2_estimate(*args) -> FAffMat2:
    args_len = len(args)
    if args_len == 2:
        arg0, arg1 = args
        if isinstance(arg0, FRectf) and isinstance(arg1, FRectf):
            # rect -> rect
            return FAffMat2_estimate(arg0.as_3pts(), arg1.as_3pts())

        elif isinstance(arg0, Iterable) and isinstance(arg1, Iterable):
            # pts -> pts
            if isinstance(arg0, FVec2Array):
                arg0 = arg0.as_np().astype(np.float64, copy=False)
            elif isinstance(arg0, np.ndarray):
                arg0 = arg0.astype(np.float64, copy=False)
            else:
                arg0 = np.array(arg0, np.float64)

            if isinstance(arg1, FVec2Array):
                arg1 = arg1.as_np().astype(np.float64, copy=False)
            elif isinstance(arg1, np.ndarray):
                arg1 = arg1.astype(np.float64, copy=False)
            else:
                arg1 = np.array(arg1, np.float64)

            return FAffMat2(umeyama(arg0, arg1)[:2])

        else:
            raise ValueError()

# set implementation
FAffMat2.estimate = FAffMat2_estimate


_nothing = ...