from typing import Iterable

import numpy as np

from .FAffMat3 import FAffMat3
from .FVec3Array import FVec3Array
from .math_ import umeyama


def FAffMat3_estimate(*args) -> FAffMat3:
    args_len = len(args)
    if args_len == 2:
        arg0, arg1 = args
        if isinstance(arg0, Iterable) and isinstance(arg1, Iterable):
            # pts -> pts
            if isinstance(arg0, FVec3Array):
                arg0 = arg0.as_np().astype(np.float64, copy=False)
            elif isinstance(arg0, np.ndarray):
                arg0 = arg0.astype(np.float64, copy=False)
            else:
                arg0 = np.array(arg0, np.float64)

            if isinstance(arg1, FVec3Array):
                arg1 = arg1.as_np().astype(np.float64, copy=False)
            elif isinstance(arg1, np.ndarray):
                arg1 = arg1.astype(np.float64, copy=False)
            else:
                arg1 = np.array(arg1, np.float64)

            return FAffMat3(umeyama(arg0, arg1)[:3])

        else:
            raise ValueError()

# set implementation
FAffMat3.estimate = FAffMat3_estimate


_nothing = ...