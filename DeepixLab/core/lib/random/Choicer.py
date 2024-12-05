from __future__ import annotations

from typing import Any, Sequence

import numpy as np


class Choicer:
    """
    Class for randomly pick items with given probabilities in range [ 0.001 .. 1.0 ]

    Supports nested Choicers. If picked item is Choicer then item picked from it.
    """

    def __init__(self, items : Sequence[ Any|Choicer ],
                       probs : Sequence[int|float]|np.ndarray ):
        """ probs   [ 0.001 .. 1.0 ] """
        self._items = items
        self._probs = probs = np.array(probs, np.float32).clip(0.001, 1.0)

        if len(probs) != len(items):
            raise ValueError('must len(probs) == len(items)')

        if len(self._items) != 0:
            # how often each item will occur
            rates = (probs/probs.min()).astype(np.int32)

            # base idx sequence, for example Choicer(['a', 'b', 'c'], [1,1,0.5]) , idxs_base == [0,0,1,1,2]
            self._idxs_base = np.concatenate([np.full( (x,), i, dtype=np.uint32) for i,x in enumerate(rates)], 0)

            self._idxs = None
            self._idx_counter = 0

    @property
    def items(self) -> Sequence[ Any|Choicer ]: return self._items
    @property
    def probs(self) -> np.ndarray: return self._probs

    def pick(self, count : int) -> Sequence[Any]:
        """pick `count` items"""
        out = []
        if len(self._items) != 0:
            while len(out) < count:
                if self._idx_counter == 0:
                    self._idxs = self._idxs_base.copy()
                    np.random.shuffle(self._idxs)
                    self._idx_counter = len(self._idxs)

                self._idx_counter -= 1
                idx = self._idxs[self._idx_counter]

                item = self._items[idx]
                if isinstance(item, Choicer):
                    item = item.pick(1)[0]

                out.append(item)
        return out