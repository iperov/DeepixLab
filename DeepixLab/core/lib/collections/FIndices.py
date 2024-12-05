from __future__ import annotations

from functools import cached_property
from typing import Collection, Iterable, Iterator, Sequence, overload

import numpy as np

from ..functools import cached_method


class FIndices(Collection):
    """Immutable positive indexes collection internally implemented as bit array for fast and efficient operations."""

    @overload
    @staticmethod
    def from_range(stop : int) -> FIndices:
        """faster than FIndices(range())"""
    @overload
    @staticmethod
    def from_range(start : int, stop : int, step : int = 1) -> FIndices:
        """faster than FIndices(range())"""
    @staticmethod
    def from_range(*args, **kwargs) -> FIndices:
        if len(args) <= 1:
            start = 0
            stop = args[0]
            step = 1
        else:
            start = args[0]
            stop = args[1]
            step = kwargs.get('step', args[2] if len(args) >= 3 else 1)

        bits = np.zeros((stop,), dtype=bool)
        bits[start:stop:step] = True
        return FIndices(_bits=bits)

    def __init__(self, indices : FIndices|Iterable|int|None = None, **kwargs):
        """
        negative indexes will be discarded automatically

        """

        super().__init__()
        if (bits := kwargs.get('_bits', None)) is not None:
            pass
        elif isinstance(indices, FIndices):
            bits = indices._bits
        else:
            if isinstance(indices, Iterable):
                pass
            elif isinstance(indices, int):
                indices = [indices]
            elif indices is None:
                indices = []
            else:
                raise ValueError()

            indices = np.fromiter(indices, dtype=np.int32)
            # Discard negative values
            indices = indices[indices >= 0]

            bits = np.zeros( (1 + indices.max() if indices.shape[0] != 0 else 0,), dtype=bool )
            bits[indices] = True

        self._bits = bits

    @cached_property
    def count(self) -> int: return int(self._bits.sum())

    @cached_property
    def min(self) -> int:
        """raise if len() == 0"""
        return int(np.argwhere(self._bits).min())

    @cached_property
    def max(self) -> int:
        """raise if len() == 0"""
        return int(np.argwhere(self._bits).max())

    def difference(self, indices : FIndices|Iterable|int|None) -> FIndices:
        """Return a new FIndices with elements in the FIndices that are not in the others.  """
        bits1, bits2 = FIndices._align_second(self._bits, FIndices(indices)._bits)
        return FIndices(_bits = bits1 & ~bits2)

    def union(self, indices : FIndices|Iterable|int|None) -> FIndices:
        """Return a new FIndices with elements from the FIndices and all others. Same as `self|indices`"""
        return self | indices

    def symmetric_difference(self, indices : FIndices|Iterable|int|None) -> FIndices:
        """Return a new FIndices with elements in either the FIndices or other but not both. Same as `self^indices`"""
        return self ^ indices

    def intersection(self, indices : FIndices|Iterable|int|None) -> FIndices:
        """Return a new FIndices with elements common to the FIndices and all others. Same as `self&indices`"""
        return self & indices

    def discard(self, indices : FIndices|Iterable|int|None, shift=False) -> FIndices:
        """
        Same as difference.

            shift(False)    Right-hand indices will be shifted
                            Example:
                            FIndices([0,3,5,8]).discard([0,5], shift=True)
                            >> FIndices([2,6])
        """
        if shift:
            bits1, bits2 = FIndices._align_second(self._bits, FIndices(indices)._bits)
            return FIndices(_bits = bits1[~bits2])
        else:
            return self.difference(indices)

    def discard_l(self, indice : int, shift=False) -> FIndices:
        """remove all indices less than `indice`

            shift(False)    Right-hand indices will be shifted
        """
        if shift:
            return FIndices(_bits=self._bits[indice:])
        else:
            bits = self._bits.copy()
            bits[:indice] = False
            return FIndices(_bits=bits)

    def discard_ge(self, indice : int) -> FIndices:
        """remove all indices greater/equal than `indice`"""
        return FIndices(_bits=self._bits[:max(0,indice)])

    def invert(self, max : int = None) -> FIndices:
        """

        """
        if max is None:
            max = self.max
        bits = FIndices._align_to_size(self._bits, size=max )
        return FIndices(_bits=1-bits)


    def indwhere(self, indices : FIndices|Iterable|int|None) -> FIndices:
        """example:

        FIndices([0,2,5,9]).indwhere([2,9,99]) -> (1,3)
        """
        indices = FIndices(indices)
        x = np.argwhere(np.in1d(np.argwhere(self._bits)[:,0],
                                np.argwhere(indices._bits)[:,0] ))[:,0]
        return FIndices(x)

    @cached_method
    def to_list(self) -> Sequence: return tuple(np.argwhere(self._bits)[:,0].tolist())

    @cached_method
    def to_set(self) -> frozenset: return frozenset(np.argwhere(self._bits)[:,0])

    # Collection
    def __len__(self) -> int:  return self.count

    # Container
    def __contains__(self, x : int) -> bool:
        if type(x) == int:
            if x < 0 or x >= self._bits.shape[0]:
                return False
            return bool(self._bits[x])
        return False

    # Iterable
    def __iter__(self) -> Iterator[int]:
        for x in np.argwhere(self._bits)[:,0]:
            yield x

    # Base
    def __and__(self, indices : FIndices|Iterable|int|None) -> FIndices:
        bits1, bits2 = FIndices._reduce_to_smaller(self._bits, FIndices(indices)._bits)
        return FIndices(_bits = bits1 & bits2)

    def __or__(self, indices : FIndices|Iterable|int|None) -> FIndices:
        bits1, bits2 = FIndices._broadcast_to_greater(self._bits, FIndices(indices)._bits)
        return FIndices(_bits = bits1 | bits2)

    def __xor__(self, indices : FIndices|Iterable|int|None) -> FIndices:
        bits1, bits2 = FIndices._broadcast_to_greater(self._bits, FIndices(indices)._bits)
        return FIndices(_bits = bits1 ^ bits2)

    def __eq__(self, indices):
        if isinstance(indices, (FIndices,Iterable) ):
            indices = FIndices(indices)
            bits1, bits2 = FIndices._broadcast_to_greater(self._bits, indices._bits)
            return np.all(bits1==bits2)

        return False

    def __repr__(self) -> str: return f'FIndices({self.to_list().__repr__()})'

    # Common code
    @staticmethod
    def _broadcast_to_greater(bits1, bits2):
        shape_diff = bits1.shape[0] - bits2.shape[0]

        if shape_diff < 0:
            bits1 = np.concatenate([bits1, np.zeros( (-shape_diff,), dtype=bool)], 0)
        elif shape_diff > 0:
            bits2 = np.concatenate([bits2, np.zeros( (shape_diff,), dtype=bool)], 0)

        return bits1, bits2

    @staticmethod
    def _reduce_to_smaller(bits1, bits2):
        shape_diff = bits1.shape[0] - bits2.shape[0]

        if shape_diff < 0:
            bits2 = bits2[:shape_diff]
        elif shape_diff > 0:
            bits1 = bits1[:-shape_diff]

        return bits1, bits2

    @staticmethod
    def _align_second(bits1, bits2):
        shape_diff = bits1.shape[0] - bits2.shape[0]

        if shape_diff < 0:
            bits2 = bits2[:shape_diff]
        elif shape_diff > 0:
            bits2 = np.concatenate([bits2, np.zeros( (shape_diff,), dtype=bool)], 0)

        return bits1, bits2

    @staticmethod
    def _align_to_size(bits, size : int, expand_value : bool = False):
        shape_diff = size - bits.shape[0]

        if shape_diff < 0:
            bits = bits[:shape_diff]
        elif shape_diff > 0:
            bits = np.concatenate([bits, np.full( (size,), expand_value, dtype=bool)], 0)

        return bits
