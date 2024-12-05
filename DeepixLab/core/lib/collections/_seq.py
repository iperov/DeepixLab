from typing import Iterable, Sequence

import numpy

from .FIndices import FIndices


def sequence_removed_indices(seq : Sequence, indices : FIndices|Iterable|int|None) -> Sequence:
    """Returns `seq` with removed `indices`"""
    indices = FIndices(indices).discard_ge(len(seq))
    if len(indices) == 0:
        return seq
    indices_seq = indices.to_list()
    
    return sum( (seq[ slice(start, stop) ] for start, stop in _slice_gen(indices_seq)), [] )

def _slice_gen(seq : Sequence):
    v = seq[0]
    yield None, v
    for x in seq:
        yield v+1, x
        v = x
    yield v+1, None

def shuffled(seq : Sequence) -> Sequence:
    idxs = [*range(len(seq))]
    numpy.random.shuffle(idxs)
    return tuple(seq[idx] for idx in idxs)
