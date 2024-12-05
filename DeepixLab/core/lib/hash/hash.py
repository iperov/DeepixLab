from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from core.lib import cc

from ..math import bit_count


class Hash64(np.uint64):
    """64-bit hash value"""

    @staticmethod
    def sorted_by_dissim(hashes : Sequence[Hash64]) -> Sequence[int]:
        """
        returns Sequence of idx of hashes with most dissimilarities in descending order.
        """
        hashes = np.array(hashes)

        x = []
        for i, hash in enumerate(hashes):

            dissim_count = bit_count(hashes ^ hash).sum()

            x.append( (i, dissim_count) )

        x = sorted(x, key=lambda v: v[1], reverse=True)
        x = [v[0] for v in x]

        return x



class Hash64Similarity:
    def __init__(self, count : int, similarity_factor : int = 8):
        """
        a class for continuous computation every-with-every similarity of Hash64'es

            count       number of indexes to be computed

            similarity_factor(8)  0..63
        """
        if not (count > 0):
            raise ValueError(f'count must be > 0')
        if not (similarity_factor >= 0 and similarity_factor <= 63):
            raise ValueError(f'similarity_factor must be in range [0..63]')

        self._count = count
        self._similarity_factor = similarity_factor

        self._hashed_map = np.zeros( (count,), dtype=np.uint8 )

        self._hashed_count = np.zeros( (1,), dtype=np.uint32 )
        self._hashed_idxs  = np.zeros( (count,), dtype=np.uint32 )
        self._hashes       = np.zeros( (count,), dtype=np.uint64 )

        self._similarities = np.ones( (count,), dtype=np.uint32 )


    def add(self, hash_idx : int, hash : Hash64):
        """
        add hash with idx, and update similarities with already added hashes

        if idx already hashed, nothing will happen
        """
        c_similarity_add(   self._similarities.ctypes.data_as(cc.c_uint32_p),
                            self._hashed_map.ctypes.data_as(cc.c_uint8_p),
                            self._hashed_idxs.ctypes.data_as(cc.c_uint32_p),
                            self._hashes.ctypes.data_as(cc.c_void_p),
                            self._hashed_count.ctypes.data_as(cc.c_uint32_p),
                            hash_idx, hash, self._similarity_factor)

    def hashed(self, idx : int) -> bool:
        return self._hashed_map[idx] != 0

    def get_similarities(self) -> np.ndarray:
        return self._similarities

lib_path = cc.get_lib_path(Path(__file__).parent, Path(__file__).stem)

@cc.lib_import(lib_path)
def c_similarity_add(similarities : cc.c_uint32_p, hashed_map : cc.c_uint8_p, hashed_idxs : cc.c_uint32_p, hashes : cc.c_void_p, hashed_count : cc.c_uint32_p, hash_idx : cc.c_uint32, hash : cc.c_uint64, similarity_factor : cc.c_uint32) -> cc.c_void: ...

def setup_compile():
    return cc.Compiler(Path(__file__).parent).shared_library(lib_path).minimal_c().optimize().include_glsl().include('hash.cpp').compile()
