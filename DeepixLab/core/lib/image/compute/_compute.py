from pathlib import Path

import numpy as np

from ... import cc


def find_nearest_hist(hist_ar : np.ndarray, c_idx : int, s_idx : int, e_idx : int, c_avg_count=16) -> int:
    """
        `hist_ar`   (N, C, BINS) int32/float32

        `c_idx`     index of hist to be checked with others

        `s_idx`     index of start searching

        `e_idx`     index of end searching (exclusive)

        `c_avg_count`   average c_idx with c_idx-c_avg_count, result is better clustering comparing to value 1

    returns index of hist the most similar to `c_idx`
    """
    S, _, _ = hist_ar.shape
    hist_ar = np.ascontiguousarray(hist_ar)

    if hist_ar.dtype == np.int32:
        return c_find_nearest_hist_u8(hist_ar.ctypes.data_as(cc.c_uint8_p), S, c_idx, c_avg_count, s_idx, e_idx )
    elif hist_ar.dtype == np.float32:
        return c_find_nearest_hist_f32(hist_ar.ctypes.data_as(cc.c_float32_p), S, c_idx, c_avg_count, s_idx, e_idx )

    raise ValueError('hist_ar.dtype ')

lib_path = cc.get_lib_path(Path(__file__).parent, Path(__file__).stem)

@cc.lib_import(lib_path)
def c_find_nearest_hist_u8(hist_ar : cc.c_uint8_p, S : cc.c_int32, c_idx : cc.c_int32, c_avg_count : cc.c_int32, s_idx : cc.c_int32, e_idx : cc.c_int32) -> cc.c_int32: ...
@cc.lib_import(lib_path)
def c_find_nearest_hist_f32(hist_ar : cc.c_float32_p, S : cc.c_int32, c_idx : cc.c_int32, c_avg_count : cc.c_int32, s_idx : cc.c_int32, e_idx : cc.c_int32) -> cc.c_int32: ...

def setup_compile():
    return cc.Compiler(Path(__file__).parent).shared_library(lib_path).minimal_c().optimize().include_glsl().include('compute.cpp').compile()

#setup_compile()
