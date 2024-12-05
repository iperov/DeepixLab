"""
portable llvm clang C++ compiler using ziglang package
"""
from ._cc import (Compiler, c_bool, c_float32, c_float32_p, c_float64,
                  c_float64_p, c_int8, c_int8_p, c_int32, c_int32_p, c_int64,
                  c_uint8, c_uint8_p, c_uint32, c_uint32_p, c_uint64, c_void,
                  c_void_p, get_lib_path, get_lib_suffix, lib_import)
