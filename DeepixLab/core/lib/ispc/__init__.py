"""Intel ISPC compiler wrapper

Uses ziglang package as c++ compiler backend.
"""

from .ispc import (c_bool, c_float, c_float_p, c_int8, c_int8_p, c_int32,
                   c_int32_p, c_uint8, c_uint8_p, c_uint32, c_uint32_p,
                   compile_dll, compile_o, dll_import, get_shared_lib_suffix)
