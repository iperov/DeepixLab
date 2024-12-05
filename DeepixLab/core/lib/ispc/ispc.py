import ctypes
import os
import platform
import subprocess
from ctypes import (c_bool, c_float, c_int8, c_int32, c_uint8, c_uint32,
                    c_void_p)
from ctypes.util import find_library
from pathlib import Path
from typing import get_type_hints

import ziglang

c_float_p = c_void_p
c_uint8_p = c_void_p
c_uint32_p = c_void_p
c_int8_p = c_void_p
c_int32_p = c_void_p

dlls_by_name = {}
def dll_import(dll_name):
    """
    decorator for import DLL func.
    always annotate return type even if it is None !
    """
    dll_name = str(dll_name)

    def decorator(func):
        dll_func = None
        def wrapper(*args):
            nonlocal dll_func

            if dll_func is None:
                dll = dlls_by_name.get(dll_name, None)
                if dll is None:
                    try:
                        dll = ctypes.cdll.LoadLibrary(find_library(dll_name))
                    except:
                        pass

                    if dll is None:
                        raise RuntimeError(f'Unable to load {dll_name} library.')

                    dlls_by_name[dll_name] = dll

                dll_func = getattr(dll, func.__name__)
                anno = list(get_type_hints(func).values())
                anno = [ None if x is type(None) else x for x in anno ]

                dll_func.argtypes = anno[:-1]
                dll_func.restype = anno[-1]

            return dll_func(*args)

        return wrapper
    return decorator


zig_bin = Path(ziglang.__file__).parent / 'zig'

if (ispc_root := os.environ.get('ISPC_PATH', None)) is not None:
    ispc_bin = Path(ispc_root) / 'bin' / 'ispc'
else:
    raise Exception('environ variable ISPC_PATH is not specified')

def cc(args, cwd : Path):
    args = [ str(zig_bin), 'c++' ] + args
    p = subprocess.Popen(args, stdout=subprocess.DEVNULL,  stderr=subprocess.PIPE, cwd=str(cwd))
    out, err = p.communicate()
    if p.returncode != 0:
        err = err.decode('utf-8') if err is not None else None
        raise Exception(err)


def compile_o(ispc_file : Path):
    """
    compile .o from .ispc file.

    raise on error
    """
    cwd = ispc_file.parent

    shared_lib_path = ispc_file.parent / (ispc_file.stem+get_shared_lib_suffix())

    ispc_h_filepath = ispc_file.parent / (ispc_file.stem+'.h')
    # ispc_tmp_c_filepath = ispc_file.parent / (ispc_file.stem+'_tmp.c')
    # ispc_tmp_lib_filepath = ispc_file.parent / (ispc_file.stem+'_tmp.lib')
    ispc_o_filepath = ispc_file.parent / (ispc_file.stem+'.o')

    include_root = Path(__file__).parent / 'include'

    args = [ str(ispc_bin), str(ispc_file), '-o', str(ispc_o_filepath),
            '-h', str(ispc_h_filepath), '-I', str(include_root), #'--opt=fast-math',
            '--pic',
            #'--werror',

              ]

    p = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, cwd=str(cwd))

    out, err = p.communicate()

    if p.returncode != 0:
        err = err.decode('utf-8') if err is not None else None
        raise Exception(err)

#     ispc_tmp_c_filepath.write_text(
# fr"""#include "{ispc_h_filepath.name}"
# extern int _fltused = 0;
# extern size_t __chkstk(size_t size)
# {{
#     __asm__(
#         "movq $0x0, %r11 \r\n"
#         "movl $0xFFFFFFFF, %r11d \r\n"
#         "and %r11, %rax \r\n"
#         "retn \r\n"
#     );
# }}
# """)

    #ispc_h_filepath.unlink()
    #ispc_o_filepath.unlink()
    # ispc_tmp_c_filepath.unlink()
    # ispc_tmp_lib_filepath.unlink()


def compile_dll(ispc_file : Path):
    """
    compile dll from .ispc file.

    raise on error
    """
    cwd = ispc_file.parent

    shared_lib_path = ispc_file.parent / (ispc_file.stem+get_shared_lib_suffix())

    ispc_h_filepath = ispc_file.parent / (ispc_file.stem+'.h')
    ispc_tmp_c_filepath = ispc_file.parent / (ispc_file.stem+'_tmp.c')
    ispc_tmp_lib_filepath = ispc_file.parent / (ispc_file.stem+'_tmp.lib')
    ispc_o_filepath = ispc_file.parent / (ispc_file.stem+'.o')

    include_root = Path(__file__).parent / 'include'

    args = [ str(ispc_bin), str(ispc_file), '-o', str(ispc_o_filepath),
            '-h', str(ispc_h_filepath), '-I', str(include_root), #'--opt=fast-math',
            '--pic',
            #'--werror',

              ]

    p = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, cwd=str(cwd))

    out, err = p.communicate()

    if p.returncode != 0:
        err = err.decode('utf-8') if err is not None else None
        raise Exception(err)

    ispc_tmp_c_filepath.write_text(
fr"""#include "{ispc_h_filepath.name}"
extern int _fltused = 0;
extern size_t __chkstk(size_t size)
{{
    __asm__(
        "movq $0x0, %r11 \r\n"
        "movl $0xFFFFFFFF, %r11d \r\n"
        "and %r11, %rax \r\n"
        "retn \r\n"
    );
}}
""")

    args = ['-g0', '-flto=thin',  '-O3',
            '-fno-unwind-tables', '-fno-exceptions',
            '-nodefaultlibs', '-nostdinc', '-nostdinc++',
            '-fno-pch-codegen','-fno-pch-debuginfo', '-Wl,--strip-debug',
            '-shared', '-o', str(shared_lib_path),
            str(ispc_tmp_c_filepath), str(ispc_o_filepath),
            ]
    cc(args, cwd=cwd)

    ispc_h_filepath.unlink()
    ispc_o_filepath.unlink()
    ispc_tmp_c_filepath.unlink()
    ispc_tmp_lib_filepath.unlink()

_os_shlib_suffix = {'Windows' : '.dll',
                    'Linux' : '.so',
                    'Darwin' : '.dylib', }

def get_shared_lib_suffix() -> str:
    suffix = _os_shlib_suffix.get(platform.system(), None)
    if suffix is None:
        raise Exception('Unknown OS')
    return suffix
