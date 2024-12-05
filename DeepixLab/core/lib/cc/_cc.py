from __future__ import annotations

import ctypes
import platform
import subprocess
import tempfile
from ctypes import (POINTER, c_bool, c_double, c_float, c_int8, c_int32,
                    c_int64, c_uint8, c_uint32, c_uint64, c_void_p)
from ctypes.util import find_library
from pathlib import Path
from typing import Self, get_type_hints

import ziglang

from .. import path as lib_path

c_void = None
c_bool = c_bool
c_int8 = c_int8
c_int32 = c_int32
c_int64 = c_int64
c_uint8 = c_uint8
c_uint32 = c_uint32
c_uint64 = c_uint64
c_float32 = c_float
c_float64 = c_double

c_void_p = c_void_p
c_uint8_p = POINTER(c_uint8)
c_uint32_p = POINTER(c_uint32)
c_int8_p = POINTER(c_int8)
c_int32_p = POINTER(c_int32)
c_float32_p = POINTER(c_float32)
c_float64_p = POINTER(c_float64)

zig_root = Path(ziglang.__file__).parent / 'zig'

_os_shlib_suffix = {'Windows' : '.dll',
                    'Linux' : '.so',
                    'Darwin' : '.dylib', }


clang_root = Path(__file__).parent
glsl_root = Path(__file__).parent / 'glsl'
glsl_include_path = str(clang_root)
glsl_include_files = [ str(glsl_root / 'glsl_internal.cpp'), ]


class Compiler():
    def __init__(self, root : Path):
        self._root = root
        self._args = []
        self._output_filepath : Path|None = None

    def clone(self) -> Self:
        f = self.__class__.__new__(self.__class__)
        f._root = self._root
        f._args = self._args
        f._output_filepath = self._output_filepath
        return f

    @property
    def output_filepath(self) -> Path|None: return self._output_filepath

    def minimal_c(self) -> Self:
        """makes only C lib available"""
        self = self.clone()
        self._args = self._args + ['-nodefaultlibs', '-nostdinc', '-nostdinc++']
        return self

    def optimize(self) -> Self:
        self = self.clone()
        self._args = self._args + [ '-g0',
                                    '-flto=thin',  '-O3',
                                    '-fno-unwind-tables', '-fno-exceptions',
                                    '-fno-pch-codegen','-fno-pch-debuginfo', '-Wl,--strip-debug',
                                    ]
        return self

    def shared_library(self, filepath : Path, dll_entry = '_dll_entry') -> Self:
        """target shared library"""
        if self._output_filepath is not None:
            raise Exception('output is already defined')

        self = self.clone()
        self._output_filepath = filepath
        self._args = self._args + ['-shared', '-o', str(self._output_filepath)]
        return self

    def include_glsl(self) -> Self:
        """makes avail "glsl/..." lib"""
        self = self.clone()
        self._args = self._args + [ f"-I{glsl_include_path}" ] + glsl_include_files
        return self

    def include(self, filename : str) -> Self:
        """include file, relative to root"""
        self = self.clone()
        self._args = self._args + [str(self._root / filename)]
        return self

    def compile(self, cwd : Path|None = None, silent_stdout = True):
        if cwd is None:
            cwd = Path(tempfile.gettempdir())

        cc(self._args, cwd, silent_stdout=silent_stdout)

        for lib_filepath in lib_path.get_files_paths(self._root, ['.lib']):
            lib_filepath.unlink()

    def print_args(self):
        print('zig c++', ' '.join(self._args))

libs_by_name = {}
def lib_import(dll_name):
    """
    decorator for import library func.
    always annotate return type even if it is c_void !
    """
    dll_name = str(dll_name)

    def decorator(func):
        dll_func = None
        def wrapper(*args):
            nonlocal dll_func

            if dll_func is None:
                dll = libs_by_name.get(dll_name, None)
                if dll is None:
                    try:
                        dll = ctypes.cdll.LoadLibrary(find_library(dll_name))
                    except:
                        pass

                    if dll is None:
                        raise RuntimeError(f'Unable to load {dll_name} library.')

                    libs_by_name[dll_name] = dll

                dll_func = getattr(dll, func.__name__)
                anno = list(get_type_hints(func).values())
                anno = [ None if x is type(None) else x for x in anno ]

                dll_func.argtypes = anno[:-1]
                dll_func.restype = anno[-1]

            return dll_func(*args)

        return wrapper
    return decorator


def cc(args, cwd : Path|None = None, silent_stdout = True):
    """run c++ compiler with args

      cwd(None)   default is tempdir

    raise on Error
    """
    if cwd is None:
        cwd = Path(tempfile.gettempdir())

    args = [ str(zig_root), 'c++' ] + args

    p = subprocess.Popen(args, stdout=subprocess.DEVNULL if silent_stdout else None,
                               stderr=subprocess.PIPE,
                               cwd=str(cwd))

    out, err = p.communicate()

    if p.returncode != 0:
        err = err.decode('utf-8') if err is not None else None
        raise Exception(err)


def get_lib_suffix() -> str:
    suffix = _os_shlib_suffix.get(platform.system(), None)
    if suffix is None:
        raise Exception('Unknown OS')
    return suffix

def get_lib_path(root : Path, lib_name : str) -> Path:
    return root / (lib_name + get_lib_suffix())









