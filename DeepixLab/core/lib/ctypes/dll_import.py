import ctypes
from ctypes.util import find_library
from typing import get_type_hints

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
