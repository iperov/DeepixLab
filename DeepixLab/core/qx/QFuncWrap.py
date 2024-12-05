from typing import Callable, TypeVar

from .. import mx, qt

T = TypeVar('T')

class QFuncWrap(mx.Disposable):
    def __init__(self, obj : T, func_name : str, wrapper : Callable):
        """
        disposable qt object function wrapper

            wrapper(*args, **kwargs)
        """
        super().__init__()
        self._obj = obj
        self._func_name = func_name
        self._super = qt.wrap(obj, func_name, lambda obj, super, *args, **kwargs: wrapper(*args, **kwargs))

    def get_super(self) -> Callable:
        return self._super

    def __dispose__(self):
        qt.unwrap(self._obj, self._func_name, self._super)
        super().__dispose__()