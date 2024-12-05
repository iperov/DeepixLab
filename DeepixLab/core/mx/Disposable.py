from __future__ import annotations

from collections import deque
from typing import Callable, Deque, Set, Self

DEBUG = False


class Disposable:
    """
    Base class for disposable objects.
    Not thread-safe.
    """

    def __init__(self):
        self.__disposed = 0
        self.__disposables : Deque[Disposable] = deque()
        self.__disposed_with : Set[Disposable] = set()

        if self.dispose.__func__ != Disposable.dispose:
            raise Exception(f'You must not to override {self.__class__.__qualname__}.dispose(), use __dispose__ instead.')

        if DEBUG:
            self.__dbg_parent = None
            self.__dbg_name = ''
            
    def __dispose__(self):
        """
        inheritable at last.

        provide your disposition code here.
        """
        self._dispose_items()
            
    def dispose(self) -> None:
        """
        """
        #print(f"{' '*Disposable._indent}Dispose {self}")
        if self.__disposed == 0:
            self.__disposed = 1
        else:
            raise Exception(f'Disposing already disposed {self.__class__.__qualname__}')

        # Remove self from Disposable's where we should to "dispose_with"
        disposed_with, self.__disposed_with = self.__disposed_with, set()
        for obj in disposed_with:
            obj.__disposables.remove(self)

        Disposable._indent += 2
        self.__dispose__()
        Disposable._indent -= 2

        self.__disposed = 2
        
        #print(f"{' '*Disposable._indent}Disposed {self}")

    def dispose_items(self) -> Self: 
        # explicit call, can be overriden and forbidden for user
        self._dispose_items()
        return self
    
    def _dispose_items(self) -> Self:
        """Dispose all child disposables in FILO order"""
        disposables = self.__disposables
        while len(disposables) > 0:
            disposables[0].dispose()
        return self
    
    def dispose_with(self, other : Disposable) -> Self:
        """
        `self` will be disposed on `other`.dispose() in First-In-Last-Out order.

        if `other` is `Disposable`, then `self` can be also disposed explicitly by Disposable.dispose_items()
        """
        if other.__disposed == 2:
            raise Exception(f'{other} already disposed.')

        if other not in self.__disposed_with:
            self.__disposed_with.add(other)
            other.__disposables.appendleft(self)

        return self
    
    def undispose_with(self, other : Disposable) -> Self:
        """
        """
        if other.__disposed == 2:
            raise Exception(f'{other} already disposed.')
        
        if other in self.__disposed_with:
            self.__disposed_with.remove(other)
            other.__disposables.remove(self)

        return self
    
    
    

    if DEBUG:
        def __setattr__(self, name: str, value) -> None:
            if isinstance(value, Disposable) and name != '_Disposable__dbg_parent':
                value.__dbg_parent = self
                value.__dbg_name = name
            super().__setattr__(name, value)

    def __del__(self):
        if self.__disposed == 0:
            if DEBUG:
                name = deque()
                obj = self
                while obj is not None:
                    name.append(f'{obj}{obj.__dbg_name}')
                    obj = obj.__dbg_parent
                name = '/'.join(name)
            else:
                name = self

            print(f'WARNING. {name}.dispose() was not called.')

    def __repr__(self): return self.__str__()
    def __str__(self): return f"[{self.__class__.__qualname__}]{'[DISPOSED]' if self.__disposed == 2 else '[DISPOSING]' if self.__disposed == 1 else ''}"

    _indent = 0

class CallOnDispose(Disposable):
    """"""
    def __init__(self, func : Callable = None):
        super().__init__()
        self._func = func

    def __dispose__(self):
        if self._func is not None:
            self._func()
            self._func = None
        super().__dispose__()
