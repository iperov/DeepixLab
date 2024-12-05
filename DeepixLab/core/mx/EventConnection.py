from __future__ import annotations

from typing import ContextManager

from .Disposable import Disposable


class EventConnection(Disposable):
    def __init__(self, func, discon_func):
        super().__init__()
        self._func = func
        self._discon_func = discon_func
        self._enabled = 1

    def __dispose__(self):
        self._discon_func(self)
        self._enabled = 0
        super().__dispose__()

    def disabled_scope(self) -> ContextManager:
        return EventConnection._disabled_context(self)

    def disable(self):
        self._enabled -= 1
        return self

    def enable(self):
        self._enabled += 1
        return self

    def emit(self, *args):
        if self._enabled >= 1:
            self._func(*args)

    def __repr__(self): return self.__str__()
    def __str__(self): return f'{super().__str__()}[{self._func.__qualname__}]'


    class _disabled_context():
        def __init__(self, conn : EventConnection):
            self._conn = conn

        def __enter__(self):
            self._conn._enabled -= 1

        def __exit__(self, *_):
            self._conn._enabled += 1