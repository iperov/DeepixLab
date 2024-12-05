from collections import deque
from typing import Callable, Generic, TypeVar

from .Disposable import Disposable

A1 = TypeVar('A1')

class ChainLink(Disposable):
    def __init__(self, func, discon_func):
        super().__init__()
        self._func = func
        self._discon_func = discon_func
        self._enabled = 1

    def __dispose__(self):
        self._discon_func(self)
        self._enabled = 0
        super().__dispose__()

    def disable(self):
        self._enabled -= 1
        return self

    def enable(self):
        self._enabled += 1
        return self

    def forward(self, value):
        if self._enabled >= 1:
            value = self._func(value)
        return value
        
    def __repr__(self): return self.__str__()
    def __str__(self): return f'{super().__str__()}[{self._func.__qualname__}]'

class IChain_v(Generic[A1]):
    def forward(self, initial : A1) -> A1:
        """"""
    def attach(self, func : Callable[[A1], A1] ) -> ChainLink:
        """"""
    
class Chain(Disposable, IChain_v[A1]):
    def __init__(self):
        super().__init__()
        self._links = deque()
        
    def forward(self, initial : A1) -> A1:
        value = initial
        for conn in self._links:
            value = conn.forward(value)
        return value
            
        
    def attach(self, func : Callable[[A1], A1] ) -> ChainLink:
        """"""
        conn = ChainLink(func, self._disconnect).dispose_with(self)
        self._links.append(conn)
        return conn
    
    def _disconnect(self, conn : ChainLink):
        try:
            self._links.remove(conn)
        except:
            ...
            
    def __repr__(self): return self.__str__()
    def __str__(self): return f'[{self.__class__.__name__}][Links:{len(self._links)}]'
