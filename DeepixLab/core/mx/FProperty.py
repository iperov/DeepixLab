from __future__ import annotations

from typing import Callable, Generic, TypeVar

from .Disposable import Disposable
from .Event import Event2, IEvent2_rv, IEvent2_v
from .EventConnection import EventConnection

T = TypeVar('T')


class IFProperty_rv(Generic[T]):
    """read-only view interface of FProperty"""
    @property
    def event(self) -> IEvent2_rv[T, T]: ...

    def listen(self, func : Callable[[T, T], None]) -> EventConnection: ...
    def reflect(self, func : Callable[[T, T], None]) -> EventConnection: ...
    def get(self) -> T: ...

class IFProperty_v(IFProperty_rv[T]):
    """view interface of FProperty"""
    def set(self, value : T): ...

class FProperty(Disposable, IFProperty_v[T]):
    """
    property for immutable F-models.
    
    """
    def __init__(self,  value : T,
                        filter : Callable[ [T, T], T ] = None,
                    ):
        """
            filter      if specified, filters the value before set and notify event

        """
        super().__init__()
        self._value = value
        self._filter = filter
        self._ev = Event2[T,T]().dispose_with(self)

    @property
    def event(self) -> IEvent2_v[T]: return self._ev

    def listen(self, func : Callable[[T,T], None]) -> EventConnection:
        return self._ev.listen(func)

    def reflect(self, func : Callable[[T,T], None]) -> EventConnection:
        conn = self._ev.listen(func)
        conn.emit(self._value, self._value)
        return conn

    def get(self) -> T:
        return self._value

    def set(self, value : T):
        if self._filter is not None:
            value = self._filter(value, self._value)
            
        old_value, self._value = self._value, value
        self._ev.emit(value, old_value)
        return self


