from __future__ import annotations

from typing import Callable, Generic, Self, TypeVar

from .Disposable import Disposable
from .Event import Event1, IEvent1_rv, IEvent1_v, IEvent_rv
from .EventConnection import EventConnection

T = TypeVar('T')


class IProperty_rv(Generic[T]):
    """read-only view interface of Property"""
    @property
    def event(self) -> IEvent1_rv[T]: ...

    def listen(self, func : Callable[[T], None]) -> EventConnection: ...
    def reflect(self, func : Callable[[T], None]) -> EventConnection: ...
    def get(self) -> T: ...

class IProperty_v(IProperty_rv[T]):
    """view interface of Property"""
    def set(self, value : T): ...

class Property(Disposable, IProperty_v[T]):
    """holds T value. Listenable/reflectable, direct get/set. Optional filter."""
    def __init__(self,  value : T,
                        filter : Callable[ [T, T], T ] = None,
                        defer : Callable[ [T, T, Property[T]], None ] = None,
                    ):
        """
            filter      if specified, filters the value before set and notify event

            defer       if specified, `set()` value is filtered and redirected to your own defer

                        in which you should call (immediately or later) `prop._set(v)` to set the value
        """
        super().__init__()
        self._value = value
        self._filter = filter
        self._defer = defer
        self._ev = Event1[T]().dispose_with(self)

    @property
    def event(self) -> IEvent1_v[T]: return self._ev

    def listen(self, func : Callable[[T], None]) -> EventConnection:
        return self._ev.listen(func)

    def reflect(self, func : Callable[[T], None]) -> EventConnection:
        conn = self._ev.listen(func)
        conn.emit(self._value)
        return conn

    def get(self) -> T:
        return self._value

    def set(self, value : T) -> Self:
        if self._filter is not None:
            value = self._filter(value, self._value)
            
        if (defer := self._defer) is not None:
            defer(value, self._value, self)
        else:
            self._set(value)
        return self
    
    def toggle(self) -> Self:
        self.set(not self.get())
        return self

    def _set(self, value : T) -> Self:
        self._value = value
        self._ev.emit(value)
        return self


class GetSetProperty(Disposable, IProperty_v[T]):
    """
    GetSetProperty built on getter, setter, and optional changed event.
    Does not hold value internally.
    """
    def __init__(self, getter : Callable[ [], T ],
                       setter : Callable[ [T], None ],
                       changed_event : IEvent_rv|None = None):

        super().__init__()
        self._ev = Event1[T]().dispose_with(self)
        self._getter = getter
        self._setter = setter

        if changed_event is not None:
            self._conn = changed_event.listen(lambda *_: self._ev.emit(self._getter())).dispose_with(self)
        else:
            self._conn = None

    @property
    def event(self) -> IEvent1_rv[T]: return self._ev

    def listen(self, func : Callable[[T], None]) -> EventConnection:
        return self._ev.listen(func)

    def reflect(self, func : Callable[[T], None]) -> EventConnection:
        conn = self._ev.listen(func)
        conn.emit(self._getter())
        return conn

    def get(self) -> T: return self._getter()

    def set(self, value : T):
        if (conn := self._conn) is not None:
            with conn.disabled_scope():
                self._setter(value)
        else:
            self._setter(value)
        self._ev.emit(self._getter())


class IEvaluableProperty_rv(IProperty_rv[T]):
    def reevaluate(self) -> T: ...

class EvaluableProperty(Disposable, IEvaluableProperty_rv[T]):
    """
    Evaluable read-only property.
    Holds evaluated T value.
    """

    def __init__(self,  evaluator : Callable[ [], T ],
                        filter : Callable[ [T], T ] = None
                        ):
        super().__init__()
        self.__ev = Event1[T]().dispose_with(self)
        self.__evaluator = evaluator
        self.__filter = filter

        self.reevaluate()

    def reevaluate(self) -> T:
        """
        evaluate latest value. Event listeners will be notified
        returns evaluated value
        """
        value = self.__evaluator()

        if self.__filter is not None:
            value = self.__filter(value)

        self.__value = value
        self.__ev.emit( value )
        return value

    def listen(self, func : Callable[[T], None]) -> EventConnection:
        return self.__ev.listen(func)

    def reflect(self, func : Callable[[T], None]) -> EventConnection:
        conn = self.__ev.listen(func)
        conn.emit( self.__value )
        return conn

    def get(self) -> T:
        """return latest evaluated value"""
        return self.__value
