from __future__ import annotations

from typing import Callable, Generic, TypeVar

from .Disposable import Disposable
from .Event import Event2, IEvent2_rv
from .EventConnection import EventConnection

T = TypeVar('T')

class IState_rv(Generic[T]):
    """read-only view interface of State"""
    
    @property
    def event(self) -> IEvent2_rv[T, bool]: ...

    def listen(self, func : Callable[[T, bool], None]) -> EventConnection: ...
    def reflect(self, func : Callable[[T, bool], None]) -> EventConnection: ...
    def get(self) -> T|None: ...


class IState_v(IState_rv[T]):
    """view interface of State"""
    def set(self, state : T|None): ...
    def swap(self, state : T|None) -> T|None: ...


class State(Disposable, IState_v[T]):
    """
    State is like a Property but has different behaviour designed for creation/disposition logic.

        func(state : T, enter : bool)

    """
    
    

    def __init__(self, filter : Callable[ [T, T], T ] = None):
        """
            filter(new_value, value) -> new_value
                        optional. 
                        Can be used to filter/preprocess/discard incoming value before it will be checked against current value.
                        For example you can replace all spaces to _ in a string, 
                        or you can discard value by returning None(undefined state)
                        In this case listeners will receive only exiting state event.
                                                
                        
        State is created with None(undefined state) value.
        
        Reflecting None(undefined state) will fire no event.
        """
        super().__init__()
        self.__state = None
        self.__filter = filter
        self.__ev = Event2[T, bool]().dispose_with(self)
        
    @property
    def event(self) -> IEvent2_rv[T, bool]: return self.__ev

    def listen(self, func : Callable[[T, bool], None]) -> EventConnection:
        return self.__ev.listen(func)

    def reflect(self, func : Callable[[T, bool], None]) -> EventConnection:
        conn = self.__ev.listen(func)
        if self.__state is not None:
            conn.emit(self.__state, True)
        return conn
    
    def swap(self, state : T|None) -> T|None:
        r = self.get()
        self.set(state)
        return r
        
    def get(self) -> T|None: 
        """returns state value or None(undefined state)"""
        return self.__state

    def set(self, state : T|None):
        """set state value or None(undefined state)"""
        if self.__filter is not None:
            state = self.__filter(state, self.__state)
            
        
        if self.__state != state: 
            if self.__state is not None:
                self.__ev.emit(self.__state, False, reverse=True)
            
            self.__state = state
            
            if state is not None:
                self.__ev.emit(state, True)
                
        return self
