from collections import deque
from typing import Callable, Generic, TypeVar

from .Disposable import Disposable
from .EventConnection import EventConnection

A1 = TypeVar('A1')
A2 = TypeVar('A2')
A3 = TypeVar('A3')

class IEvent_rv:
    """read-only view interface of any Event(N)"""
    def has_listeners(self) -> bool: ...
    
    def listen(self, func : Callable) -> EventConnection: ...

class IEvent0_rv(IEvent_rv):
    """read-only view interface of Event0"""
    def listen(self, func : Callable[[], None]) -> EventConnection: ...

class IEvent1_rv(IEvent_rv, Generic[A1]):
    """read-only view interface of Event1"""
    def listen(self, func : Callable[ [A1], None]) -> EventConnection: ...

class IEvent2_rv(IEvent_rv, Generic[A1, A2]):
    """read-only view interface of Event2"""
    def listen(self, func : Callable[ [A1, A2], None]) -> EventConnection: ...

class IEvent3_rv(IEvent_rv, Generic[A1, A2, A3]):
    """read-only view interface of Event3"""
    def listen(self, func : Callable[ [A1, A2, A3], None]) -> EventConnection: ...

class IEvent_v(IEvent_rv):
    """view interface of any EventN"""
    def emit(self, *args, reverse=False): ...

class IEvent0_v(IEvent0_rv):
    """view interface of Event0"""
    def emit(self, *args, reverse=False): ...

class IEvent1_v(IEvent1_rv[A1]):
    """view interface of Event1"""
    def emit(self, a1 : A1, reverse=False): ...

class IEvent2_v(IEvent2_rv[A1, A2]):
    """view interface of Event2"""
    def emit(self, a1 : A1, a2 : A2, reverse=False): ...

class IEvent3_v(IEvent3_rv[A1, A2, A3]):
    """view interface of Event3"""
    def emit(self, a1 : A1, a2 : A2, a3 : A3, reverse=False): ...

class Event(Disposable, IEvent_v):
    def __init__(self):
        super().__init__()
        self._conns = deque()
        
    def has_listeners(self) -> bool: return len(self._conns) != 0

    def listen(self, func : Callable) -> EventConnection:
        conn = EventConnection(func, self._disconnect).dispose_with(self)
        self._conns.append(conn)
        return conn

    def emit(self, *args, reverse=False):
        for conn in reversed(tuple(self._conns)) if reverse else tuple(self._conns):
            conn.emit(*args)
        return self

    def _disconnect(self, conn : EventConnection):
        try:
            self._conns.remove(conn)
        except:
            ...

    def __repr__(self): return self.__str__()
    def __str__(self): return f'[{self.__class__.__name__}][Conns:{len(self._conns)}]'

class Event0(Event, IEvent0_v):
    def listen(self, func : Callable[[], None]) -> EventConnection: return super().listen(func)
    def emit(self, reverse=False): return super().emit(reverse=reverse)

class Event1(Event, IEvent1_v[A1]):
    def listen(self, func : Callable[ [A1], None]) -> EventConnection: return super().listen(func)
    def emit(self, a1 : A1, reverse=False): return super().emit(a1, reverse=reverse)

class Event2(Event, IEvent2_v[A1, A2]):
    def listen(self, func : Callable[ [A1, A2], None]) -> EventConnection: return super().listen(func)
    def emit(self, a1 : A1, a2 : A2, reverse=False): return super().emit(a1, a2, reverse=reverse)

class Event3(Event, IEvent3_v[A1, A2, A3]):
    def listen(self, func : Callable[ [A1, A2, A3], None]) -> EventConnection: return super().listen(func)
    def emit(self, a1 : A1, a2 : A2, a3 : A3, reverse=False): return super().emit(a1, a2, a3, reverse=reverse)



class IReplayEvent_rv:
    """read-only view interface of any ReplayEvent(N)"""
    def listen(self, func : Callable, replay : bool) -> EventConnection: ...

class IReplayEvent1_rv(IReplayEvent_rv, Generic[A1]):
    """read-only view interface of any ReplayEvent1"""
    def listen(self, func : Callable[[A1], None], replay : bool) -> EventConnection:  ...

class IReplayEvent_v(IReplayEvent_rv):
    """view interface of any ReplayEvent"""
    def emit(self, *args, reverse=False): ...

class IReplayEvent1_v(IReplayEvent1_rv[A1]):
    """view interface of any ReplayEvent1"""
    def emit(self, a1 : A1, reverse=False): ...

class ReplayEvent(Disposable, IReplayEvent_v):
    def __init__(self, replayer : Callable = None):
        super().__init__()
        self._replayer = replayer
        self._ev = Event().dispose_with(self)

    def listen(self, func : Callable, replay : bool) -> EventConnection:
        conn = self._ev.listen(func)
        if replay and (replayer := self._replayer) is not None:
            replayer(conn)
        return conn

    def emit(self, *args, reverse=False):
        return self._ev.emit(*args, reverse=reverse)


class ReplayEvent1(ReplayEvent, IReplayEvent1_v[A1]):
    def __init__(self, replayer : Callable[[EventConnection], None] = None):
        """
            replayer   func(conn : EventConnection)
        """
        super().__init__(replayer)

    def listen(self, func : Callable[[A1], None], replay : bool) -> EventConnection:
        return super().listen(func, replay=replay)

    def emit(self, a1 : A1, reverse=False):
        return super().emit(a1, reverse=reverse)


