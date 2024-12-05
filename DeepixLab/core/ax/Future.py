from __future__ import annotations

import threading
from collections import deque
from time import sleep
from typing import (Any, Callable, Deque, Generic, Iterable, Iterator, Self, Set, Tuple,
                    TypeVar)

from .FutureGroup import FutureGroup

_UserParam = TypeVar('_UserParam')

class Future[T]:
    """
    Thread-safe Future object.

    Can be used as result of deferred or async operations.
    """

    def __init__(self):
        super().__init__()

        self._lock = threading.RLock()
        self._state = 0 # 0:active
                        # 1:succeeded
                        # 2:cancelled

        self._fgs : Set[FutureGroup] = set()
        #^ accessed inside Future._lock only

        self.__on_finish_funcs = deque()

    @property
    def finished(self) -> bool:  return self._state > 0

    @property
    def succeeded(self) -> bool:
        """Avail only if .finished"""
        if self._state == 0:
            raise Exception(f'{self} succeeded avail only in finished state.')
        return self._state == 1

    @property
    def result(self) -> T:
        """Avail only if succeeded"""
        if self._state != 1:
            raise Exception(f'{self} result avail only in succeeded state.')
        return self.__result

    @property
    def error(self) -> Exception|None:
        """Avail only if not succeeded"""
        if self._state != 2:
            raise Exception(f'{self} error avail only in non-succeeded state.')
        return self.__error

    def attach_to(self, fg : FutureGroup) -> bool:
        """Try to attach to FutureGroup."""
        with self._lock:
            if self._state == 0:
                if fg not in self._fgs:

                    with fg._lock:
                        fg_futures = fg._futures
                        if fg_futures is not None and self not in fg_futures:
                            fg_futures.add(self)

                            self._fgs.add(fg)
                            return True
        return False

    def detach(self):
        """Detach Future from all FutureGroup's"""
        with self._lock:
            for fg in self._fgs:
                with fg._lock:
                    if (futures := fg._futures) is not None:
                        try:
                            futures.remove(self)
                        except: ...
            self._fgs.clear()


    def call_on_finish(self, func : Callable[ [Future], None ]) -> Self:
        """
        Call func from undetermined thread when Future is finished.
        If Future is already finished, func will be called immediately from caller thread.
        """
        with self._lock:
            if self._state == 0:
                self.__on_finish_funcs.append(func)
                return self
        func(self)
        return self

    def cancel(self, error : Exception|None = None) -> Self:
        """
        Finish Future in blocking mode.
        Stop its execution and set Future to cancelled state with optional error(default None).
        If Future is already finished, nothing will happen.
        
            error(None)     optional Exception
        """
        self._finish(False, error=error)
        return self

    def success(self, result : T = None) -> Self:
        """
        Finish Future, stop its execution and set Future to succeeded state optional result(default None).
        If it is already finished, nothing will happen.
        
            result(None)    optional result
        """
        self._finish(True, result)
        return self

    def wait(self) -> Self:
        """"""
        while not self.finished:
            sleep(0.001)
        return self
    
    def _on_finish(self):
        """inheritable at last"""
        fgs, self._fgs = self._fgs, None
        for fg in fgs:
            try:
                if (futures := fg._futures) is not None:
                    futures.remove(self)
            except:
                ...


    def _finish(self, success : bool, result = None, error : Exception = None):
        if self._state == 0:
            on_finish_funcs = ()

            with self._lock:
                if self._state == 0:
                    if success:
                        self.__result = result
                        self._state = 1
                    else:
                        self.__error = error
                        self._state = 2

                    self._on_finish()

                    on_finish_funcs, self.__on_finish_funcs = self.__on_finish_funcs, None

            for func in on_finish_funcs:
                func(self)


    def __repr__(self): return self.__str__()
    def __str__(self):
        s = f'[Future]'

        if self.finished:
            if self.succeeded:
                s += f'[SUCCEEDED][Result: {type(self.result).__name__}]'
            else:
                s += f'[CANCELLED]'
                error = self.error
                if error is not None:
                    s += f'[Exception:{error}]'
        else:
            s += f'[ACTIVE]'

        return s


# class FutureQueue[T]:
#     """

#     """

#     def __init__(self,  fut_gen :   Iterable[    Tuple[Future[T], Any]|Future[T]|None ] |
#                                     Callable[[], Tuple[Future[T], Any]|Future[T]|None ],
#                         max_parallel : int,
#                         max_buffer : int = 0 ): #
#         """
#         FutureQueue

#         generates futures from your generator either limited or unlimited,

#         keeps `max_parallel` working futures,

#         pauses if `max_buffer` reached,

#         stores finished in a queue that you can get from.

#             `fut_gen`   a Future generator.
#                         You have to return `(Future, userparam)`
#                         You can return `None` if you have no Future right now,
#                         the Iterator will not be stopped in this case.

#                         can be:
#                         `Iterable[ Tuple[Future[T], UserParam] | None]`
#                             any iterable or generator

#                         `Callable[[], Tuple[Future[T], UserParam] | None]`
#                             your own func that can `raise StopIteration` if generation is done

#             `max_parallel`      maximum simultaneously working futures.
#                                 Will not be larger than `max_buffer`

#             `max_buffer`(0)     reach `max_buffer` of generated futures (finished or working) and pause generating

#                                 until user .get() finished futures

#                                 0 - no limit

#         """
#         super().__init__()

#         if isinstance(fut_gen, Iterable):
#             self._fut_iter = iter(fut_gen)
#             self._fut_gen = None
#         elif callable(fut_gen):
#             self._fut_iter = None
#             self._fut_gen = fut_gen
#         else:
#             raise ValueError('Wrong fut_gen type')

#         self._finished = False

#         self._max_parallel = max_parallel
#         self._max_buffer = max_buffer

#         self._lock = lock = threading.Lock()
#         self._parallel_count = parallel_count = [0]

#         self._inc_parallel = lambda *_: (lock.acquire(), parallel_count.append(parallel_count.pop(0)+1), lock.release() )
#         self._dec_parallel = lambda *_: (lock.acquire(), parallel_count.append(parallel_count.pop(0)-1), lock.release() )

#         self._buffer : Deque[ Tuple[Future, Any]|Future ] = deque()

#     def _process(self):
#         if not self._finished:
#             while self._parallel_count[0] < self._max_parallel and (self._max_buffer == 0 or len(self._buffer) < self._max_buffer):
#                 try:
#                     if self._fut_iter is None:
#                         v = self._fut_gen()
#                         if isinstance(v, Iterable):
#                             self._fut_iter = v
#                             self._fut_gen = None
#                             v = next(self._fut_iter)
#                     else:
#                         v = next(self._fut_iter)
                        
#                     if v is not None:
#                         if isinstance(v, Tuple):
#                             fut = v[0]
#                         else:
#                             fut = v

#                         if fut is not None:
#                             if isinstance(fut, Future):
#                                 self._inc_parallel()
#                                 fut.call_on_finish(self._dec_parallel)
#                                 self._buffer.append(v)
#                             else:
#                                 raise ValueError(f'Future generator returns wrong value')
#                         else:
#                             break
#                     else:
#                         break
#                 except StopIteration:
#                     self._finished = True
#                     break
        
#     def empty(self) -> bool:
#         """
#         process inner logic, and indicates that queue currently has no pending futures that can be .get()

#         can raise unhandled error from future_generator
#         """
#         self._process()

#         buffer = self._buffer
        
#         if len(buffer) != 0:
#             if isinstance(buffer[0], Tuple):
#                 fut = buffer[0][0]
#             else:
#                 fut = buffer[0]
#             return not fut.finished
                
#         return True

#     def finished(self) -> bool:
#         """
#         process inner logic, and indicates that queue has no pending futures and new will not be generated

#         can raise unhandled error from future_generator
#         """
#         self._process()

#         return self._finished and len(self._buffer) == 0

#     def get(self) -> Tuple[Future[T], Any]|Future[T]|None:
#         """
#         process inner logic, and return `(Future, user_param)` if currently avail otherwise `(None, None)`

#         can raise unhandled error from future_generator
#         """
#         self._process()

#         buffer = self._buffer
#         if len(buffer) != 0:
#             if isinstance(buffer[0], Tuple):
#                 fut = buffer[0][0]
#             else:
#                 fut = buffer[0]
#             if fut.finished:
#                 return buffer.popleft()

#         return None

# def iter_future_queue[T](   fut_gen :   Iterable[    Tuple[Future[T], Any]|Future[T]|None ] |
#                                         Callable[[], Tuple[Future[T], Any]|Future[T]|None ],
#                             max_parallel : int,
#                             max_buffer : int = 0) -> Iterator[ Tuple[Future[T], Any]|Future[T]|None ]:
#     """
#     iterate over FutureQueue from fut_gen until finish.
#     """
#     q = FutureQueue(fut_gen, max_parallel, max_buffer=max_buffer)
#     while not q.finished():
#         yield q.get()
