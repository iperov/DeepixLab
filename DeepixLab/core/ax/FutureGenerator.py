from __future__ import annotations

import threading
from collections import deque
from typing import Deque, Generic, Iterable, Iterator, Tuple, TypeVar

from .Future import Future
from .Task import sleep, switch_to, task
from .Thread import get_current_thread

T = TypeVar('T')
_UserParam = TypeVar('_UserParam')

class FutureGenerator(Generic[T, _UserParam], 
                      Iterable[ Tuple[Future[T], _UserParam]|None ]):
    """

    """

    def __init__(self,  fut_gen : Iterable[Tuple[Future[T], _UserParam] | 
                                           None ] ,
                        max_parallel : int,
                        max_buffer : int = 0,
                         
                        ordered=True,
                          ): #
        """
        FutureGenerator

        generates futures from your generator either limited or unlimited,

        keeps `max_parallel` working futures,

        pauses if `max_buffer` reached,

        stores finished in a queue you can get from.
        
        Background task is created in the same thread as FutureGenerator and depended on parent task as usual.
        

            `fut_gen`   your iterable or coroutine that generates a futures.
                        You have to return `(Future, userparam)`
                        You can return `None` if you have no Future right now,
                        the Iterator will not be stopped in this case.


            `max_parallel`      maximum simultaneously working futures.
                                Will not be larger than `max_buffer`

            `max_buffer`(0)     reach `max_buffer` of generated futures (finished or working) and pause generating

                                until user .get() finished futures

                                0 - no limit
            
            `ordered`       You get the completed Future in the same order in which they were created.
                            Otherwise - first one completed
        """
        super().__init__()
        
        if isinstance(fut_gen, Iterable):
            self._fut_iter = iter(fut_gen)
            self._fut_gen = None
        else:
            raise ValueError('Wrong fut_gen type')

        self._max_parallel = max_parallel
        self._max_buffer = max_buffer
        self._ordered = ordered
        
        self._main_thread = get_current_thread()
        self._lock = threading.Lock()
        self._parallel_count = 0
        self._finished = False
        self._buffer : Deque[ Tuple[Future[T], _UserParam] ] = deque()
        
        self._bg_task()
        
    def _inc_parallel(self): 
        with self._lock: self._parallel_count += 1
    def _dec_parallel(self):
        with self._lock: self._parallel_count -= 1
    
    
            
    @task
    def _bg_task(self):
        yield switch_to(self._main_thread)
        
        while not self.finished:
            while self._parallel_count < self._max_parallel and \
                  (self._max_buffer == 0 or len(self._buffer) < self._max_buffer):
                try:
                    v = next(self._fut_iter)
                except StopIteration:
                    self._finished = True
                    break
                    
                if v is not None:
                    fut = v[0]

                    if isinstance(fut, Future):
                        self._inc_parallel()
                        if self._ordered:
                            self._buffer.append(v)
                        else:
                            fut.call_on_finish(lambda *_, v=v: self._buffer.append(v))
                        fut.call_on_finish(lambda *_: self._dec_parallel())
                    else:
                        raise ValueError(f'Future generator returns wrong value')
                else:
                    break
            
                
            yield sleep(0)
            
    @property
    def finished(self) -> bool:
        return self._finished and self._parallel_count == 0 and len(self._buffer) == 0
            
    def next(self) -> Tuple[Future[T], _UserParam]|None:
        """
        return `(Future, user_param)` if currently avail otherwise `None`
        ```
        ordered==True:
        
        [0][1][2][3]
               ^ - finished
         ^ waiting first finished even if others are finished
        ```
        """
        buffer = self._buffer
        if len(buffer) != 0:
            fut = buffer[0][0]
            if fut.finished:
                return buffer.popleft()
        return None
          
    # Iterable            
    def __iter__(self) -> Iterator[ Tuple[Future[T], _UserParam]|None ]:
        while not self.finished:
            yield self.next()


