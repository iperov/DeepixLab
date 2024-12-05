import itertools
import threading
import time
from collections import deque
from typing import Callable, Dict

from .. import mx
from ..lib import os as lib_os
from .g_log import g_log
from .ThreadLocalStorage import ThreadLocalStorage


class Thread(mx.Disposable):
    Priority = lib_os.ThreadPriority
    
    _by_ident : Dict[int, 'Thread'] = {}
    _unnamed_counter = itertools.count()

    def __init__(self, name : str = None, priority : Priority = None, **kwargs):
        """
        Spawn ax.Thread
        """
        super().__init__()

        self._name = name if name is not None else f'Unnamed #{next(Thread._unnamed_counter)}'
        self._priority = priority
        
        self._spawned = spawn = not kwargs.get('register', False)
        self._lock = threading.Lock()
        self._active_tasks = deque()
        self._active_task_count = 0
        self._call_soon = deque()

        self._disposing_ev = threading.Event()
        self._disposed_ev = threading.Event()

        self._ident = None
        if spawn:
            if g_log.get_level() >= 2:
                print(f"{('Spawning'):12} {self}")

            self._t = threading.Thread(target=self._thread_func, daemon=True)
            self._t.start()
        else:
            self._t = None
            self._initialize_thread(threading.get_ident())

    def __dispose__(self):
        """
        Dispose the Thread.
        All active tasks assigned to the Thread will be cancelled.
        New tasks which are switching to disposed thread will be cancelled immediately.
        """
        self._lock.acquire()

        if not self._disposing_ev.is_set():
            self._disposing_ev.set()
            self._lock.release()

            if g_log.get_level() >= 2:
                print(f"{('Disposing'):12} {self}")

            if self._spawned:
                if threading.get_ident() == self._ident:
                    raise Exception('Unable to dispose spawned Thread while executing in this Thread. Switch to other Thread and then call dispose.')
                else:
                    self._disposed_ev.wait()
            else:
                if threading.get_ident() != self._ident:
                    raise Exception('non-spawned Thread.dispose() must be called from the same OS thread.')
                self._finalize_thread()
        else:
            # Already disposing
            self._lock.release()
            self._disposed_ev.wait()
        super().__dispose__()

    @property
    def ident(self) -> int: return self._ident

    @property
    def is_spawned(self) -> bool:
        """whether Thread is spawned or registered"""
        return self._spawned

    @property
    def active_task_count(self) -> int:
        return self._active_task_count

    def assert_current_thread(self):
        if threading.get_ident() != self._ident:
            raise Exception('Wrong current thread.')

    def call_soon(self, func : Callable) -> mx.Disposable:
        """
        Calls `func` in next tick of this Thread 

        Calling `call_soon()` on disposed Thread will raise error.
        
        return Disposable that can be disposed to prevent func call
        """
        with self._lock:
            if self._call_soon is None:
                raise Exception('Thread is disposed.')
            
            self._call_soon.append(callsoon := _Callsoon(func))
            return callsoon.callable

    def get_active_tasks(self):
        active_tasks = self._active_tasks
        return deque(active_tasks) if active_tasks is not None else deque()

    def get_name(self): return self._name
    def get_tls(self) -> ThreadLocalStorage: return ThreadLocalStorage._by_ident[self._ident]
    def set_name(self, name : str): self._name = name

    def execute_tasks_once(self):
        """
        Execute active tasks once.
        """
        if threading.get_ident() != self._ident:
            raise Exception('execute_tasks_once must be called from OS thread where the Thread was spawned/registered.')

        for callsoon in self._fetch_call_soon_funcs():
            callsoon.call()
            callsoon.dispose()
        
        self._active_task_count = len(self._active_tasks)
        for task in self._fetch_active_tasks():
            task._exec()

    def execute_tasks_loop(self, exit_if : Callable[[], bool] = None, max_time : float = None):
        """
        Execute active tasks in loop until Thread dispose or exit_if() is True

            exit_if(None)     Callable    called every tick once to check if exit from loop is needed.
        """
        perf = time.perf_counter()
        total_time = time.perf_counter()
        while not self._disposing_ev.is_set():
            if exit_if is not None and exit_if():
                break

            if max_time is not None and (time.perf_counter() - total_time) >= max_time:
                break

            # Sleep design principle.
            # Python-code Tasks are tend to be as short as possible.
            # Heavy load tasks are tend to be as few as possible in single thread.
            # Heavy load tasks are tend to be GIL-free, i.e. use numpy/cv and similar libs.
            # Thus.
            # If all tasks worked < 5ms(200 fps), then they don't need to work again, and Thread can sleep to let other Threads work.
            # If Thread has heavy task, it already freed GIL, so we need to sleep only 1ms every sec to process OS events.

            exec_time = time.perf_counter()
            self.execute_tasks_once()
            exec_time = time.perf_counter() - exec_time

            sleep_time = 0
            if exec_time < 0.005:
                # Tasks were working less than 5ms
                # add sleep 5ms diff
                sleep_time += 0.005-exec_time

            if (cur_perf := time.perf_counter()) - perf >= 1.0:
                # Extra 1ms sleep every sec if heavy load task is working.
                perf = cur_perf
                sleep_time += 0.001 

            if sleep_time != 0:
                lib_os.sleep_precise(sleep_time)

    def _thread_func(self):
        self._initialize_thread(threading.get_ident())
        self.execute_tasks_loop()
        self._finalize_thread()

    def _initialize_thread(self, ident):
        if ident in Thread._by_ident:
            raise Exception(f'Thread {ident} is already registered.')

        self._ident = ident
        Thread._by_ident[ident] = self
        ThreadLocalStorage._by_ident[ident] = ThreadLocalStorage()
        
        if self._priority is not None:
            lib_os.set_thread_priority(self._priority)

        if g_log.get_level() >= 2:
            print(f"{('Spawned'):12} {self}")

    def _finalize_thread(self):
        # Func executes in actual Thread.

        for callsoon in self._fetch_call_soon_funcs(dispose=True):
            callsoon.dispose()

        # Cancel remaining tasks registered in thread.
        for task in self._fetch_active_tasks(dispose=True):
            task.cancel()

        Thread._by_ident.pop(self._ident)
        ThreadLocalStorage._by_ident.pop(self._ident)
        self._disposed_ev.set()

        if g_log.get_level() >= 2:
            print(f"{('Disposed'):12} {self}")

    def _add_task(self, task) -> bool:
        with self._lock:
            if self._active_tasks is not None:                
                self._active_tasks.append(task)
                self._active_task_count += 1
                return True
            return False

    def _fetch_active_tasks(self, dispose=False):
        with self._lock:
            tasks = self._active_tasks
            self._active_tasks = None if dispose else deque()
        return tasks

    def _fetch_call_soon_funcs(self, dispose=False):
        with self._lock:
            funcs = self._call_soon
            self._call_soon = None if dispose else deque()
        return funcs

    def get_printable_info(self, include_tasks=False) -> str:
        s = f"{super().__str__()}{'[S]' if self._spawned else '[R]'}[{self._name}][{self._ident if self._ident is not None else '...'}]"

        if include_tasks:
            with self._lock:
                if self._active_tasks is not None:
                    active_tasks = tuple( task for task in self._active_tasks if not task.finished )
                    if len(active_tasks) != 0:
                        s += '\nThread active tasks:'
                        for i, task in enumerate(active_tasks):
                            s += f'\n[{i}]: {task}'
        return s

    def __repr__(self): return self.__str__()
    def __str__(self): return self.get_printable_info()


def create_thread(name : str = None) -> Thread:
    return Thread(name=name, create=True,  check=1)

def get_current_thread() -> Thread|None:
    """get current ax.Thread or register new"""
    thread = Thread._by_ident.get(threading.get_ident(), None)
    if thread is None:
        thread = Thread(register=True)
    return thread



        
class _Callsoon(mx.Disposable):
    """"""
    def __init__(self, func : Callable):
        super().__init__()
        self._callable = _Callsoon._Callable(func).dispose_with(self)
        
        mx.CallOnDispose(lambda: setattr(self, '_callable', None)).dispose_with(self._callable)
        
    @property
    def callable(self) -> mx.Disposable:
        return self._callable
        
    def call(self):
        if (callable := self._callable) is not None:
            callable.call()
            
    class _Callable(mx.Disposable):
        def __init__(self, func):
            super().__init__()
            self._func = func
            
        def call(self):
            self._func()