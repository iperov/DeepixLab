from __future__ import annotations

import threading
import traceback
from datetime import datetime
from types import GeneratorType
from typing import (TYPE_CHECKING, Callable, Dict, Iterable, ParamSpec, Set,
                    Tuple, TypeVar)

from .Future import Future
from .g_debug import g_debug
from .g_log import g_log
from .Thread import Thread, get_current_thread
from .ThreadPool import ThreadPool

if TYPE_CHECKING:
    from .FutureGroup import FutureGroup

#T = TypeVar('T')
#P = ParamSpec('P')
class Task[T](Future[T]):
    """
    Task is Future which result is set by result of generator execution.
    """

    def __init__(self, name : str = None):
        super().__init__()
        self._name = name

        self._child_tasks : Set[Task] = set()
        #^ added - inside Task._lock and Task execution only
        #  remove - no lock

        self._creation_time = datetime.now().timestamp()

        # Execution variables
        self._gen_tms : Tuple[Dict, threading.RLock] = None
        self._gen = None
        self._gen_next = True
        self._gen_thread = current_thread = get_current_thread()
        self._gen_yield = None

        exec_stack = current_thread.get_tls()._task_exec_stack
        if len(exec_stack) != 0:
            # Task created inside execution of other Task in current thread
            # connect parent-child
            parent_task = self._parent = exec_stack[-1]
            parent_task._child_tasks.add(self)
        else:
            self._parent : Task = None

        g_debug._on_task_created(self)

        if g_log.get_level() >= 2:
            print(f"{('Starting'):12} {self}")

    @property
    def name(self) -> str: return self._name

    @property
    def creation_time(self) -> float: return self._creation_time

    @property
    def alive_time(self) -> float: return datetime.now().timestamp() - self._creation_time

    def attach_to(self, fg : FutureGroup, max_tasks : int = None, detach_parent : bool = True) -> bool:
        """
        Try to attach to FutureGroup.

            max_tasks(None)     return false if FutureGroup.count >= max_tasks.

            detach_parent(True)     Task will be detached from current parent task.
        """
        with self._lock:
            with fg._lock:
                if max_tasks is not None:
                    if fg.count >= max_tasks:
                        return False

                if (result := super().attach_to(fg)):
                    if detach_parent:
                        if self._parent is not None:
                            self._parent._child_tasks.remove(self)
                            self._parent = None
                    if g_log.get_level() >= 2:
                        print(f"{('Attached'):12} {self} to {fg}")
                return result


    def detach(self):
        """Detach Task from all FutureGroup's or from parent task."""
        with self._lock:
            super().detach()
            if self._parent is not None:
                self._parent._child_tasks.remove(self)
                self._parent = None

    def wait(self):
        """
        Block execution and wait Task in current (or automatically registered) ax.Thread

        raises Exception if calling wait() inside Task.
        """
        if get_current_task() != None:
            raise Exception('Unable to .wait() inside Task. Use yield ax.wait(task)')

        get_current_thread().execute_tasks_loop(exit_if=lambda: self.finished)
        return self

    def _finish(self, success: bool, result=None, error: Exception = None):
        if get_current_task() is self:
            raise Exception('Unable to finish the Task while Thread is executing it.')

        super()._finish(success, result, error)

    def _on_finish(self):
        if g_log.get_level() >= 2:
            child_tasks_len = len(self._child_tasks)
            if child_tasks_len != 0:
                print(f"{('Finishing'):12} {self} Child tasks:[{child_tasks_len}]")
            else:
                print(f"{('Finishing'):12} {self}")

        if self._gen is not None:
            try:
                self._gen.throw( TaskFinishException(self, error=self.error if not self.succeeded else None) )
            except Exception as e:
                ...
                
            self._gen.close()
            self._gen = None

        for child_task in tuple(self._child_tasks):
            child_task.cancel()

        if self._parent is not None:
            self._parent._child_tasks.remove(self)
            self._parent = None

        super()._on_finish()

        g_debug._on_task_finished(self)

        if g_log.get_level() >= 2:
            print(f"{('Finish'):12} {self}")


    def _exec(self):
        while True:
            if self._lock.acquire(timeout=0.005):
                break
            if self._state != 0:
                # Task not in active state
                # may be in cancelling state in other Thread
                # don't wait and return
                return

        # Acquired lock
        if self._state == 0:
            # Task in active state
            exec_stack = get_current_thread().get_tls()._task_exec_stack

            # Execute Task until interruption yield-command
            while self._state == 0:

                if self._gen_next:
                    self._gen_next = False

                    # add Task to ThreadLocalStorage execution stack
                    exec_stack.append(self)
                    try:
                        self._gen_yield = next(self._gen)
                    except StopIteration as e:
                        # Method returns value directly
                        exec_stack.pop()
                        self.success(e.value)
                        break
                    except Exception as e:
                        # Unhandled exception
                        if g_log.get_level() >= 1:
                            print(f'Unhandled exception {e} occured during execution of task {self}. Traceback:\n{traceback.format_exc()}')
                        exec_stack.pop()
                        self.cancel(error=e)
                        break
                    exec_stack.pop()

                else:
                    # Process yield value

                    gen_yield = self._gen_yield
                    gen_yield_t = type(gen_yield)

                    if gen_yield_t is wait:
                        if gen_yield.is_finished():
                            self._gen_next = True

                    elif gen_yield_t is switch_to:
                        thread = gen_yield._thread_or_pool

                        if isinstance(thread, ThreadPool):
                            thread = thread._get_next_thread()

                        if thread is not None:
                            gen_yield._thread_or_pool = thread

                            if self._gen_thread.ident == thread.ident:
                                self._gen_next = True
                            else:
                                self._gen_thread = thread

                    elif gen_yield_t is sleep:
                        if gen_yield._sec > 0:
                            if gen_yield.is_finished():
                                self._gen_next = True
                        else:
                            gen_yield._ticks += 1
                            if gen_yield._ticks > 1:
                                self._gen_next = True

                    elif gen_yield_t is attach_to:
                        if gen_yield._cancel_all:
                            gen_yield._fg.cancel_all()

                        if self.attach_to(gen_yield._fg, max_tasks=gen_yield._max_tasks, detach_parent=gen_yield._detach_parent):
                            self._gen_next = True
                        else:
                            self.cancel()
                            break

                        # elif gen_yield_t is time_barrier:
                        #     tms, tms_lock = self._gen_tms

                        #     key = (time_barrier, gen_yield._key)

                        #     if (tb := tms.get(key, None)) is None:
                        #         with tms_lock:
                        #             if (tb := tms.get(key, None)) is None:
                        #                 tb = tms[key] = TimeBarrier()


                        #     if (r := tb.try_pass(gen_yield._interval, gen_yield._max_task)) >= 0:
                        #         self._gen_yield = sleep(r)
                        #     else:
                        #         self.cancel()
                        #         break


                        # elif gen_yield_t is bottleneck:
                        #     tms, tms_lock = self._gen_tms

                        #     key = (bottleneck, gen_yield._key)

                        #     if (cb := tms.get(key, None)) is None:
                        #         with tms_lock:
                        #             if (cb := tms.get(key, None)) is None:
                        #                 cb = tms[key] = Bottleneck()

                        #     if cb.try_add(self, gen_yield._max_task):
                        #         self.call_on_finish(lambda _: cb.remove(self))

                        #         self._gen_next = True
                        #     else:
                        #         if gen_yield._cancel:
                        #             self.cancel()
                        #             break



                    elif gen_yield_t is cancel:
                        self.cancel(gen_yield._error)
                        break

                    elif gen_yield_t is success:
                        self.success(result=gen_yield._result)
                        break

                    elif gen_yield_t is detach:
                        self.detach()
                        self._gen_next = True

                    elif gen_yield_t is propagate:
                        gen_yield._future.call_on_finish(lambda other_future: (
                                                            self.success(other_future.result) if other_future.succeeded else
                                                            self.cancel(error=other_future.error)))
                        break

                    else:
                        print(f'{self} Unknown type of yield value: {gen_yield}')
                        self.cancel()
                        break

                    if not self._gen_next:
                        if not self._gen_thread._add_task(self):
                            self.cancel()
                        break
        self._lock.release()

    def __repr__(self): return self.__str__()
    def __str__(self):
        s = f'[Task][{self.name}]'

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


def get_current_task() -> Task|None:
    current_thread = get_current_thread()
    tls = current_thread.get_tls()
    if len(tls._task_exec_stack) != 0:
        return tls._task_exec_stack[-1]
    return None

class wait:
    """Stop execution until future_or_list will be entered to finished state."""
    def __init__(self, future_or_list : Future|Iterable[Future]):
        if not isinstance(future_or_list, Iterable):
            future_or_list = (future_or_list,)

        self._lock = threading.Lock()
        self._count = len(future_or_list)

        for task in future_or_list:
            task.call_on_finish(self._on_future_finish)

    def _on_future_finish(self, _):
        with self._lock:
            self._count -= 1

    def is_finished(self):
        return self._count == 0

class switch_to:
    """
    Switch thread of this Task.
    If Task already in Thread, execution will continue immediately.
    If Thread is disposed, Task will be cancelled.
    """
    def __init__(self, thread_or_pool : Thread|ThreadPool):
        self._thread_or_pool = thread_or_pool

class sleep:
    """
    Sleep execution of this Task.

    `sleep(0)` sleep single tick, i.e. minimum possible amount of time between two executions of task loop.
    """
    def __init__(self, sec : float):
        self._time = datetime.now().timestamp()
        self._sec = sec
        self._ticks = 0

    def is_finished(self):
        return (datetime.now().timestamp() - self._time) >= self._sec

class attach_to:
    """```
    Attach Task to FutureGroup.
    If FutureGroup is disposed, current Task will be cancelled without exception immediatelly.

        cancel_all(False)    cancel all tasks from FutureGroup before addition.

        max_tasks(None)     cancel if FutureGroup.count >= max_tasks.
                            Thread-safe check+attach.

        detach_parent(True)     detach from parent Task, thus if parent Task is cancelled,
                                this Task will not be cancelled.

    ```"""
    def __init__(self, fg : FutureGroup, cancel_all : bool = False, max_tasks : int = None, detach_parent : bool = True):
        self._fg = fg
        self._max_tasks = max_tasks
        self._cancel_all = cancel_all
        self._detach_parent = detach_parent

class success:
    """
    Same as `return <result>`, but:
    yield.success inside `try except ax.TaskFinishException`: will be caught.

    All tasks created during execution of this task and if they are not in FutureGroup's (child tasks) - will be cancelled.
    """
    def __init__(self, result = None):
        self._result = result

class cancel:
    """
    Finish task execution and mark this Task as cancelled with optional error.

    yield.cancel inside `try except ax.TaskFinishException`: will be caught.

    All tasks created during execution of this task and if they are not in FutureGroup's (child tasks) - will be cancelled.
    """
    def __init__(self, error : Exception = None):
        self._error = error

class propagate:
    """Wait Future and returns it's result as result of this Task"""
    def __init__(self, future : Future):
        self._future = future

class detach:
    """Detach Task from all FutureGroup's or from parent task."""

class TaskFinishException(Exception):
    """
    an Exception inside ax.task to catch finish event, either success or cancel.

    The current thread in which finish was invoked.

    After this exception any `yield` will stop execution and will have no effect.
    """
    def __init__(self, task : Task, error : Exception|None ):
        self._task = task
        self._error = error
    
    @property
    def error(self) -> Exception|None: return self._error

    def __str__(self): return f'TaskFinishException of {self._task}'
    def __repr__(self): return self.__str__()

def task[T, **P](func : Callable[P, T]) -> Callable[P, Task[T] ]:
    """decorator.

    Turns func to ax.Task.

    Decorated func always returns Task object with type hint of func return value.

    available yields inside task:

    ```
        yield ax.wait

        yield ax.sleep

        yield ax.switch_to

        yield ax.attach_to

        yield ax.try_pass

        yield ax.success

        yield ax.cancel

        yield ax.propagate

        yield ax.detach
    ```

    """
    tms = {}
    lock = threading.RLock()
    
    def wrapper(*args : P.args, **kwargs : P.kwargs) -> Task[T]:
        task = Task(name=f'{func.__qualname__}')

        if isinstance(ret := func(*args, **kwargs), GeneratorType):
            task._gen_tms = (tms, lock)
            task._gen = ret
            task._exec()
        else:
            task.success(ret)

        return task

    return wrapper

