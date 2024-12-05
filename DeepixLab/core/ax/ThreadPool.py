import multiprocessing
from enum import Enum, auto

from .. import mx
from .Thread import Thread

CPU_COUNT = multiprocessing.cpu_count()

class ThreadPool(mx.Disposable):

    class BalanceType(Enum):
        TaskCount = auto()
        Queue = auto()

    def __init__(self, count : int = None, balance_type : BalanceType = BalanceType.TaskCount, max_task_per_thread = None, name : str = None):
        """
            count(None)     None = cpu count.

            max_task_per_thread(None)   None: if no free threads, switching task will wait
        """

        super().__init__()
        self._count = count = max(1, count if count is not None else CPU_COUNT)
        self._balance_type = balance_type
        self._max_task_per_thread = max_task_per_thread

        self._counter = 0
        self._threads = [ Thread(f'{name}_{i}' if name is not None else None).dispose_with(self) for i in range(count)]

    @property
    def count(self) -> int:
        return self._count

    def _get_next_thread(self) -> Thread|None:
        """

        returns None if no Thread available right now
        """
        if self._balance_type == ThreadPool.BalanceType.TaskCount:
            t = sorted(self._threads, key=lambda elem: elem.active_task_count)[0]
        elif self._balance_type == ThreadPool.BalanceType.Queue:
            c = self._counter
            self._counter += 1
            t = self._threads[c % len(self._threads)]

        if (max_task_per_thread := self._max_task_per_thread) is not None:
            if t.active_task_count >= max_task_per_thread:
                return None
        return t


