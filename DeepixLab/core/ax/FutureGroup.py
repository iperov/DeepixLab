import threading
from typing import Generic, Set, TypeVar, TYPE_CHECKING

from .. import mx
from .g_log import g_log

if TYPE_CHECKING:
    from .Future import Future

T = TypeVar('T')

class FutureGroup(mx.Disposable, Generic[T]):

    def __init__(self, name : str = None):
        """
        Provides a set of Futures used to control execution.
        Task detaches from his parent task when added to FutureGroup.
        """
        super().__init__()
        self._name = name
        self._lock = threading.RLock()
        self._futures : Set[Future] = set()

    def __dispose__(self):
        """
        Dispose FutureGroup. Cancel all Futures.
        New Futures that are added to FutureGroup using yield will be automatically cancelled.
        """
        if g_log.get_level() >= 2:
            print(f"{('Dispose'):12} {self}")

        with self._lock:
            futures, self._futures = self._futures, None

        for fut in futures:
            fut.cancel()

        super().__dispose__()

    @property
    def count(self) -> int:
        """Amount of registered futures."""
        futures = self._futures
        return 0 if futures is None else len(futures)

    @property
    def is_empty(self) -> bool: return self.count() == 0

    @property
    def name(self) -> str: return self._name

    def cancel_all(self, error : Exception|None = None):
        """Cancel all current active Futures in FutureGroup."""
        with self._lock:
            futures, self._futures = self._futures, set()

        if g_log.get_level() >= 2 and len(futures) != 0:
            print(f'Cancelling {len(futures)} futures in {self._name} FutureGroup')

        for fut in futures:
            fut.cancel(error=error)


    def __repr__(self): return self.__str__()
    def __str__(self):
        s = '[FutureGroup]'
        if self._name is not None:
            s += f'[{self._name}]'
        fut = self._futures
        if fut is not None:
            s += f'[{len(self._futures)} fut]'
        else:
            s += '[FINALIZED]'
        return s


