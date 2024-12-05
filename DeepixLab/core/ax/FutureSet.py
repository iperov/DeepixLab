from typing import Generic, Set, TypeVar

from .Future import Future

T = TypeVar('T')

class FutureSet(Generic[T]):
    """
    Simple Set of Futures.

    Non thread-safe.

    Useful when need to check the status of multiple Future periodically in local func.
    """

    def __init__(self):
        self._futs : Set[Future[T]] = set()

    @property
    def count(self) -> int: return len(self._futs)
    @property
    def count_finished(self) -> int: return sum(1 for fut in self._futs if fut.finished)
    @property
    def count_succeeded(self) -> int: return sum(1 for fut in self._futs if fut.finished and fut.succeeded)
    @property
    def count_cancelled(self) -> int: return sum(1 for fut in self._futs if fut.finished and not fut.succeeded)
    @property
    def empty(self) -> bool: return len(self._futs) == 0
     
    
    def add(self, fut : Future[T]):     self._futs.add(fut)
    def remove(self, fut : Future[T]):  self._futs.remove(fut)
    
    def fetch_first(self, finished=None, succeeded=None) -> Future[T]|None:
        for fut in self._futs:
            
            if (finished is None or finished==fut.finished) and \
               (succeeded is None or (fut.finished and succeeded==fut.succeeded)):

                self._futs.remove(fut)
                return fut
        return None                
    
    def fetch(self, finished=None, succeeded=None, max_count=None) -> Set[Future[T]]:
        """
        fetch Future's from FutureSet with specific conditions

            finished(None)  None : don't check
                        True : fut is finished
                        False : fut is not finished

            succeeded(None)  None : don't check
                           True : fut is finished and succeeded
                           False : fut is finished and not succeeded
            
            max_count(None)
            
        if both args None, fetches all futures.
        """
        out_futs = set()

        for fut in self._futs:
            if max_count is not None and len(out_futs) >= max_count:
                break
            
            if (finished is None or finished==fut.finished) and \
               (succeeded is None or (fut.finished and succeeded==fut.succeeded)):
                out_futs.add(fut)

        self._futs.difference_update(out_futs)

        return out_futs

    def cancel_all(self, remove=True):
        """Cancel all current futures in FutureSet"""
        if len(self._futs) != 0:
            for fut in self._futs:
                fut.cancel()
            if remove:
                self._futs = set()
    
    # def _count(self, finished=None, succeeded=None) -> int:
    #     if finished is None and succeeded is None:
    #         return len(self._futs)
        
    #     count = 0
    #     for fut in self._futs:
    #         if (finished is None or finished==fut.finished) and \
    #            (succeeded is None or (fut.finished and succeeded==fut.succeeded)):
    #             count += 1
    #     return count
    
    def __repr__(self): return self.__str__()
    def __str__(self):
        s = "[FutureSet]"
        if self._name is not None:
            s += f"[{self._name}]"
        s += f'[contains {len(self._futs)} futures]'
        return s
