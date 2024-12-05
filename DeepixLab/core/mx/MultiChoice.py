from __future__ import annotations

from typing import Callable, Self, Sequence, TypeVar

from .Property import (EvaluableProperty, IEvaluableProperty_rv, IProperty_v,
                       Property)

T = TypeVar('T')

class IMultiChoice_v(IProperty_v[ Sequence[T] ]):
    """view interface of MultiChoice"""
    @property
    def mx_avail(self) -> IEvaluableProperty_rv[Sequence[T]]:  ...

    def update(self): ...
    def update_added(self, v): ...
    def update_removed(self, v): ...

class MultiChoice(Property[ Sequence[T] ], IMultiChoice_v):
    """
    MultiChoice is Property[ Sequence[T] ] that is filtered by dynamic unique .mx_avail values and optional your own filter
    """
    def __init__(self,  availuator : Callable[[], Sequence[T]],
                        filter : Callable[[ Sequence[T], Sequence[T] ], Sequence[T] ] = None,
                        defer : Callable[ [ Sequence[T], Sequence[T], MultiChoice[T]], None ] = None,
                 ):
        """
        
        ```
            availuator      callable must return sequence of available T values. Can be zero length.
        ```
        
        Initial value is set to [] immediatelly and filtered.
        """
        self.__filter = filter
        super().__init__([], filter=self.__filter_func, defer=defer)
        
        e_avail = self.__e_avail = EvaluableProperty[Sequence[T]](availuator).dispose_with(self)
        e_avail.reflect(self._on_e_avail)
    
    @property
    def mx_avail(self) -> EvaluableProperty[Sequence[T]]:
        """EvaluableProperty of available values for MultiChoice"""
        return self.__e_avail
    
    def reevaluate(self) -> Self:
        """reevaluate avail items and re-set `set(get())` current value in order to validate it"""
        return self.__e_avail.reevaluate()
    
    def _on_e_avail(self, avail : Sequence[T]):
        # Avail is changed
        # Re-set current value to be filtered and notify listeners
        self.set(self.get())
        

    def update_added(self, v):
        """"""
        self.set(self.get() + (v,))
        return self

    def update_removed(self, v):
        """"""
        self.set(tuple(x for x in self.get() if x != v) )
        return self
    
    def update(self): 
        """same as .set(.get())"""
        self.set(self.get())
        return self
        
    def __filter_func(self, new_value : Sequence[T], value : Sequence[T]):
        avail = self.__e_avail.get()
        
        # Avail-filter and remove duplicates without sorting
        new_value = tuple(dict.fromkeys(x for x in new_value if x in avail))

        if self.__filter is not None:
            new_value = self.__filter(new_value, value)
        return new_value

