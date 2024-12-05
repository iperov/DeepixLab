from typing import Callable, Self, Sequence, TypeVar

from .Property import EvaluableProperty, IEvaluableProperty_rv
from .State import IState_v, State

T = TypeVar('T')

class IStateChoice_v(IState_v[T]):
    """view interface of StateChoice"""
    @property
    def mx_avail(self) -> IEvaluableProperty_rv[Sequence[T]]:  ...


class StateChoice(State[T], IStateChoice_v):
    """
    StateChoice is State[T] where the value is chosen from dynamic .mx_avail values
    """
    def __init__(self,  availuator : Callable[[], Sequence[T]],
                        filter : Callable[[T, T], T] = None
                 ):
        """
        ```
            availuator      callable must return sequence of available T values.
                            Don't use None value in sequence.
                            Can return zero length sequence. In this case StateChoice will stay in None(undefined state)

            filter          optional.
                            Can be used to filter/preprocess/discard incoming value before it will be checked against current value.
                            For example you can replace all spaces to _ in a string,
                            or you can discard value by returning None(undefined state)
                            In this case listeners will receive only 'exit' state event.
        ```

        State is created with None(undefined state) value.
        """
        super().__init__(filter=self.__filter_func)
        self.__filter = filter

        e_avail = self.__e_avail = EvaluableProperty[Sequence[T]](availuator,
                                                                  # Filter removes duplicates without sorting
                                                                  filter=lambda avail: tuple(dict.fromkeys(avail)) ).dispose_with(self)
        e_avail.listen(self._on_e_avail)



    @property
    def mx_avail(self) -> EvaluableProperty[Sequence[T]]:
        """EvaluableProperty of available values for StateChoice"""
        return self.__e_avail

    def _set_next(self, diff : int):
        avail = self.mx_avail.get()

        value = self.get()
        if value is not None:
            value = avail[(avail.index(value) + diff) % len(avail)]
        else:
            value = avail[0] if len(avail) != 0 else None

        self.set(value)
        return self

    def set_prev(self): return self._set_next(-1)
    def set_next(self): return self._set_next(1)

    def set(self, state : T|None, default=None) -> Self:
        """
        set state value or None(undefined state)
        
        if result state is None, try set to `default`
        """ 
        super().set(state)
        if default is not None and self.get() is None:
            super().set(default)
        return self
    
    def reevaluate(self) -> Self:
        """reevaluate avail items and re-set `set(get())` current value in order to validate it"""
        return self.__e_avail.reevaluate()
    
     
    def _on_e_avail(self, avail : Sequence[T]):
        # Avail is changed. Try to re-set current value that will be filtered by avail
        self.set(self.get())

    def __filter_func(self, new_value : T, value : T):
        # Latest evaluated avail values
        avail = self.__e_avail.get()

        if new_value is not None and \
           new_value not in avail:
            # Revert old value
            new_value = value
            if new_value not in avail:
                # Old value also not in avail, set to None(undefined state)
                new_value = None

        if new_value is not None and \
           self.__filter is not None:
            # User filter
            new_value = self.__filter(new_value, value)
        return new_value

