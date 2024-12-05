from typing import Callable, Generic, Sequence, TypeVar

from .Disposable import Disposable

T = TypeVar('T')

class IMenu_v(Generic[T]):
    """view interace of Menu"""
    @property
    def avail_choices(self) -> Sequence[T]: ...
    
    def choose(self, choice : T): ...

class Menu(Disposable, IMenu_v[T]):
    def __init__(self,  avail_choices : Callable[ [], Sequence[T] ]|Sequence[T],
                        on_choose : Callable[[T], None] = None,
                 ):
        super().__init__()
        self._avail_choices = avail_choices if callable(avail_choices) else lambda: avail_choices
        self._on_choose = on_choose

    @property
    def avail_choices(self) -> Sequence[T]: return self._avail_choices()

    def choose(self, choice : T):
        avail_choices = tuple(self._avail_choices())

        if choice in avail_choices:
            if (on_choose := self._on_choose) is not None:
                on_choose(choice)
