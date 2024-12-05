import sys
from typing import (Generic, Iterable, Iterator, Self, Sequence, Type, TypeVar,
                    overload)

from .FDict import FDict

T = TypeVar('T')
C = TypeVar('C')

class FList(Sequence, Generic[T]):

    def __init__(self, values : Sequence = ()):
        super().__init__()
        self._values = tuple(values)

    def clone(self) -> Self:
        f = self.__class__.__new__(self.__class__)
        f._values = self._values
        return f

    def append(self, value : T) -> Self:
        self = self.clone()
        self._values = self._values + (value,)
        return self

    def extend(self, values : Iterable[T]) -> Self:
        self = self.clone()
        self._values = self._values + tuple(values)
        return self

    @overload
    def remove(self, value : T) -> Self:
        """raise on error"""
    @overload
    def remove(self, values : Sequence[T]) -> Self:
        """raise on error"""
    def remove(self, *args) -> Self:
        args_len = len(args)
        if args_len == 1:
            arg0 = args[0]
            if isinstance(arg0, Sequence):
                self = self.clone()
                self._values = tuple( x for x in self._values if x not in arg0)
            else:
                self = self.clone()
                l = self._values
                idx = l.index(arg0)
                self._values = l[:idx] + l[idx+1:]
        else:
            raise ValueError()

        return self

    def get_first_by_class(self, cls : Type[C]) -> C|None:
        for x in self._values:
            if isinstance(x, cls):
                return x
        return None

    def get_first_by_class_prio(self, clss : Sequence[Type[C]]) -> C|None:
        for cls in clss:
            if (result := self.get_first_by_class(cls)) is not None:
                return result
        return None

    def get_all_by_class(self, cls : Type[C]) -> Sequence[C]:
        return [x for x in self._values if isinstance(x, cls)]

    def get_state(self) -> FDict:
        raise NotImplementedError()

    # Sequence
    def __getitem__(self, index) -> T:
        return self._values[index]

    def index(self, value: T, start: int = 0, stop: int = sys.maxsize) -> int:
        return self._values.index(value, start, stop)
    def count(self, value: T) -> int: return self._values.count(value)

    # Collection
    def __len__(self) -> int:  return len(self._values)
    # Container
    def __contains__(self, x) -> bool: return x in self._values
    # Iterable
    def __iter__(self) -> Iterator[T]: return self._values.__iter__()

    def __str__(self): return self.__class__.__name__ + ': [' + ','.join(x.__repr__() for x in self._values) + ']'
    def __repr__(self): return self.__str__()


