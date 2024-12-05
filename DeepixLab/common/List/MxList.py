from typing import Sequence

from core import mx


class MxList[T](mx.Disposable):
    """base class for mx lists"""
    def __init__(self):
        super().__init__()
        self.__values = []
        self._mx_added = mx.Event2[int, T]().dispose_with(self)
        self._mx_remove = mx.Event2[int, T]().dispose_with(self)
        self._mx_removed = mx.Event1[T]().dispose_with(self)

    @property
    def mx_added(self) -> mx.IEvent2_rv[int, T]:
        """added value (idx, value)"""
        return self._mx_added

    @property
    def mx_remove(self) -> mx.IEvent2_rv[int, T]:
        """about to remove value (idx, value)"""
        return self._mx_remove

    @property
    def mx_removed(self) -> mx.IEvent1_rv[T]:
        """value has been removed (idx, value)"""
        return self._mx_removed
    @property
    def values(self) -> Sequence[T]: return self.__values

    def append_new(self):
        raise NotImplementedError()

    def append(self, value : T):
        self.__values.append(value)
        idx = len(self.__values)-1

        self._mx_added.emit(idx, value)

    def remove(self, value : T):
        """
        remove first occurence of value
        raise ValueError if value not present
        """
        self.pop(self.__values.index(value))

    def pop(self, index : int):
        """
        raise ValueError if value not present
        """
        value = self.__values[index]
        self._mx_remove.emit(index, value, reverse=True)
        self.__values.pop(index)
        self._mx_removed.emit(value, reverse=True)
