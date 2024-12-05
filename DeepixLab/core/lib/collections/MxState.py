from typing import (Callable, ItemsView, Iterator, KeysView, Mapping,
                    MutableMapping, Self, ValuesView, overload)

from ... import mx
from .FDict import FDict, HFDict


class MxState(mx.Disposable, MutableMapping):
    def __init__(self, d : HFDict|FDict|Mapping|None=None):
        """childable HFDict"""
        super().__init__()
        self._hf_dict = HFDict(d)
        self._ev = mx.Event0().dispose_with(self)

    def to_f_dict(self) -> FDict: return self._hf_dict.to_f_dict()

    def to_dict(self) -> dict:
        """returns a copy of MxState as dict and makes all nested dict to be mutable.
        Sequences, sets, values are still immutable."""
        return self._hf_dict.to_dict()

    # MutableMapping
    def clear(self): self._hf_dict.clear()

    @overload
    def pop(self, key): ...
    @overload
    def pop(self, key, default): ...
    def pop(self, *args):
        self._hf_dict.clear()

    def set(self, key, value) -> Self:
        self._hf_dict.set(key, value)
        return self

    @overload
    def update(self) -> Self:
        """force update this and all child states"""
    @overload
    def update(self, m : Mapping) -> Self:
        """update this by mapping"""
    def update(self, *args) -> Self:
        args_len = len(args)
        if args_len == 0:
            self._ev.emit(reverse=True)
        elif args_len == 1:
            self._hf_dict.update(args[0])
        return self

    def __setitem__(self, key, value): self.set(key, value)
    def __delitem__(self, key): self.pop(key)


    # Mapping
    @overload
    def get(self, key): ...
    @overload
    def get(self, key, default): ...
    def get(self, *args): return self._hf_dict.get(*args)
    def items(self) -> ItemsView: return self._hf_dict.items()
    def keys(self) -> KeysView: return self._hf_dict.keys()
    def values(self) -> ValuesView: return self._hf_dict.values()
    def __getitem__(self, key): return self._hf_dict.__getitem__(key)
    def __contains__(self, key) -> bool: return self._hf_dict.__contains__(key)

    # Collection
    def __len__(self) -> int: return self._hf_dict.__len__()

    # Iterable
    def __iter__(self) -> Iterator: return self._hf_dict.__iter__()

    # Base
    def __hash__(self) -> int: return hash(id(self))
    def __eq__(self, other) -> bool:
        if isinstance(other, MxState):
            return self._hf_dict == other._hf_dict
        return False
    def __str__(self): return self.__repr__()
    def __repr__(self): return self._hf_dict.__repr__()


    def create_child(self, key) -> Self:
        d = self._hf_dict.get(key, FDict())

        child = MxState(d)

        self._ev.listen(lambda: ( child.update(),
                                  self._hf_dict.update({key : child.to_f_dict()}),
                                 ) ).dispose_with(child)

        return child

    def listen(self, func : Callable[ [HFDict], None]) -> mx.EventConnection:
        """
        func called when MxState is required to be updated

        update hfdict in the func
        """
        return self._ev.listen(lambda: func(self._hf_dict) )

