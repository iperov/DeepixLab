from __future__ import annotations

import os
import pickle
from pathlib import Path
from types import NoneType
from typing import (AbstractSet, Callable, ItemsView, Iterator, KeysView,
                    Mapping, MutableMapping, Self, Sequence, ValuesView,
                    overload)

import numpy as np


class FDict(Mapping):
    """
    FDict is functional-like immutable dict, immutable nested containers, immutable values.

    Types of values are restricted and automatically converted to immutable versions.

    Modifying methods return new version of changed dict.
    """
    @staticmethod
    def load(file, Path_func : Callable[ [Path], Path ] = None) -> FDict:
        """raise on error"""
        return FDict(pickle.load(file), Path_func=Path_func)
    @staticmethod
    def loads(buffer, Path_func : Callable[ [Path], Path ] = None) -> FDict:
        """raise on error"""
        return FDict(pickle.loads(buffer), Path_func=Path_func)

    @staticmethod
    def from_file(path : Path, Path_func : Callable[ [Path], Path ] = None) -> FDict:
        """raise on error"""
        with open(path, 'rb') as file:
            return FDict.load(file, Path_func=Path_func)

    def __init__(self, d : HFDict|FDict|Mapping|None = None, **kwargs):
        super().__init__()

        if not kwargs.pop('_checked', False):
            if d is None:
                d = {}
            elif isinstance(d, HFDict):
                d = d._f_dict._d
            elif isinstance(d, FDict):
                d = d._d
            elif isinstance(d, Mapping):
                d = { _checked_key(k) : _immutable_repack(v, **kwargs) for k, v in d.items() }
            else:
                raise ValueError('d must be a FDict|Mapping|None')
        self._d = d

    def dumps(self, Path_func : Callable[ [Path], Path ] = None ) -> bytes:
        """raise on error"""
        return pickle.dumps(_mutable_repack(self, Path_func=Path_func))

    def dump(self, file, Path_func : Callable[ [Path], Path ] = None):
        """raise on error"""
        pickle.dump(_mutable_repack(self, Path_func=Path_func), file)

    def to_dict(self) -> dict:
        """returns a copy as dict and makes all nested dict to be mutable.
        Sequences, sets, values are still immutable."""
        return _mutable_repack(self)

    def dump_to_file(self, path : Path, Path_func : Callable[ [Path], Path ] = None):
        """raise on error"""
        meta_dict_dump = self.dumps(Path_func=Path_func)

        err = None
        file = None
        try:
            file = open(path, 'wb')
            file.write(meta_dict_dump)
            file.flush()
            os.fsync(file.fileno())
        except Exception as e:
            err=e
        finally:
            if file is not None:
                file.close()
                file = None
            if err is not None:
                path.unlink(missing_ok=True)
                raise err


    def remove(self, *key_or_list, silent=True) -> Self:
        """
        returns a new version of updated FDict with removed key.

            silent(True)    if True, doesn't raise KeyError
        """
        d = self._d.copy()
        if silent:
            for key in key_or_list:
                d.pop(key, None)
        else:
            for key in key_or_list:
                d.pop(key)
        return FDict(d)

    def set(self, key, value) -> Self:
        """returns a new version of updated FDict with [key] = value"""
        d = self._d.copy()
        d[_checked_key(key)] = _immutable_repack(value)
        return FDict(d, _checked=True)

    def update(self, m : Mapping) -> Self:
        """returns a new version of updated FDict with m"""
        d = self._d.copy()

        for key, value in m.items():
            d[_checked_key(key)] = _immutable_repack(value)
        return FDict(d, _checked=True)

    def __or__(self, d : Mapping) -> Self:
        return self.update(d)

    # Mapping
    @overload
    def get(self, key): ...
    @overload
    def get(self, key, default): ...
    def get(self, *args, **kwargs): return self._d.get(*args, **kwargs)
    def items(self) -> ItemsView: return self._d.items()
    def keys(self) -> KeysView: return self._d.keys()
    def values(self) -> ValuesView: return self._d.values()
    def __getitem__(self, key): return self._d.__getitem__(key)
    def __contains__(self, key) -> bool: return self._d.__contains__(key)

    # Collection
    def __len__(self) -> int: return self._d.__len__()

    # Iterable
    def __iter__(self) -> Iterator: return self._d.__iter__()

    # Base
    def __eq__(self, other) -> bool:
        if isinstance(other, FDict):
            return self._d is other._d
        return False

    def __str__(self): return self.__repr__()
    def __repr__(self): return self._d.__repr__()

class HFDict(MutableMapping):
    """
    HFDict  is FDict holder. Modifying methods replaces internal FDict.
    """
    def __init__(self, d : HFDict|FDict|Mapping|None=None):
        if isinstance(d, HFDict):
            d = d._d
        else:
            d = FDict(d)

        self._f_dict = d

    def to_f_dict(self) -> FDict:
        return self._f_dict

    def to_dict(self) -> dict:
        """returns a copy of FDict as dict and makes all nested dict to be mutable.
        Sequences, sets, values are still immutable."""
        return self._f_dict.to_dict()

    # MutableMapping
    def clear(self):
        self._f_dict = FDict()

    @overload
    def pop(self, key): ...
    @overload
    def pop(self, key, default): ...
    def pop(self, *args):
        """"""
        d = self._f_dict
        key = args[0]
        if key in d:
            value = d[key]
        else:
            if len(args) == 2:
                value = args[1]
            else:
                raise KeyError(key)

        self._f_dict = d.remove(key)
        return value

    def set(self, key, value) -> Self:
        self._f_dict = self._f_dict.set(key, value)
        return self

    def update(self, m : Mapping) -> Self:
        self._f_dict = self._f_dict.update(m)
        return self

    def __setitem__(self, key, value): self.set(key, value)
    def __delitem__(self, key): self.pop(key)

    # Mapping
    @overload
    def get(self, key): ...
    @overload
    def get(self, key, default): ...
    def get(self, *args): return self._f_dict.get(*args)
    def items(self) -> ItemsView: return self._f_dict.items()
    def keys(self) -> KeysView: return self._f_dict.keys()
    def values(self) -> ValuesView: return self._f_dict.values()
    def __getitem__(self, key): return self._f_dict.__getitem__(key)
    def __contains__(self, key) -> bool: return self._f_dict.__contains__(key)

    # Collection
    def __len__(self) -> int: return self._f_dict.__len__()

    # Iterable
    def __iter__(self) -> Iterator: return self._f_dict.__iter__()

    # Base
    def __eq__(self, other) -> bool:
        if isinstance(other, HFDict):
            return self._f_dict == other._f_dict
        return False

    def __repr__(self): return self._f_dict.__repr__()


class _FSequence(tuple):
    def __new__(self, iterable, **kwargs) -> _FSequence:
        return tuple.__new__(_FSequence, (_immutable_repack(v, **kwargs) for v in iterable) )

class _FSet(frozenset):
    def __new__(self, iterable, **kwargs) -> _FSet:
        return frozenset.__new__(_FSet, (_immutable_repack(v, **kwargs) for v in iterable) )

_base_types = (NoneType, int, float, bool, str, bytes)
_np_types = (np.ndarray, np.generic)
_key_types = (int, float, str, bool)
_state_types = (_FSequence, _FSet, FDict)

def _checked_key(key):
    if not (type(key) in _key_types):
        raise ValueError(f'Key must be one of {_key_types}')
    return key

def _immutable_repack(value, **kwargs):
    value_type = type(value)
    if value_type in _base_types:
        pass
    elif isinstance(value, _np_types):
        if not value.data.readonly:
            value = value.copy()
            value.setflags(write=False)
    elif value_type is bytearray:
        value = bytes(bytearray)
    elif isinstance(value, Path):
        if (Path_func := kwargs.get('Path_func', None)) is not None:
            value = Path_func(value)
    elif isinstance(value, _state_types):
        pass
    elif isinstance(value, Sequence):
        value = _FSequence(value, **kwargs)
    elif isinstance(value, AbstractSet):
        value = _FSet(value, **kwargs)
    elif isinstance(value, Mapping):
        value = FDict(value, **kwargs)
    else:
        raise ValueError(f'Type of value {value_type} is not allowed in state')
    return value


def _mutable_repack(value, **kwargs):
    value_type = type(value)
    if value_type in _base_types:
        pass
    elif isinstance(value, _np_types):
        if not value.data.readonly:
            value = value.copy()
            value.setflags(write=False)
    elif value_type is bytearray:
        value = bytes(bytearray)
    elif isinstance(value, Path):
        if (Path_func := kwargs.get('Path_func', None)) is not None:
            value = Path_func(value)
    elif isinstance(value, Sequence):
        value = tuple( _mutable_repack(v, **kwargs) for v in value )
    elif isinstance(value, AbstractSet):
        value = frozenset( _mutable_repack(v, **kwargs) for v in value )
    elif isinstance(value, Mapping):
        value = { _checked_key(k) : _mutable_repack(v, **kwargs) for k, v in value.items() }
    else:
        raise ValueError(f'Type of value {value_type} is not allowed in state')

    return value
