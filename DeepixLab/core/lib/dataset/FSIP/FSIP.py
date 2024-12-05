from __future__ import annotations

import itertools
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Self, Sequence

from ... import path as lib_path
from ...collections import FIndices, sequence_removed_indices
from ...image import ImageFormatSuffixes


class IFSIP_rv:
    """read-only interface for FSIP"""
    @dataclass
    class OpResult:
        errors : List[Exception]   # error during op

    @dataclass
    class MoveResult(OpResult):
        moved_ids : FIndices

    @dataclass
    class CopyResult(OpResult):
        copied_ids : FIndices

    @dataclass
    class DeleteResult(OpResult):
        deleted_ids : FIndices

    @property
    def root(self) -> Path: ...
    @property
    def info(self) -> FSIPInfo: ...
    @property
    def item_count(self) -> int: ...
    @property
    def pair_types(self) -> Sequence[str]: ...

    def get_item_path(self, item_id : int) -> Path: ...
    def get_pair_path(self, item_id : int, pair_type : str) -> Path|None: ...

    def has_pair_type(self, pair_type : str) -> bool: ...
    def has_pair(self, item_id : int, pair_type : str) -> bool: ...

class IFSIP_v(IFSIP_rv):
    """interface for FSIP"""
    def add_pair_type(self, pair_type : str): ...
    def add_pair_path(self, item_id : int, pair_type : str, pair_suffix : str) -> Path: ...

    def delete_pair(self, item_id : int, pair_type : str): ...
    def delete_pair_type(self, pair_type : str): ...

    def delete_items(self, indices : FIndices|Iterable|int|None) -> IFSIP_rv.DeleteResult: ...
    def copy_items(self, indices : FIndices|Iterable|int|None|Mapping[int, str], fsip_root : Path) -> IFSIP_v.CopyResult: ...
    def move_items(self, indices : FIndices|Iterable|int|Mapping[int, str], fsip_root : Path) -> IFSIP_rv.MoveResult: ...

    def filtered_view(self, indices : FIndices|Iterable|int) -> IFilteredFSIP_v: ...

class IFilteredFSIP_v(IFSIP_v):
    """interface for filtered FSIP"""
    def from_orig_indices(self, indices : FIndices|Iterable|int|None) -> FIndices:
        """transform original indices to filtered"""
    def to_orig_indices(self, indices : FIndices|Iterable|int) -> FIndices:
        """transform filtered indices to original"""

class FSIPInfo:
    """retrieve various information about FSIP directory"""

    def __init__(self, root : Path):
        self._root = root

    @property
    def root(self) -> Path: return self._root

    @property
    def avail_image_suffixes(self) -> Sequence[str]: return ImageFormatSuffixes

    def get_pair_dir_path(self, pair_type : str) -> Path: return self._root / pair_type.lower()

    def load_item_rel_paths(self) -> Sequence[Path]:
        """
        relative to root image paths

        raise on error
        """
        return lib_path.get_files_paths(self._root, extensions=self.avail_image_suffixes, relative=True)

    def load_pair_rel_paths(self, pair_type : str) -> Sequence[Path]:
        """
        relative to pair_type dir image paths

        raise on error
        """
        return lib_path.get_files_paths(self.get_pair_dir_path(pair_type), extensions=self.avail_image_suffixes, relative=True)

    def load_pair_types(self) -> Sequence[str]:
        """
        loads sorted unique sequence of pair_types

        raise on error
        """
        pair_types = set(path.name.lower() for path in lib_path.get_dir_paths(self._root))
        pair_types = tuple(sorted(pair_types))
        return pair_types


class FSIP(IFSIP_v):
    """
    FileSystem-hosted Image Pair dataset

    Files structure:

    root / images...

    root / pairtype1 / images...

    root / pairtype2 / images...
    """


    class Pair:
        def __init__(self, type : str, suffix : str):
            self._type = type
            self._suffix = suffix

        @property
        def type(self) -> str: return self._type
        @property
        def suffix(self) -> str:
            """example '.png' """
            return self._suffix

        def __str__(self): return f"[Pair {self._type} {self._suffix}]"
        def __repr__(self): return self.__str__()


    class Item:
        """Immutable info about FSIP item"""

        def __init__(self, name : str, suffix : str, pairs : Sequence[FSIP.Pair] = ()):
            self._name = name
            self._suffix = suffix
            self._pairs = pairs

        def clone(self) -> Self:
            f = self.__class__.__new__(self.__class__)
            f._name = self._name
            f._suffix = self._suffix
            f._pairs = self._pairs
            return f

        @property
        def name(self) -> str:
            """unique item name"""
            return self._name
        @property
        def suffix(self) -> str:
            """example '.png' """
            return self._suffix
        @property
        def name_suffix(self) -> str:
            return self._name + self._suffix
        @property
        def pairs(self) -> Sequence[FSIP.Pair]:
            """avail pairs for this item"""
            return self._pairs
        def has_pair(self, pair_type : str) -> bool: return self.get_pair(pair_type) is not None
        def get_pair(self, pair_type : str) -> FSIP.Pair|None:
            for pair in self._pairs:
                if pair.type == pair_type:
                    return pair
            return None

        def add_pair(self, pair : FSIP.Pair) -> Self:
            self = self.clone()
            self._pairs = self._pairs + (pair,)
            return self

        def remove_pair(self, pair : FSIP.Pair) -> Self:
            self = self.clone()
            pairs = self._pairs
            idx = pairs.index(pair)
            self._pairs = pairs[:idx] + pairs[idx+1:]
            return self

        def __str__(self): return f"[Item {self._name} {self._suffix} {self._pairs}]"
        def __repr__(self): return self.__str__()

    @staticmethod
    def open(root : Path|str) -> FSIP:
        """
        raise on error
        """
        root = Path(root)

        errs = []

        info = FSIPInfo(root)
        item_rel_paths = info.load_item_rel_paths()

        if len(item_rel_paths) == len(set(x.stem for x in item_rel_paths)):
            # Paths have no duplicates by stem
            items_data = {  image_rel_path.stem :
                                {   'id' : i,
                                    'suffix' : image_rel_path.suffix,
                                    'pairs' : {}, } for i, image_rel_path in enumerate(item_rel_paths) }

            for pair_type in (pair_types := info.load_pair_types()):
                for pair_rel_path in info.load_pair_rel_paths(pair_type):

                    if (item_dict := items_data.get(pair_rel_path.stem, None)) is not None:
                        pairs_dict = item_dict['pairs']

                        if pair_type not in pairs_dict:
                            pairs_dict[pair_type] = pair_rel_path.suffix
                        else:
                            errs.append(f"Duplicate: {pair_type}/{pair_rel_path}")
                    else:
                        errs.append(f"Unused pair: {pair_type}/{pair_rel_path}")

            items = [None]*len(item_rel_paths)
            for item_name, item_data in items_data.items():
                item_id = item_data['id']
                items[item_id] = FSIP.Item(item_name,
                                            item_data['suffix'],
                                            tuple( FSIP.Pair(pair_type, pair_suffix)
                                                   for pair_type, pair_suffix in item_data['pairs'].items() ) )
        else:
            # Find duplicates by stem
            dups = [group_list for key, group in itertools.groupby(item_rel_paths, key=lambda x: x.stem)
                    if len( group_list := list(group) ) > 1 ]

            errs.append(f"Duplicated items: {';'.join([ str(x) for x in sum(dups, []) ])}")

        if len(errs) != 0:
            raise Exception('Corrupted dataset:\n'+'\n'.join(errs))

        return FSIP(root=root, info=info, pair_types=pair_types, items=items)

    def __init__(self, **kwargs):
        self._root : Path = kwargs['root']
        self._info : FSIPInfo = kwargs['info']
        self._pair_types : Sequence[str] = kwargs['pair_types']
        self._items : Sequence[FSIP.Item] = kwargs['items']

    @property
    def root(self) -> Path: return self._root
    @property
    def info(self) -> FSIPInfo: return self._info
    @property
    def item_count(self) -> int: return len(self._items)
    @property
    def pair_types(self) -> Sequence[str]: return self._pair_types

    def add_pair_type(self, pair_type : str):
        r"""```
            pair_type   str     will be automatically filtered
                                allowed chars: letter(unicode),number,_
                                regex filter: re.sub('\W', '', s)

        pair dir will be created

        raise on error
        ```"""
        pair_type = re.findall(r'\w+', pair_type.lower())
        if len(pair_type) != 0:
            pair_type = pair_type[0]
            if pair_type not in self._pair_types:
                self._info.get_pair_dir_path(pair_type).mkdir(parents=True, exist_ok=True)
                self._pair_types = tuple(sorted(self._pair_types + (pair_type,)))


    def add_pair_path(self, item_id : int, pair_type : str, pair_suffix : str) -> Path:
        """
        register pair_type and pair_suffix for particular item_id.

        raise on error
        """
        pair_type = pair_type.lower()
        if pair_type in self._pair_types:
            item = self._items[item_id]
            if not item.has_pair(pair_type):
                self._items[item_id] = item.add_pair( FSIP.Pair(pair_type, pair_suffix) )
                return self.get_pair_path(item_id, pair_type)
            else:
                raise Exception('pair_path already registered')
        else:
            raise Exception(f'pair_type {pair_type} is not registered')

    def delete_pair(self, item_id : int, pair_type : str):
        """
        delete pair if file exists

        raise on error
        """
        pair_type = pair_type.lower()
        if pair_type in self._pair_types:
            item = self._items[item_id]
            if (pair := item.get_pair(pair_type)) is not None:
                pair_path = self._root / pair_type / (item.name+pair.suffix)
                pair_path.unlink(missing_ok=True)

                self._items[item_id] = item.remove_pair(pair)
        else:
            raise Exception(f'pair_type {pair_type} is not registered')


    def delete_pair_type(self, pair_type : str):
        """
        pair_dir will be removed with all content

        pair_type will be removed from all items

        raise on error
        """
        pair_type = pair_type.lower()
        if pair_type in self._pair_types:
            items = self._items
            for item_id in range(len(items)):
                item = items[item_id]
                if (pair := item.get_pair(pair_type)) is not None:
                    items[item_id] = item = item.remove_pair(pair)

            shutil.rmtree(self._info.get_pair_dir_path(pair_type))

            idx = self._pair_types.index(pair_type)
            self._pair_types = tuple(self._pair_types[:idx] + self._pair_types[idx+1:])

        else:
            raise Exception(f'pair_type {pair_type} is not registered')

    def get_item_name(self, item_id : int) -> str:
        """get unique name for item_id"""
        return self._items[item_id].name

    def get_item_path(self, item_id : int) -> Path:
        """"""
        item = self._items[item_id]
        return self._root / (item.name+item.suffix)

    def get_pair_path(self, item_id : int, pair_type : str) -> Path|None:
        """
        get pair_path for specific item_id+pair_type

        if pair_path does not exist, returns None
        """
        pair_type = pair_type.lower()
        item = self._items[item_id]
        if (pair := item.get_pair(pair_type)) is not None:
            return self._root / pair_type / (item.name+pair.suffix)
        return None

    def generate_free_names(self, amount : int) -> Sequence[str]:
        """generate amount of non exist names"""
        all_item_names = set(self.get_item_name(item_id) for item_id in range(self.item_count))
        names = []
        n = 0
        i = 0
        while i < amount:
            name = f'{(i+n):07}'
            if name not in all_item_names:
                names.append(name)
                i += 1
            else:
                n += 1
        return names

    def has_pair_type(self, pair_type : str) -> bool: return pair_type.lower() in self._pair_types
    def has_pair(self, item_id : int, pair_type : str) -> bool: return self._items[item_id].has_pair(pair_type)

    def filtered_view(self, indices : FIndices|Iterable|int) -> IFilteredFSIP_v:
        """creates view interface with filtered indices

        for example

        original indices: [0, 1, 2, 3] Total:4

        filtered_view([1,3]) -> result [0,1] Total:2
        """
        return _FSIPFilteredView(self, FIndices(indices))

    def delete_items(self, indices : FIndices|Iterable|int|None) -> IFSIP_v.DeleteResult:
        """
        Delete one or multiple items.

        returns DeleteResult

        raise NO errors
        """
        # Move items to temp dataset
        try:
            temp_fsip = FSIPInfo(self.root.parent / f'{self.root.name}_{uuid.uuid4().hex}')
            temp_fsip.root.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return FSIP.DeleteResult(errors=[e], deleted_ids=FIndices())

        result = self.move_items(indices, temp_fsip.root)

        try:
            # Delete temp dataset
            shutil.rmtree(temp_fsip.root)
        except Exception as e:
            result.errors.append(e)

        return FSIP.DeleteResult(errors=result.errors, deleted_ids=result.moved_ids, non_deleted_ids=result.non_moved_ids)

    def copy_items(self, indices : FIndices|Iterable|int|None|Mapping[int, str], fsip_root : Path) -> IFSIP_v.CopyResult:
        """
        Copy one or multiple items to other fsip root.
        Existing files will be replaced.

        ```
            indices     FIndices|Iterable|int|None
                        copy keeping the same name

                        Mapping[int, str]
                        keys are indices
                        values are new names
                        example
                        {0:'sdf'} will copy and rename
                                  source 00000.jpg to sdf.jpg

        ```
        returns CopyResult
        raise NO errors
        """
        try:
            info = FSIPInfo(fsip_root)
            info.root.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return FSIP.CopyResult(errors=[e], copied_ids=FIndices())

        if isinstance(indices, Mapping):
            item_names = indices
            indices = FIndices(indices.keys())
        else:
            item_names = {}

        items = self._items

        copied_ids = set()
        errors = []
        for item_id in indices.discard_ge(len(items)):
            item = items[item_id]
            try:
                new_name = item_names.get(item_id, item.name)

                # Collect copyables for transaction
                copyables = [ ( self._root / item.name_suffix, info.root / (new_name + item.suffix) ) ]
                for pair in item.pairs:
                    copyables.append((self._root / pair.type / (item.name + pair.suffix),
                                       info.root / pair.type / (new_name  + pair.suffix)))

                lib_path.copy_transaction(copyables)

                copied_ids.add(item_id)
            except Exception as e:
                errors.append(Exception(f'{item.name_suffix} : {e}'))

        copied_ids = FIndices(copied_ids)
        return FSIP.CopyResult(errors=errors, copied_ids=copied_ids)

    def move_items(self, indices : FIndices|Iterable|int|None|Mapping[int, str], fsip_root : Path) -> IFSIP_v.MoveResult:
        """
        Move one or multiple items to other fsip root.
        Existing files will be replaced.

        ```
            indices     FIndices|Iterable|int|None
                        move keeping the same name

                        Mapping[int, str]
                        keys are indices
                        values are new names
                        example
                        {0:'sdf'} will move and rename
                                  source 00000.jpg to sdf.jpg
        ```
        returns MoveResult
        raise NO errors
        """
        try:
            info = FSIPInfo(fsip_root)
            info.root.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return FSIP.MoveResult(errors=[e], moved_ids=FIndices())

        if isinstance(indices, Mapping):
            item_names = indices
            indices = FIndices(indices.keys())
        else:
            item_names = {}

        items = self._items

        removed_ids = set()
        errors = []
        for item_id in indices.discard_ge(len(items)):
            item = items[item_id]
            try:
                new_name = item_names.get(item_id, item.name)

                # Collect movables for transaction
                movables = [ ( self._root / item.name_suffix, info.root / (new_name + item.suffix) ) ]
                for pair in item.pairs:
                    movables.append((self._root / pair.type / (item.name + pair.suffix),
                                      info.root / pair.type / (new_name  + pair.suffix)))

                lib_path.move_transaction(movables)

                removed_ids.add(item_id)
            except Exception as e:
                errors.append(Exception(f'{item.name_suffix} : {e}'))

        removed_ids = FIndices(removed_ids)

        self._items = sequence_removed_indices(items, removed_ids)

        return FSIP.MoveResult(errors=errors, moved_ids=removed_ids)


class _FSIPFilteredView(IFilteredFSIP_v):
    def __init__(self, fsip : IFSIP_v, indices : FIndices):
        if indices.count != 0 and indices.max >= fsip.item_count:
            raise ValueError()
        self._fsip = fsip
        self._indices = indices

    @property
    def root(self) -> Path: return self._fsip.root
    @property
    def info(self) -> FSIPInfo: return self._fsip.info
    @property
    def item_count(self) -> int: return self._indices.count
    @property
    def pair_types(self) -> Sequence[str]: return self._fsip.pair_types

    def add_pair_type(self, pair_type : str): self._fsip.add_pair_type(pair_type)
    def add_pair_path(self, item_id : int, pair_type : str, pair_suffix : str) -> Path: return self._fsip.add_pair_path(self._indices.to_list()[item_id], pair_type, pair_suffix)

    def delete_pair(self, item_id : int, pair_type : str): return self._fsip.delete_pair(self._indices.to_list()[item_id], pair_type)
    def delete_pair_type(self, pair_type : str): self._fsip.delete_pair(pair_type)

    def get_item_path(self, item_id : int) -> Path:                       return self._fsip.get_item_path(self._indices.to_list()[item_id])
    def get_pair_path(self, item_id : int, pair_type : str) -> Path|None: return self._fsip.get_pair_path(self._indices.to_list()[item_id], pair_type)

    def has_pair_type(self, pair_type : str) -> bool:           return self._fsip.has_pair_type(pair_type)
    def has_pair(self, item_id : int, pair_type : str) -> bool: return self._fsip.has_pair(self._indices.to_list()[item_id], pair_type)

    def delete_items(self, indices : FIndices|Iterable|int|None) -> IFSIP_rv.DeleteResult:
        indices = FIndices(indices)

        result = self._fsip.delete_items(self.to_orig_indices(indices))

        deleted_ids = self.from_orig_indices(result.deleted_ids)

        self._indices = self._indices.discard(result.deleted_ids, shift=True)

        return self.DeleteResult(errors=result.errors, deleted_ids=deleted_ids)

    def copy_items(self, indices : FIndices|Iterable|int|Mapping[int, str], fsip_root : Path) -> IFSIP_rv.CopyResult:
        indices = FIndices(indices)

        result = self._fsip.copy_items(self.to_orig_indices(indices), fsip_root=fsip_root )
        copied_ids = self.from_orig_indices(result.deleted_ids)
        return self.CopyResult(errors=result.errors, copied_ids=copied_ids)

    def move_items(self, indices : FIndices|Iterable|int|Mapping[int, str], fsip_root : Path) -> IFSIP_rv.MoveResult:
        indices = FIndices(indices)

        result = self._fsip.move_items(self.to_orig_indices(indices), fsip_root=fsip_root )

        moved_ids = self.from_orig_indices(result.deleted_ids)
        self._indices = self._indices.discard(result.moved_ids, shift=True)

        return self.MoveResult(errors=result.errors, moved_ids=moved_ids)


    def from_orig_indices(self, indices : FIndices|Iterable|int|None) -> FIndices:
        return self._indices.indwhere(indices)

    def to_orig_indices(self, indices : FIndices|Iterable|int|None) -> FIndices:
        l = self._indices.to_list()
        return FIndices(l[indice] for indice in FIndices(indices))


