from __future__ import annotations

from typing import Any

from .. import qt
from ._helpers import q_init
from .QEvent import QEvent0, QEvent2
from .QFuncWrap import QFuncWrap
from .QObject import QObject


class QAbstractItemModel(QObject):
    def __init__(self, **kwargs):
        super().__init__(q_object=q_init('q_abstract_item_model', _QAbstractItemModelImpl, qt.QAbstractItemModel, **kwargs), **kwargs)

        q_abstract_item_model = self.q_abstract_item_model

        self.__ev_layout_changed = QEvent0(q_abstract_item_model.layoutChanged).dispose_with(self)
        self.__ev_data_changed = QEvent2[qt.QModelIndex, qt.QModelIndex](q_abstract_item_model.dataChanged).dispose_with(self)

        if isinstance(q_abstract_item_model, _QAbstractItemModelImpl):
            self._flags_wrap = QFuncWrap(q_abstract_item_model, 'flags', lambda *args, **kwargs: self.flags(*args, **kwargs)).dispose_with(self)

            self._rowCount_wrap = QFuncWrap(q_abstract_item_model, 'rowCount', lambda *args, **kwargs: self.row_count(*args, **kwargs)).dispose_with(self)
            self._columnCount_wrap = QFuncWrap(q_abstract_item_model, 'columnCount', lambda *args, **kwargs: self.column_count(*args, **kwargs)).dispose_with(self)
            self._index_wrap = QFuncWrap(q_abstract_item_model, 'index', lambda *args, **kwargs: self.index(*args, **kwargs)).dispose_with(self)
            self._header_data_wrap = QFuncWrap(q_abstract_item_model, 'headerData', lambda *args, **kwargs: self.header_data(*args, **kwargs)).dispose_with(self)

            self._data_wrap = QFuncWrap(q_abstract_item_model, 'data', lambda *args, **kwargs: self.data(*args, **kwargs)).dispose_with(self)
            self._parent_wrap = QFuncWrap(q_abstract_item_model, 'parent', lambda *args, **kwargs: self._on_parent_wrap(*args, **kwargs)).dispose_with(self)

    @property
    def q_abstract_item_model(self) -> qt.QAbstractItemModel: return self.q_object

    @property
    def _ev_layout_changed(self) -> QEvent0:
        return self.__ev_layout_changed

    @property
    def _ev_data_changed(self) -> QEvent2[qt.QModelIndex, qt.QModelIndex]:
        """top_left, bottom_right"""
        return self.__ev_data_changed

    def create_index(self, row : int, col : int, obj) -> qt.QModelIndex:
        return self.q_abstract_item_model.createIndex(row, col, obj)

    def flags(self, index : qt.QModelIndex) -> qt.Qt.ItemFlag:
        raise NotImplementedError()

    def row_count(self, index : qt.QModelIndex) -> int:
        raise NotImplementedError()

    def column_count(self, index : qt.QModelIndex) -> int:
        raise NotImplementedError()

    def index(self, row : int, column : int, parent : qt.QModelIndex) -> qt.QModelIndex:
        raise NotImplementedError()

    def header_data(self, section: int, orientation: qt.Qt.Orientation, role: int = ...) -> Any:
        raise NotImplementedError()

    def data(self, index : qt.QModelIndex, role: int = ...) -> Any:
        raise NotImplementedError()

    def parent(self, child : qt.QModelIndex) -> qt.QModelIndex:
        raise NotImplementedError()

    def _on_parent_wrap(self, child : qt.QModelIndex = ...):
        if child is Ellipsis:
            return self._parent_wrap.get_super()()

        return self.parent(child)

class _QAbstractItemModelImpl(qt.QAbstractItemModel): ...
