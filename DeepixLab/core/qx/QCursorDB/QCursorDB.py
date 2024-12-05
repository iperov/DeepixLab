from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from ... import mx, qt
from .._constants import Size
from .CursorDB import CursorDB

T = TypeVar('T')

Size_to_icon_size = {   Size.XXL : 64,
                        Size.XL : 48,
                        Size.L : 32,
                        Size.M : 24,
                        Size.S : 16}


class QCursorDB(mx.Disposable):

    @staticmethod
    def instance() -> QCursorDB:
        if QCursorDB._instance is None:
            raise Exception('No QCursorDB instance.')
        return QCursorDB._instance

    def __init__(self):
        super().__init__()
        if QCursorDB._instance is not None:
            raise Exception('QCursorDB instance already exists.')
        QCursorDB._instance = self
        self._cached = {}

    def __dispose__(self):
        QCursorDB._instance = None
        self._cached = None
        super().__dispose__()

    def pixmap(self, cursor : CursorDB, size : Size = Size.M) -> qt.QPixmap:
        return self._get(cursor, size, qt.QPixmap)

    def cursor(self, cursor : CursorDB, size : Size = Size.M) -> qt.QImage:
        return self._get(cursor, size, qt.QCursor)

    def _get(self, cursor : CursorDB, size : Size, out_cls):
        key = (cursor, size, out_cls)#color.getRgb(), 

        result = self._cached.get(key, None)

        if result is None:
            if issubclass(out_cls, qt.QPixmap):
                icon_size = Size_to_icon_size[size]
                pixmap = qt.QPixmap(str(Path(__file__).parent / 'assets' / (cursor.name+'.png')))
                pixmap = pixmap.scaled(icon_size, icon_size, mode=qt.Qt.TransformationMode.SmoothTransformation)
                result = self._cached[key] = pixmap

            elif issubclass(out_cls, qt.QCursor):
                result = self._cached[key] = qt.QCursor(self._get(cursor, size, qt.QPixmap))
            else:
                raise ValueError('Unknown type out_cls')

        return result

    _instance : QCursorDB = None

