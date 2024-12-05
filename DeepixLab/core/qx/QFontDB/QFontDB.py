from __future__ import annotations

from os import scandir
from pathlib import Path

from ... import mx, qt
from .._constants import Size
from .FontDB import FontDB


class QFontDB(mx.Disposable):


    @staticmethod
    def instance() -> QFontDB:
        if QFontDB._instance is None:
            raise Exception('No QFontDB instance.')
        return QFontDB._instance

    def __init__(self):
        super().__init__()
        if QFontDB._instance is not None:
            raise Exception('QFontDB instance already exists.')
        QFontDB._instance = self

        self._cached = {}
        self._cached_appfont = set()

    def __dispose__(self):
        QFontDB._instance = None
        self._cached = None
        self._cached_appfont = None
        super().__dispose__()

    def default(self, size : int|Size = Size.Default) -> qt.QFont: 
        return self.get(FontDB.Default, size)

    def fixed_width(self, size : int|Size = Size.Default) -> qt.QFont:
        return self.get(FontDB.FixedWidth, size)

    def digital(self, size : int|Size = Size.Default) -> qt.QFont:
        return self.get(FontDB.Digital, size)

    def get(self, font : FontDB, size : Size|int = Size.Default, italic=False, bold=False) -> qt.QFont:
        if isinstance(size, Size):
            size = Font_ESize_to_font_size[font][size]

        key = (font, size, italic, bold)
        q_font = self._cached.get(key, None)
        if q_font is None:
            name = Font_to_name[font]

            if name not in self._cached_appfont:
                dir = Path(__file__).parent / 'assets' / name
                if dir.exists() and dir.is_dir():
                    for filepath in sorted( Path(x.path) for x in scandir(str(dir)) ):
                        qt.QFontDatabase.addApplicationFont(str(filepath))
                self._cached_appfont.add(name)

            q_font = self._cached[key] = qt.QFont(name, size)
            q_font.setItalic(italic)
            q_font.setBold(bold)

        return q_font

    _instance : QFontDB = None


Font_to_name = {
    FontDB.Default       : 'Noto Sans',
    FontDB.FixedWidth    : 'Noto Mono',
    FontDB.Digital       : 'Digital-7 Mono' }


Font_ESize_to_font_size = {
    FontDB.Default: {  Size.XXL : 32,
                        Size.XL : 22,
                        Size.L : 14,
                        Size.M : 8,
                        Size.S : 6},

    FontDB.FixedWidth: {   Size.XXL : 32,
                            Size.XL : 22,
                            Size.L : 14,
                            Size.M : 8,
                            Size.S : 6},

    FontDB.Digital: {  Size.XXL : 32,
                        Size.XL : 22,
                        Size.L : 14,
                        Size.M : 11,
                        Size.S : 6},
}
