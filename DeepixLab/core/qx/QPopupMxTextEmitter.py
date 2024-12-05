from typing import Iterable, Sequence

from .. import lx, mx, qt
from ._constants import WindowType
from .QApplication import QApplication
from .QFontDB import FontDB
from .QIconDB import IconDB
from .QObject import QObject
from .QPushButton import QPushButton
from .QTextEdit import QTextEdit
from .QWindow import QWindow


class QPopupMxTextEmitter(QObject):
    def __init__(self, text_emitter : mx.ITextEmitter_v|Sequence[mx.ITextEmitter_v], title : str = None):
        """
        """
        super().__init__()
        self._title = title

        self._wnd = None
        self._text_edit : QTextEdit = None

        if not isinstance(text_emitter, Iterable):
            text_emitter = (text_emitter,)

        for x in text_emitter:
            x.listen(self._on_text).dispose_with(self)

    def show_popup(self):
        if self._wnd is None:
            self._text_edit = QTextEdit().set_font(FontDB.FixedWidth).set_read_only(True)
            self._wnd = (QWindow().set_parent(self)
                            .set_window_size(300, 300)
                            .set_window_flags(WindowType.Window | WindowType.WindowTitleHint | WindowType.CustomizeWindowHint |  WindowType.WindowStaysOnTopHint)
                            .set_window_icon(IconDB.alert_circle_outline, qt.QColor(255,0,0))
                            .set_title(self._title)

                            .add(self._text_edit)
                            .add(QPushButton().v_compact().set_icon(IconDB.checkmark_done, qt.QColor(100,200,0))
                                    .inline(lambda btn: btn.mx_clicked.listen(lambda: self._wnd.dispose())))

                            .show())

            mx.CallOnDispose(lambda: (setattr(self, '_wnd', None), setattr(self, '_text_edit', None)) ).dispose_with(self._wnd)

    def _on_text(self, text : str):
        self.show_popup()

        text = lx.L(text, QApplication.instance().mx_language.get())

        self._text_edit.set_plain_text( self._text_edit.get_plain_text() + text + '\r\n')
