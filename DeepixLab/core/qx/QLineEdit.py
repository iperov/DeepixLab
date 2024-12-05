from typing import Callable

from .. import lx, mx, qt
from ._helpers import q_init
from .QApplication import QApplication
from .QEvent import QEvent0, QEvent1
from .QWidget import QWidget


class QLineEdit(QWidget):
    def __init__(self, **kwargs):
        super().__init__(q_widget=q_init('q_line_edit', qt.QLineEdit, **kwargs), **kwargs)

        q_line_edit = self.q_line_edit

        mx_editing_finished = self._mx_editing_finished = QEvent0(q_line_edit.editingFinished).dispose_with(self)
        mx_editing_finished.listen(lambda: self.clear_focus())

        self._mx_text = mx.GetSetProperty[str](self.get_text, self.set_text, QEvent1[str](q_line_edit.textChanged).dispose_with(self) ).dispose_with(self)

        self.set_placeholder_text(None)

    @property
    def mx_text(self) -> mx.IProperty_v[str]: return self._mx_text
    @property
    def mx_editing_finished(self) -> mx.IEvent0_rv: return self._mx_editing_finished

    @property
    def q_line_edit(self) -> qt.QLineEdit: return self.q_widget

    def get_text(self) -> str:
        return self.q_line_edit.text()

    def set_text(self, text : str):
        self.q_line_edit.setText(text)
        return self

    def set_placeholder_text(self, text : str|None):
        if (disp := getattr(self, '_QLineEdit_text_disp', None)) is not None:
            disp.dispose()
        self._QLineEdit_text_disp = QApplication.instance().mx_language.reflect(lambda lang: self.q_line_edit.setPlaceholderText(
                                                lx.L(text, lang) if text is not None else '...'
                                            )).dispose_with(self)
        return self

    def set_read_only(self, read_only : bool):
        self.q_line_edit.setReadOnly(read_only)
        return self

    def set_filter(self, func : Callable[ [str], str ] ):
        """Filter string using callable func"""
        self.q_line_edit.setValidator(QFuncValidator(func))
        return self

class QFuncValidator(qt.QValidator):
    def __init__(self, func : Callable[ [str], str ]):
        super().__init__()
        self._func = func

    def fixup(self, s: str) -> str:
        return self._func(s)

    def validate(self, s: str, pos: int) -> object:
        if self._func(s) == s:
            return qt.QValidator.State.Acceptable
        return qt.QValidator.State.Invalid

