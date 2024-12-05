from typing import Tuple, overload

from .. import lx, mx, qt
from ._constants import Size, icon_Size_to_int
from ._helpers import q_init
from .QApplication import QApplication
from .QEvent import QEvent0, QEvent1
from .QWidget import QWidget
from .QIconDB import IconDB, QIconDB
from .StyleColor import StyleColor

class QAbstractButton(QWidget):
    def __init__(self, **kwargs):
        super().__init__(q_widget=q_init('q_abstract_button', _QAbstractButtonImpl, qt.QAbstractButton, **kwargs))

        q_abstract_button = self.q_abstract_button
        self._mx_clicked = QEvent0(q_abstract_button.clicked).dispose_with(self)
        self._mx_pressed = QEvent0(q_abstract_button.pressed).dispose_with(self)
        self._mx_released = QEvent0(q_abstract_button.released).dispose_with(self)
        self._mx_toggled = QEvent1[bool](q_abstract_button.toggled).dispose_with(self)

        if isinstance(q_abstract_button, _QAbstractButtonImpl):
            ...

    @property
    def q_abstract_button(self) -> qt.QPushButton: return self.q_widget

    @property
    def mx_clicked(self) -> mx.IEvent0_rv: return self._mx_clicked
    @property
    def mx_pressed(self) -> mx.IEvent0_rv: return self._mx_pressed
    @property
    def mx_released(self) -> mx.IEvent0_rv: return self._mx_released
    @property
    def mx_toggled(self) -> mx.IEvent1_rv[bool]: return self._mx_toggled
    
    @property
    def text(self) -> str: return self.q_abstract_button.text()
    def is_down(self) -> bool: return self.q_abstract_button.isDown()
    def is_checked(self) -> bool: return self.q_abstract_button.isChecked()

    def click(self):
        self.q_abstract_button.click()
        return self

    def toggle(self):
        self.q_abstract_button.toggle()
        return self

    def set_checkable(self, checkable : bool):
        self.q_abstract_button.setCheckable(checkable)
        return self

    def set_checked(self, checked : bool):
        self.q_abstract_button.setChecked(checked)
        return self

    @overload
    def set_icon(self, icon : qt.QIcon): ...
    @overload
    def set_icon(self, icon : IconDB, color : qt.QColor = StyleColor.ButtonText): ...
    def set_icon(self, *args, **kwargs):
        if len(args) == 1:
            arg0 = args[0]
            if isinstance(arg0, qt.QIcon):
                self.q_abstract_button.setIcon(args[0])
            else:
                args = (arg0, kwargs.get('color', StyleColor.ButtonText))
            
        if len(args) == 2:
            self.set_icon(QIconDB.instance().icon(args[0], args[1]))
            
        return self

    def set_icon_size(self, size : Tuple[int, int] | Size):
        if isinstance(size, Size):
            size = (icon_Size_to_int[size],)*2
        self.q_abstract_button.setIconSize(qt.QSize(*size))
        return self

    def set_text(self, text : str|None):
        if (disp := getattr(self, '_QAbstractButton_text_disp', None)) is not None:
            disp.dispose()
        self._QAbstractButton_text_disp = QApplication.instance().mx_language.reflect(lambda lang: self.q_abstract_button.setText(lx.L(text, lang))).dispose_with(self)
        return self


class _QAbstractButtonImpl(qt.QAbstractButton): ...








