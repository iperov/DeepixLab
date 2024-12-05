from .. import lx, mx, qt
from ._helpers import q_init
from .QApplication import QApplication
from .QEvent import QEvent0
from .QWidget import QWidget


class QAbstractSpinBox(QWidget):
    def __init__(self, **kwargs):
        super().__init__(q_widget=q_init('q_abstract_spin_box', _QAbstractSpinBoxImpl, qt.QAbstractSpinBox, **kwargs), **kwargs)

        q_abstract_spin_box = self.q_abstract_spin_box
        self._mx_editing_finished = QEvent0(q_abstract_spin_box.editingFinished).dispose_with(self)

        if isinstance(q_abstract_spin_box, _QAbstractSpinBoxImpl):
            ...

    @property
    def q_abstract_spin_box(self) -> qt.QAbstractSpinBox: return self.q_widget
    
    @property
    def mx_editing_finished(self) -> mx.IEvent0_rv: return self._mx_editing_finished

    def set_read_only(self, r : bool):
        self.q_abstract_spin_box.setReadOnly(r)
        return self

    def set_special_value_text(self, text : str|None):
        if (disp := getattr(self, '_QAbstractSpinBox_svt_disp', None)) is not None:
            disp.dispose()
        self._QAbstractSpinBox_svt_disp = \
            QApplication.instance().mx_language.reflect(lambda lang: self.q_abstract_spin_box.setSpecialValueText(lx.L(text, lang))).dispose_with(self)



class _QAbstractSpinBoxImpl(qt.QAbstractSpinBox): ...