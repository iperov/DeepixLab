from .. import lx, mx, qt
from ._constants import Align, Align_to_AlignmentFlag, Orientation
from ._helpers import q_init
from .QApplication import QApplication
from .QEvent import QEvent1
from .QWidget import QWidget


class QProgressBar(QWidget):

    def __init__(self, **kwargs):
        super().__init__(q_widget=q_init('q_progress_bar', qt.QProgressBar, **kwargs), **kwargs)

        q_progress_bar = self.q_progress_bar

        self._mx_value = mx.GetSetProperty[int](self.get_value, self.set_value, QEvent1[int](q_progress_bar.valueChanged).dispose_with(self) ).dispose_with(self)

        self.set_orientation(Orientation.Horizontal)

    @property
    def q_progress_bar(self) -> qt.QProgressBar: return self.q_widget

    @property
    def mx_value(self) -> mx.IProperty_v[int]: return self._mx_value


    def get_value(self) -> int: return self.q_progress_bar.value()

    def set_value(self, value : int):
        self.q_progress_bar.setValue(value)
        return self

    def set_alignment(self, align : Align):
        self.q_progress_bar.setAlignment(Align_to_AlignmentFlag[align])
        return self

    def set_orientation(self, orientation : Orientation):
        self.q_progress_bar.setOrientation(orientation)
        if orientation == Orientation.Horizontal:
            self.v_compact()
            self.h_normal()
        else:
            self.h_compact()
            self.v_normal()

        return self

    def set_minimum(self, min : int):
        self.q_progress_bar.setMinimum(min)
        return self

    def set_maximum(self, max : int):
        self.q_progress_bar.setMaximum(max)
        return self

    def set_format(self, format : str):
        """
        string used to generate the current text

        %p - is replaced by the percentage completed. %v - is replaced by the current value. %m - is replaced by the total number of steps.

        The default value is "%p%".
        """
        self.q_progress_bar.setFormat(format)

        if (disp := getattr(self, '_QProgressBar_text_disp', None)) is not None:
            disp.dispose()
        self._QProgressBar_text_disp = QApplication.instance().mx_language.reflect(lambda lang: self.q_progress_bar.setFormat(lx.L(format, lang))).dispose_with(self)


        return self

    def set_text_visible(self, visible : bool):
        self.q_progress_bar.setTextVisible(visible)
        return self



