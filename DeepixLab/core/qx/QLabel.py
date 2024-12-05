from .. import lx, qt
from ._constants import Align, Align_to_AlignmentFlag, TextInteractionFlag
from ._helpers import q_init
from .QApplication import QApplication
from .QWidget import QWidget


class QLabel(QWidget):
    def __init__(self, **kwargs):
        super().__init__(q_widget=q_init('q_label', qt.QLabel, **kwargs), **kwargs)

    @property
    def q_label(self) -> qt.QLabel: return self.q_widget

    def set_text_interaction_flags(self, flags : TextInteractionFlag):
        self.q_label.setTextInteractionFlags(flags)
        return self

    def set_text(self, text : str|None):
        if (disp := getattr(self, '_QLabel_text_disp', None)) is not None:
            disp.dispose()
        self._QLabel_text_disp = QApplication.instance().mx_language.reflect(lambda lang: self.q_label.setText(lx.L(text, lang))).dispose_with(self)
        return self

    def set_align(self, align : Align):
        self.q_label.setAlignment(Align_to_AlignmentFlag[align])
        return self

    def set_pixmap(self, pixmap : qt.QPixmap|None):
        self.q_label.setPixmap(pixmap if pixmap is not None else qt.QPixmap())
        return self

    def set_word_wrap(self, word_wrap : bool):
        self.q_label.setWordWrap(word_wrap)
        return self

    def set_scaled_contents(self, scaled_contents : bool):
        self.q_label.setScaledContents(scaled_contents)
        return self






