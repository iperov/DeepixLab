from .. import lx, mx, qt
from ._helpers import q_init
from .QApplication import QApplication
from .QEvent import QEvent0
from .QObject import QObject


class QAction(QObject):
    def __init__(self, **kwargs):
        super().__init__(q_object=q_init('q_action', qt.QAction, **kwargs), **kwargs)

        q_action = self.q_action
        self._mx_triggered = QEvent0(q_action.triggered).dispose_with(self)

    @property
    def mx_triggered(self) -> mx.IEvent0_rv: return self._mx_triggered

    @property
    def q_action(self) -> qt.QAction: return self.q_object

    def set_checkable(self, checkable : bool):
        self.q_action.setCheckable(checkable)
        return self

    def set_text(self, text : str|None):
        if (disp := getattr(self, '_QAction_text_disp', None)) is not None:
            disp.dispose()
        self._QAction_text_disp = QApplication.instance().mx_language.reflect(lambda lang: self.q_action.setText(lx.L(text, lang))).dispose_with(self)
        return self

    def enable(self):
        self.set_enabled(True)
        return self

    def disable(self):
        self.set_enabled(False)
        return self

    def set_enabled(self, enabled : bool):
        self.q_action.setEnabled(enabled)
        return self

