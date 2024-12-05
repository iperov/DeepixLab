from __future__ import annotations
from typing import Callable

from .. import qt
from ._helpers import q_init
from .QEvent import QEvent0
from .QObject import QObject


class QTimer(QObject):
    def __init__(self, on_timeout : Callable[ [QTimer], None ], **kwargs):
        super().__init__(q_object=q_init('q_timer', qt.QTimer, **kwargs), **kwargs)

        q_timer = self.q_timer
        QEvent0(q_timer.timeout).dispose_with(self).listen(lambda: on_timeout(self))

    def __dispose__(self):
        self.q_timer.stop()
        super().__dispose__()

    @property
    def q_timer(self) -> qt.QTimer: return self.q_object

    def set_interval(self, msec : int):
        self.q_timer.setInterval(msec)
        return self

    def start(self):
        self.q_timer.start()
        return self
    
    def stop(self):
        self.q_timer.stop()
        return self