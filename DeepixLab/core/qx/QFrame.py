from .. import qt
from ._helpers import q_init
from .QBox import QHBox, QVBox
from .QWidget import QWidget


class QFrame(QWidget):
    def __init__(self, **kwargs):
        super().__init__(q_widget=q_init('q_frame', qt.QFrame, **kwargs), **kwargs)
    
    @property
    def q_frame(self) -> qt.QFrame: return self.q_widget
    

class QHFrame(QHBox):
    def __init__(self):
        q_frame = self.__q_frame = qt.QFrame()
        q_frame.setFrameShape(qt.QFrame.Shape.NoFrame)
        q_frame.setAutoFillBackground(True)
        q_frame.setBackgroundRole(qt.QPalette.ColorRole.Mid)
        super().__init__(q_widget=q_frame)

    @property
    def q_frame(self) -> qt.QFrame: return self.q_widget


class QVFrame(QVBox):
    def __init__(self):
        q_frame = self.__q_frame = qt.QFrame()
        q_frame.setFrameShape(qt.QFrame.Shape.NoFrame)
        q_frame.setAutoFillBackground(True)
        q_frame.setBackgroundRole(qt.QPalette.ColorRole.Mid)
        super().__init__(q_widget=q_frame)

    @property
    def q_frame(self) -> qt.QFrame: return self.q_widget
