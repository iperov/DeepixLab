from typing import TypeVar

from .. import mx

A0 = TypeVar('A0')
A1 = TypeVar('A1')

class QEvent0(mx.Event0):
    """wraps qt signal as mx.Event0"""
    def __init__(self, qt_signal):
        super().__init__()
        self._qt_signal = qt_signal
        self._qt_conn = qt_signal.connect( lambda *args, super_emit=super().emit: super_emit(*args) )

    def emit(self):
        self._qt_signal.emit()
        return self

    def __dispose__(self):
        self._qt_signal.disconnect(self._qt_conn)
        super().__dispose__()

class QEvent1(mx.Event1[A0]):
    """wraps qt signal as mx.Event1"""
    def __init__(self, qt_signal):
        super().__init__()
        self._qt_signal = qt_signal
        self._qt_conn = qt_signal.connect( lambda *args, super_emit=super().emit: super_emit(*args) )

    def emit(self, a0 : A0):
        self._qt_signal.emit(a0)
        return self

    def __dispose__(self):
        self._qt_signal.disconnect(self._qt_conn)
        super().__dispose__()

class QEvent2(mx.Event2[A0, A1]):
    """wraps qt signal as mx.Event2"""
    def __init__(self, qt_signal):
        super().__init__()
        self._qt_signal = qt_signal
        self._qt_conn = qt_signal.connect( lambda *args, super_emit=super().emit: super_emit(*args) )

    def emit(self, a0 : A0, a1 : A1):
        self._qt_signal.emit(a0, a1)
        return self

    def __dispose__(self):
        self._qt_signal.disconnect(self._qt_conn)
        super().__dispose__()
