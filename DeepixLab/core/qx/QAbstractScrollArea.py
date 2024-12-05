from .. import qt
from ._helpers import q_init
from .QFrame import QFrame


class QAbstractScrollArea(QFrame):
    def __init__(self, **kwargs):
        super().__init__(q_frame=q_init('q_abstract_scroll_area', _QAbstractScrollAreaImpl, qt.QAbstractScrollArea, **kwargs), **kwargs)

        q_abstract_scroll_area = self.q_abstract_scroll_area

        if isinstance(q_abstract_scroll_area, _QAbstractScrollAreaImpl):
            ...

    @property
    def q_abstract_scroll_area(self) -> qt.QAbstractScrollArea: return self.get_q_frame()


class _QAbstractScrollAreaImpl(qt.QAbstractScrollArea): ...