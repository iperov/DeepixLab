from .. import qt
from ._helpers import q_init

from .QAbstractScrollArea import QAbstractScrollArea

class QAbstractItemView(QAbstractScrollArea):
    def __init__(self, **kwargs):
        super().__init__(q_abstract_scroll_area=q_init('q_abstract_item_view', _QAbstractItemViewImpl, qt.QAbstractItemView, **kwargs), **kwargs)

        q_abstract_item_view = self.q_abstract_item_view

        if isinstance(q_abstract_item_view, _QAbstractItemViewImpl):
            ...

    @property
    def q_abstract_item_view(self) -> qt.QAbstractItemView: return self.q_abstract_scroll_area


class _QAbstractItemViewImpl(qt.QAbstractItemView): ...