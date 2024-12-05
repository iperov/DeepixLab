from .. import qt
from ._helpers import q_init
from .QAbstractItemModel import QAbstractItemModel
from .QAbstractItemView import QAbstractItemView


class QTreeView(QAbstractItemView):
    def __init__(self, **kwargs):
        super().__init__(q_abstract_item_view=q_init('q_tree_view', qt.QTreeView, **kwargs), **kwargs)

    @property
    def q_tree_view(self) -> qt.QTreeView: return self.q_abstract_item_view

    def set_model(self, model : QAbstractItemModel):
        self.q_tree_view.setModel(model.q_abstract_item_model)
        return self