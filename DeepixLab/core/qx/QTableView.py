from .. import qt
from ._helpers import q_init
from .QAbstractItemModel import QAbstractItemModel
from .QHeaderView import QHeaderView
from .QWidget import QWidget


class QTableView(QWidget):
    def __init__(self, **kwargs):
        super().__init__(q_widget=q_init('q_table_view', qt.QTableView, **kwargs), **kwargs)

        self._horizontal_header = QHeaderView(q_header_view=self.q_table_view.horizontalHeader(), wrap_mode=True).dispose_with(self)
    
    @property
    def q_table_view(self) -> qt.QTableView: return self.q_widget

    def get_horizontal_header(self) -> QHeaderView:
        return self._horizontal_header

    def set_model(self, model : QAbstractItemModel):
        self.q_table_view.setModel(model.q_abstract_item_model)
        return self

    def set_column_width(self, column : int, width : int):
        self.q_table_view.setColumnWidth(column, width)
        return self