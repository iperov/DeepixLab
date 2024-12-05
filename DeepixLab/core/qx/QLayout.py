from .. import qt
from ._helpers import q_init
from .QWidget import QWidget


class QLayout(QWidget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__q_layout = q_init('q_layout', None, qt.QLayout, **kwargs)
        self.q_widget.setLayout(self.__q_layout)
    
    @property
    def q_layout(self) -> qt.QLayout: return self.__q_layout
    
    def get_count(self) -> int: return self.q_layout.count()

    def widget_at(self, idx : int) -> QWidget|None:
        item = self.q_layout.itemAt(idx)
        return getattr(item.widget(), '__owner', None)

    def index_of(self, widget : QWidget) -> int:
        return self.q_layout.indexOf(widget.q_widget)

    def remove_widget(self, widget : QWidget):
        if not (widget.parent is self):
            raise Exception(f'Widget {widget} was not added to {self}.')
        self.q_layout.removeWidget(widget.q_widget)
        widget.set_parent(None)
        return self

    def set_spacing(self, spacing : int):
        self.q_layout.setSpacing(spacing)
        return self