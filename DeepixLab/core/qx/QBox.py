from .. import qt
from ._constants import Align, Align_to_AlignmentFlag
from ._helpers import q_init
from .QLayout import QLayout
from .QWidget import QWidget


class QBox(QLayout):
    def __init__(self, **kwargs):
        super().__init__(q_layout=q_init('q_box_layout', None, qt.QBoxLayout, **kwargs), **kwargs)

    @property
    def q_box_layout(self) -> qt.QBoxLayout: return self.q_layout

    def add(self, widget : QWidget|None, stretch : int = 0, align : Align = Align.CenterE):
        if widget is not None:
            widget.set_parent(self)
            self.q_box_layout.addWidget(widget.q_widget, stretch=stretch, alignment=Align_to_AlignmentFlag[align])
        return self

    def insert(self, idx : int, widget : QWidget):
        if widget.parent is not None:
            raise Exception(f'Widget {widget} already has parent. Unable to add.')
        widget.set_parent(self)
        self.q_box_layout.insertWidget(idx, widget.q_widget)
        return self


class QHBox(QBox):
    def __init__(self, **kwargs):
        super().__init__(q_box_layout=q_init('q_hbox_layout', qt.QHBoxLayout, **kwargs), **kwargs)

    @property
    def q_hbox_layout(self) -> qt.QHBoxLayout: return self.q_layout

    def add_spacer(self, size : int):
        self.add(QWidget().h_compact(size).v_normal())
        return self

class QVBox(QBox):
    def __init__(self, **kwargs):
        super().__init__(q_box_layout=q_init('q_vbox_layout', qt.QVBoxLayout, **kwargs), **kwargs)
    
    @property
    def q_vbox_layout(self) -> qt.QVBoxLayout: return self.q_layout

    def add_spacer(self, size : int):
        self.add(QWidget().h_normal().v_compact(size))
        return self