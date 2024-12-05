from __future__ import annotations

from .. import qt
from ._helpers import q_init
from .QMenu import QMenu
from .QWidget import QWidget


class QMenuBar(QWidget):
    def __init__(self, **kwargs):
        super().__init__(q_widget=q_init('q_menu_bar', qt.QMenuBar, **kwargs), **kwargs)
    
    @property
    def q_menu_bar(self) -> qt.QMenuBar: return self.q_widget

    def add(self, menu : QMenu|None):
        if menu is not None:
            menu.set_parent(self)
            self.q_menu_bar.addMenu(menu.q_menu)
        return self
