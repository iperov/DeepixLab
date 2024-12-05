from __future__ import annotations

from ._constants import Align
from .QBox import QHBox, QVBox
from .QFrame import QHFrame
from .QInfoBar import QInfoBar
from .QMenuBar import QMenuBar
from .QWindow import QWindow


class QMainWIndow(QWindow):
    def __init__(self):
        """
        provides menubar, central vbox, bottom info bar
        """
        super().__init__()

        self._q_menu_bar = QMenuBar()
        self._q_central_vbox = QVBox()
        self._q_top_bar_hbox = QHBox()
        self._q_info_bar = QInfoBar()

        (self   .add(QHBox().v_compact()
                    .add(self._q_menu_bar.h_compact(), align=Align.CenterV)
                    .add(QHFrame().add(self._q_top_bar_hbox, align=Align.LeftF)))
                .add_spacer(4)
                .add(self._q_central_vbox)
                .add(self._q_info_bar.v_compact()))

    @property
    def q_central_vbox(self) -> QVBox: return self._q_central_vbox
    @property
    def q_menu_bar(self) -> QMenuBar: return self._q_menu_bar
    @property
    def q_top_bar_hbox(self) -> QHBox: return self._q_top_bar_hbox
    @property
    def q_info_bar(self) -> QInfoBar: return self._q_info_bar
