from typing import List

from .. import mx, qt
from ..lib.collections import HFDict
from ._constants import Orientation
from ._helpers import q_init
from .QEvent import QEvent2
from .QSettings import QSettings
from .QWidget import QWidget


class QSplitter(QWidget):
    def __init__(self, **kwargs):
        super().__init__(q_widget=q_init('q_splitter', qt.QSplitter, **kwargs), **kwargs)

        self.__default_sizes = []

        q_splitter = self.q_splitter
        self.__mx_splitter_moved = QEvent2[int, int](q_splitter.splitterMoved).dispose_with(self)

        self._mx_settings.reflect(lambda settings, enter, bag=mx.Disposable().dispose_with(self): self.__ref_settings(settings, enter, bag))

    @property
    def q_splitter(self) -> qt.QSplitter: return self.q_widget

    @property
    def mx_splitter_moved(self) -> mx.IEvent2_rv[int, int]: return self.__mx_splitter_moved

    @property
    def sizes(self) -> List[int]: return self.q_splitter.sizes()

    def add(self, widget : QWidget|None):
        if widget is not None:
            widget.set_parent(self)
            self.q_splitter.addWidget(widget.q_widget)
        return self

    def set_default_sizes(self, sizes : List[int]):
        self.__default_sizes = sizes
        return self

    def set_orientation(self, orientation : Orientation):
        self.q_splitter.setOrientation(orientation)
        return self

    def set_sizes(self, sizes : List[int]):
        self.q_splitter.setSizes(sizes)

    def _show_event(self, ev: qt.QShowEvent):
        super()._show_event(ev)
        self._apply_state()

    def __ref_settings(self, settings : QSettings, enter : bool, bag : mx.Disposable):
        if enter:
            self._apply_state(settings.state)
            self.__mx_splitter_moved.listen(lambda *_: settings.state.set('sizes', self.sizes) if self.visible else ...).dispose_with(bag)
        else:
            bag.dispose_items()

    def _apply_state(self, state : HFDict = None):
        if state is not None:
            self.__state = state

        sizes = self.__state.get('sizes', self.__default_sizes)[:len(self.sizes)]
        if len(sizes) != 0:
            self.set_sizes(sizes)






