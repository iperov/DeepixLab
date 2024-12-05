from __future__ import annotations

from typing import Callable

from .. import lx, qt, mx
from ._helpers import q_init
from .QApplication import QApplication
from .QBox import QVBox
from .QEvent import QEvent1
from .QSettings import QSettings
from .QWidget import QWidget


class QTab(QVBox):
    def __init__(self, owner : QTabWidget, **kwargs):
        super().__init__(**kwargs)
        self._owner = owner

    def set_title(self, title : str):
        if (disp := getattr(self, '_QTab_title_disp', None)) is not None:
            disp.dispose()
        self._QTab_title_disp = QApplication.instance().mx_language.reflect(lambda lang:
                (q_tab_widget := self._owner.q_tab_widget).setTabText(q_tab_widget.indexOf(self.q_widget), lx.L(title, lang))
            ).dispose_with(self)

        return self

class QTabWidget(QWidget):
    TabPosition = qt.QTabWidget.TabPosition

    def __init__(self, **kwargs):
        super().__init__(q_widget=q_init('q_tab_widget', qt.QTabWidget, **kwargs), **kwargs)
        
        q_tab_widget = self.q_tab_widget
        
        self.__mx_current_changed = QEvent1[int](q_tab_widget.currentChanged).dispose_with(self)
        
        self._settings_bag = mx.Disposable()
        self._mx_settings.reflect(lambda settings, enter, bag=self._settings_bag: self.__ref_settings(settings, enter, bag))
    
    def __dispose__(self):
        self._settings_bag.dispose()
        super().__dispose__()
    
    @property
    def q_tab_widget(self) -> qt.QTabWidget: return self.q_widget
    
    @property
    def mx_current_changed(self) -> QEvent1[int]: return self.__mx_current_changed
    
    @property
    def current_index(self) -> int: return self.q_tab_widget.currentIndex()

    def add_tab(self, inline : Callable[ [QTab], None]):
        tab = QTab(self).set_parent(self)
        self.q_tab_widget.addTab(tab.q_widget, '')
        inline(tab)
        return self

    def set_tab_position(self, position : TabPosition):
        self.q_tab_widget.setTabPosition(position)
        return self
        
    def __ref_settings(self, settings : QSettings, enter : bool, bag : mx.Disposable ):
        if enter:
            if (current_index := settings.state.get('current_index', None)) is not None:
                self.q_tab_widget.setCurrentIndex(current_index)
            self.__mx_current_changed.listen(lambda idx: settings.state.set('current_index', idx)).dispose_with(bag)
        else:
            bag.dispose_items()
            
