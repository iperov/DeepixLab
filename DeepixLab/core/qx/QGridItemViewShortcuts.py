from __future__ import annotations

import itertools

from .. import ax, qt
from .QGridItemView import QGridItemView
from .QObject import QObject
from .QShortcut import QShortcut


class QGridItemViewShortcuts(QObject):
    """
    generic set of shortcuts for controlling QGridItemView
    """
    def __init__(self, item_view : QGridItemView):
        super().__init__()
        self._item_view = item_view

        self._nav_fg = ax.FutureGroup().dispose_with(self)
        
        self._shortcut_select_unselect_all = \
            QShortcut(qt.QKeyCombination(qt.Qt.KeyboardModifier.ControlModifier, qt.Qt.Key.Key_A)).set_parent(self).inline(lambda shortcut:
                                    shortcut.mx_press.listen(lambda: item_view.apply_model(item_view.model.unselect_all() if item_view.model.is_selected_all() else item_view.model.select_all())),)


        QShortcut(qt.QKeyCombination(qt.Qt.Key.Key_Home),     ).set_parent(self).inline(lambda shortcut: shortcut.mx_press.listen(lambda: item_view.apply_model(item_view.model.select_prev(first=True))))
        QShortcut(qt.QKeyCombination(qt.Qt.Key.Key_End),      ).set_parent(self).inline(lambda shortcut: shortcut.mx_press.listen(lambda: item_view.apply_model(item_view.model.select_next(last=True))))

        QShortcut(qt.QKeyCombination(qt.Qt.Key.Key_Left),     ).set_parent(self).inline(lambda shortcut: (shortcut.mx_press.listen(self._shortcut_left), shortcut.mx_release.listen(lambda: self._nav_fg.cancel_all()))  )
        QShortcut(qt.QKeyCombination(qt.Qt.Key.Key_Up),       ).set_parent(self).inline(lambda shortcut: (shortcut.mx_press.listen(self._shortcut_up),   shortcut.mx_release.listen(lambda: self._nav_fg.cancel_all()))  )
        QShortcut(qt.QKeyCombination(qt.Qt.Key.Key_PageUp),   ).set_parent(self).inline(lambda shortcut: (shortcut.mx_press.listen(self._shortcut_pgup), shortcut.mx_release.listen(lambda: self._nav_fg.cancel_all()))  )

        QShortcut(qt.QKeyCombination(qt.Qt.Key.Key_PageDown), ).set_parent(self).inline(lambda shortcut: (shortcut.mx_press.listen(self._shortcut_pgdown), shortcut.mx_release.listen(lambda: self._nav_fg.cancel_all()))  )
        QShortcut(qt.QKeyCombination(qt.Qt.Key.Key_Down),     ).set_parent(self).inline(lambda shortcut: (shortcut.mx_press.listen(self._shortcut_down), shortcut.mx_release.listen(lambda: self._nav_fg.cancel_all()))  )
        QShortcut(qt.QKeyCombination(qt.Qt.Key.Key_Right),    ).set_parent(self).inline(lambda shortcut: (shortcut.mx_press.listen(self._shortcut_right), shortcut.mx_release.listen(lambda: self._nav_fg.cancel_all()))  )

        self._shortcut_mark_unmark_selected = \
            QShortcut(qt.QKeyCombination(qt.Qt.Key.Key_Space), ).set_parent(self).inline(lambda shortcut:
                                    shortcut.mx_press.listen(lambda: item_view.apply_model(item_view.model.mark( item_view.model.marked_items ^ item_view.model.selected_items) )))
        
        self._shortcut_select_marked = \
            QShortcut(qt.QKeyCombination(qt.Qt.KeyboardModifier.ControlModifier, qt.Qt.Key.Key_Space), ).set_parent(self).inline(lambda shortcut:
                                    shortcut.mx_press.listen(lambda: item_view.apply_model(item_view.model.select(item_view.model.marked_items))))
        
        self._shortcut_invert_selection = \
            QShortcut(qt.QKeyCombination(qt.Qt.KeyboardModifier.ControlModifier, qt.Qt.Key.Key_I), ).set_parent(self).inline(lambda shortcut:
                                    shortcut.mx_press.listen(lambda: item_view.apply_model(item_view.model.select(item_view.model.selected_items.invert(max=item_view.model.item_count) ))))
       
    @property
    def shortcut_select_unselect_all(self) -> QShortcut: return self._shortcut_select_unselect_all
    @property
    def shortcut_mark_unmark_selected(self) -> QShortcut: return self._shortcut_mark_unmark_selected
    @property
    def shortcut_select_marked(self) -> QShortcut: return self._shortcut_select_marked
    @property
    def shortcut_invert_selection(self) -> QShortcut: return self._shortcut_invert_selection
    
     
    @ax.task
    def _shortcut_left(self):
        yield ax.attach_to(self._nav_fg, cancel_all=True)
        for i in itertools.count():
            self._item_view.apply_model(self._item_view.model.select_prev())
            yield ax.sleep(0.5 if i == 0 else 0.05)

    @ax.task
    def _shortcut_up(self):
        yield ax.attach_to(self._nav_fg, cancel_all=True)
        for i in itertools.count():
            self._item_view.apply_model(self._item_view.model.select_prev(row_up=True))
            yield ax.sleep(0.5 if i == 0 else 0.05)

    @ax.task
    def _shortcut_pgup(self):
        yield ax.attach_to(self._nav_fg, cancel_all=True)
        for i in itertools.count():
            self._item_view.apply_model(self._item_view.model.select_prev(page_up=True))
            yield ax.sleep(0.5 if i == 0 else 0.05)

    @ax.task
    def _shortcut_pgdown(self):
        yield ax.attach_to(self._nav_fg, cancel_all=True)
        for i in itertools.count():
            self._item_view.apply_model(self._item_view.model.select_next(page_down=True))
            yield ax.sleep(0.5 if i == 0 else 0.05)

    @ax.task
    def _shortcut_down(self):
        yield ax.attach_to(self._nav_fg, cancel_all=True)
        for i in itertools.count():
            self._item_view.apply_model(self._item_view.model.select_next(row_down=True))
            yield ax.sleep(0.5 if i == 0 else 0.05)

    @ax.task
    def _shortcut_right(self):
        yield ax.attach_to(self._nav_fg, cancel_all=True)
        for i in itertools.count():
            self._item_view.apply_model(self._item_view.model.select_next())
            yield ax.sleep(0.5 if i == 0 else 0.05)


        