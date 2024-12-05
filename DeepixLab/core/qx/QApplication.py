from __future__ import annotations

from pathlib import Path
from typing import Set

from core.lib.collections import FDict

from .. import ax, mx, qt
from ..lib import os as lib_os
from ..lib import path as lib_path
from ..lib.collections import FDict, HFDict
from ._constants import ProcessPriority
from .QClipboard import QClipboard
from .QCursorDB import QCursorDB
from .QDarkFusionStyle import QDarkFusionStyle
from .QEvent import QEvent2
from .QFontDB import QFontDB
from .QAnimDB import QAnimDB
from .QIconDB import QIconDB
from .QObject import QObject
from .QSettings import QSettings
from .QTimer import QTimer


class QApplication(QObject):


    @staticmethod
    def instance() -> QApplication:
        if QApplication._instance is None:
            raise Exception('No QApplication instance.')
        return QApplication._instance

    def __init__(self, app_name : str = None, state_path : Path = None):
        """
        """
        if QApplication._instance is not None:
            raise Exception('QApplication instance already exists.')
        QApplication._instance = self
        QObject._QApplication = QApplication

        q_app = qt.QApplication.instance() or qt.QApplication()
        if not isinstance(q_app, qt.QApplication):
            raise ValueError('q_app must be an instance of QApplication')
        self.__q_app = q_app

        super().__init__(q_object=q_app)

        self.__state_path = state_path
        self.__rel_path = state_path.parent if state_path is not None else None
        
        self.__q_clipboard = QClipboard(q_clipboard=q_app.clipboard(), wrap_mode=True).dispose_with(self)
        QFontDB().dispose_with(self)
        QIconDB().dispose_with(self)
        QAnimDB().dispose_with(self)
        QCursorDB().dispose_with(self)

        q_app.deleteLater = lambda *_: ... # Supress QApplication object deletion
        q_app.setQuitOnLastWindowClosed(False)
        q_app.setApplicationName(app_name or 'QApplication')
        q_app.setFont(QFontDB.instance().default())
        q_app.setStyle(QDarkFusionStyle())
        self.__mx_focus_widget = mx.Property[qt.QWidget|None](None).dispose_with(self)

        self.__mx_focus_changed = QEvent2(q_app.focusChanged).dispose_with(self)
        self.__mx_focus_changed.listen(lambda old, new: self.__mx_focus_widget.set(new))

        self.__mx_language = mx.Property[str]('en').dispose_with(self)
        self.__mx_process_priority = mx.StateChoice[ProcessPriority](availuator=lambda: ProcessPriority).dispose_with(self)
        self.__mx_process_priority.listen(lambda prio, enter: lib_os.set_process_priority(prio) if enter else ... )

        self.__show_console = False
        self.__timer_counter = 0
        QTimer(self._on_timer).set_interval(0).start().dispose_with(self)

        self.__objects_settings_subscribers : Set[QObject] = set()
        self.__objects_settings = {}
        
        try:
            state = FDict.from_file(self.__state_path, Path_func=lambda path: lib_path.abspath(path, self.__rel_path))
        except:
            state = FDict()
        self._mx_settings.set( _QSettingsImpl( FDict(state) ) )

        self._mx_settings.reflect(lambda settings, enter, bag=mx.Disposable().dispose_with(self): 
                                  self.__ref_settings(settings, enter, bag))

    def __ref_settings(self, settings : QSettings, enter : bool, bag : mx.Disposable):
        if enter:
            for q_obj in self.__objects_settings_subscribers.copy():
                if q_obj in self.__objects_settings_subscribers:
                    old_q_settings = q_obj._mx_settings.swap(None)
                    if old_q_settings is not None:
                        old_q_settings.dispose()
                        
            self.__mx_language.set( settings.state.get('app_language', 'en') )
            self.__mx_process_priority.set(lib_os.ProcessPriority(settings.state.get('app_process_priority', lib_os.ProcessPriority.BELOW_NORMAL.value)))
            self.__show_console = settings.state.get('app_show_console', self.__show_console)
            
            self.__objects_settings = objects_settings = settings.state.get('app_objects_settings', FDict()).to_dict()
            
            for q_obj in self.__objects_settings_subscribers.copy():
                q_obj._mx_settings.set( _QSettingsImpl(FDict(objects_settings.get(q_obj.tree_name, {}))) )
                        
            settings.ev_update.listen(
                lambda: settings.state  .update({   'app_language' : self.__mx_language.get(),
                                                    'app_process_priority' : self.__mx_process_priority.get().value,
                                                    'app_show_console' : self.__show_console,
                                                    'app_objects_settings' : 
                                                        FDict(self.__objects_settings) |
                                                            {   q_obj.tree_name : (q_obj._mx_settings.get().ev_update.emit(), q_obj._mx_settings.get().state)[-1]
                                                                for q_obj in self.__objects_settings_subscribers  } 
                                                })  ).dispose_with(bag)
            
            
            
        else:
            bag.dispose_items()

    def _obj_settings_subscribe(self, q_obj : QObject):
        subscribers = self.__objects_settings_subscribers
        if q_obj in subscribers:
            raise Exception('already subscribed')
        subscribers.add(q_obj)

        q_obj._mx_settings.set( _QSettingsImpl(FDict(self.__objects_settings.get(q_obj.tree_name, {}))) )


    def _obj_settings_unsubscribe(self, q_obj : QObject):
        subscribers = self.__objects_settings_subscribers
        if q_obj not in subscribers:
            raise Exception('not subscribed')

        if (cur_q_settings := q_obj._mx_settings.get()) is not None:
            cur_q_settings.ev_update.emit()
            
            self.__objects_settings[q_obj.tree_name] = FDict(cur_q_settings.state)

            q_obj._mx_settings.set(None)
            cur_q_settings.dispose()

        subscribers.remove(q_obj)

    def __dispose__(self):
        super().__dispose__()
        
        old_q_settings = self._mx_settings.swap(None)
        old_q_settings.dispose()
        
        self.__q_app = None
        QApplication._instance = None

    @property
    def mx_language(self) -> mx.IProperty_v[str]:
        """Language state"""
        return self.__mx_language
    @property
    def mx_process_priority(self) -> mx.IStateChoice_v[ProcessPriority]:
        """"""
        return self.__mx_process_priority
    @property
    def mx_focus_widget(self) -> mx.IProperty_rv[qt.QWidget|None]:
        """Current focus widget."""
        return self.__mx_focus_widget
    
    @property
    def q_clipboard(self) -> QClipboard: return self.__q_clipboard

    def exec(self):
        lib_os.set_console_window(self.__show_console)
        self.__q_app.exec()
        lib_os.set_console_window(True)
        print('Exiting...')
        # Quitting
        self.save_state()
        
    def switch_console_window(self):
        show_console = self.__show_console = not lib_os.is_visible_console_window()
        lib_os.set_console_window(show_console)
        
    def quit(self):
        """request app to quit"""
        self.__q_app.quit()

    def reset_state(self):
        """Reinitialize app with default(empty) state"""
        old_q_settings = self._mx_settings.swap( _QSettingsImpl(FDict()) )
        old_q_settings.dispose()
        
    def save_state(self):
        """Save state to the file."""
        if self.__state_path is not None:
            q_settings = self._mx_settings.get()
            q_settings.ev_update.emit()
            
            FDict(q_settings.state).dump_to_file(self.__state_path, 
                                                 Path_func=lambda path: lib_path.relpath(path, self.__rel_path))


    def restore_override_cursor(self):
        self.__q_app.restoreOverrideCursor()

    def set_override_cursor(self, cursor : qt.QCursor | qt.Qt.CursorShape):
        if isinstance(cursor, qt.QCursor):
            cursor = cursor._get_q_cursor()
        self.__q_app.setOverrideCursor(cursor)

    def _on_timer(self, *_):
        ax.get_current_thread().execute_tasks_once()

        if self.__timer_counter % 10 == 0:
            lib_os.sleep_precise(0.001)

        self.__timer_counter += 1

    _instance : QApplication = None


class _QSettingsImpl(mx.Disposable, QSettings):
    def __init__(self, state : FDict):
        super().__init__()
        self._state = HFDict(state)
        self._ev_update = mx.Event0().dispose_with(self)

    @property
    def state(self) -> HFDict: return self._state
    @property
    def ev_update(self) -> mx.IEvent0_rv: return self._ev_update
