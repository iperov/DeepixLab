from typing import Self, overload

from .. import lx, mx, qt
from ..lib.collections import HFDict
from ._constants import WindowType
from .QApplication import QApplication
from .QBox import QVBox
from .QIconDB import IconDB, QIconDB



class QWindow(QVBox):
    """
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__q_window = None

        self.__state = HFDict()
        self.__state_first_time = True

        self.__mx_window_key_press = mx.Event1[qt.QKeyEvent]().dispose_with(self)
        self.__mx_window_key_release = mx.Event1[qt.QKeyEvent]().dispose_with(self)
        self.__mx_window_leave = mx.Event0().dispose_with(self)

        self._mx_settings.reflect(lambda settings, enter: self._apply_state(settings.state) if enter else ...)
        
        #self.q_widget.setBaseSize(16,16)

    @property
    def mx_window_key_press(self) -> mx.IEvent1_rv[qt.QKeyEvent]: return self.__mx_window_key_press
    @property
    def mx_window_key_release(self) -> mx.IEvent1_rv[qt.QKeyEvent]: return self.__mx_window_key_release
    @property
    def mx_window_leave(self) -> mx.IEvent0_rv: return self.__mx_window_leave

    @property
    def q_window_handle(self) -> qt.QWindow | None: return self.__q_window
    @property
    def q_window_icon(self) -> qt.QIcon: return self.q_widget.windowIcon()
    
    @property
    def window_flags(self) -> qt.Qt.WindowType: return self.q_widget.windowFlags()

    def activate(self):
        self.q_widget.activateWindow()
        return self

    def set_title(self, title : str|None = None):
        if (disp := getattr(self, '_QWindow_title_disp', None)) is not None:
            disp.dispose()
        self._QWindow_title_disp = QApplication.instance().mx_language.reflect(lambda lang: self.q_widget.setWindowTitle(lx.L(title, lang))).dispose_with(self)
        return self

    @overload
    def set_window_icon(self, icon : qt.QIcon) -> Self: ...
    @overload
    def set_window_icon(self, icon : IconDB, color : qt.QColor) -> Self: ...
    def set_window_icon(self, *args, **kwargs) -> Self:
        if len(args) == 1:
            arg0 = args[0]
            if isinstance(arg0, qt.QIcon):
                self.q_widget.setWindowIcon(args[0])
            else:
                if 'color' not in kwargs:
                    raise ValueError('color should be specified')
                args = (arg0, kwargs['color'])

        if len(args) == 2:
            self.set_window_icon(QIconDB.instance().icon(args[0], args[1]))

        return self

    def set_window_size(self, width : int, height : int):
        self.q_widget.setFixedWidth(width)
        self.q_widget.setFixedHeight(height)
        return self

    def set_window_flags(self, wnd_type : WindowType):
        self.q_widget.setWindowFlags(wnd_type)
        return self

    def _event_filter(self, object: qt.QObject, ev: qt.QEvent) -> bool:
        r = super()._event_filter(object, ev)

        ev_type = ev.type()
        if ev_type == ev.Type.KeyPress:
            self.__mx_window_key_press.emit(ev)
        elif ev_type == ev.Type.KeyRelease:
            self.__mx_window_key_release.emit(ev)
        elif ev_type == ev.Type.Leave:
            self.__mx_window_leave.emit()
        return r

    def _apply_state(self, state : HFDict = None):
        reload = False
        if state is not None:
            self.__state = state
            reload = True
            
        if self.visible:
            state_first_time, self.__state_first_time = self.__state_first_time, False
            if reload or state_first_time:
                    
                window_state = self.__state.get('windowState', qt.Qt.WindowState.WindowNoState.value)
                window_state = qt.Qt.WindowState(window_state)

                pos = self.__state.get('geometry.pos', None)
                size = self.__state.get('geometry.size', None)

                window_state_m = (window_state & qt.Qt.WindowState.WindowMaximized)  == qt.Qt.WindowState.WindowMaximized or \
                                 (window_state & qt.Qt.WindowState.WindowMinimized)  == qt.Qt.WindowState.WindowMinimized or \
                                 (window_state & qt.Qt.WindowState.WindowFullScreen) == qt.Qt.WindowState.WindowFullScreen

                if size is None or window_state_m:
                    size = (960,640)
                    
                if (self.window_flags & qt.Qt.WindowType.MSWindowsFixedSizeDialogHint) == 0:
                    self.q_widget.resize( qt.QSize(*size) )

                if not window_state_m and pos is not None:
                    self.q_widget.move(qt.QPoint(*pos))
                else:
                    # Center on screen
                    app : qt.QGuiApplication = qt.QApplication.instance()
                    screen_size = app.primaryScreen().size()
                    widget_width, widget_height = self.q_widget.size().width(), self.q_widget.size().height()
                    self.q_widget.move( (screen_size.width() - widget_width) // 2,  (screen_size.height() - widget_height) // 2 )

                if window_state is not None:
                    self.q_widget.setWindowState(qt.Qt.WindowState(window_state))

    def _show_event(self, ev: qt.QShowEvent):
        if self.q_object.parent() is not None:
            raise Exception(f'{self} must have no parent')

        if self.__q_window is None:
            self.__q_window = q_window = self.q_widget.windowHandle()
            q_window.installEventFilter(self.q_object)

        super()._show_event(ev)

        self._apply_state()


    def _hide_event(self, ev: qt.QHideEvent):
        super()._hide_event(ev)
        if self.__q_window is not None:
            self.__q_window.removeEventFilter(self.q_object)
            self.__q_window = None

    def _move_event(self, ev : qt.QMoveEvent):
        super()._move_event(ev)
        if self.visible:
            v = self.q_widget.windowState()
            if (v & qt.Qt.WindowState.WindowMaximized) == qt.Qt.WindowState.WindowNoState and \
               (v & qt.Qt.WindowState.WindowMinimized) == qt.Qt.WindowState.WindowNoState and \
               (v & qt.Qt.WindowState.WindowFullScreen) == qt.Qt.WindowState.WindowNoState:
                self.__state['geometry.pos'] = self.q_widget.pos().toTuple()

    def _resize_event(self, ev : qt.QResizeEvent):
        super()._resize_event(ev)

        if self.visible:
            v = self.q_widget.windowState()
            if (v & qt.Qt.WindowState.WindowMaximized) == qt.Qt.WindowState.WindowNoState and \
               (v & qt.Qt.WindowState.WindowMinimized) == qt.Qt.WindowState.WindowNoState and \
               (v & qt.Qt.WindowState.WindowFullScreen) == qt.Qt.WindowState.WindowNoState:
                self.__state['geometry.size'] = self.q_widget.size().toTuple()

    def _change_event(self, ev: qt.QEvent):
        super()._change_event(ev)
        if ev.type() == qt.QEvent.Type.WindowStateChange:
            v = self.q_widget.windowState()
            self.__state['windowState'] = v.value
            
            if (v & qt.Qt.WindowState.WindowMaximized) == qt.Qt.WindowState.WindowNoState and \
               (v & qt.Qt.WindowState.WindowMinimized) == qt.Qt.WindowState.WindowNoState and \
               (v & qt.Qt.WindowState.WindowFullScreen) == qt.Qt.WindowState.WindowNoState:
                self.__state['geometry.size'] = self.q_widget.size().toTuple()
                
