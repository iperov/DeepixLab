from .. import lx, mx, qt
from ._constants import LayoutDirection
from ._helpers import q_init
from .QApplication import QApplication
from .QFontDB import FontDB, QFontDB
from .QFuncWrap import QFuncWrap
from .QObject import QObject


class QWidget(QObject):

    def __init__(self, **kwargs):
        super().__init__(q_object=q_init('q_widget', qt.QWidget, **kwargs), **kwargs)

        self._min_width = None
        self._min_height = None
        self._max_width = None
        self._max_height = None

        self._mx_show = mx.Event1[qt.QShowEvent]().dispose_with(self)
        self._mx_hide = mx.Event1[qt.QHideEvent]().dispose_with(self)
        self._mx_close = mx.Event1[qt.QCloseEvent]().dispose_with(self)
        self._mx_change = mx.Event1[qt.QEvent]().dispose_with(self)
        self._mx_resize = mx.Event1[qt.QResizeEvent]().dispose_with(self)
        self._mx_focus_in = mx.Event1[qt.QFocusEvent]().dispose_with(self)
        self._mx_focus_out = mx.Event1[qt.QFocusEvent]().dispose_with(self)
        self._mx_enter = mx.Event1[qt.QEnterEvent]().dispose_with(self)
        self._mx_leave = mx.Event1[qt.QEvent]().dispose_with(self)
        self._mx_move = mx.Event1[qt.QMoveEvent]().dispose_with(self)
        self._mx_key_press = mx.Event1[qt.QKeyEvent]().dispose_with(self)
        self._mx_key_release = mx.Event1[qt.QKeyEvent]().dispose_with(self)
        self._mx_mouse_move = mx.Event1[qt.QMouseEvent]().dispose_with(self)
        self._mx_mouse_press = mx.Event1[qt.QMouseEvent]().dispose_with(self)
        self._mx_mouse_release = mx.Event1[qt.QMouseEvent]().dispose_with(self)
        self._mx_wheel = mx.Event1[qt.QWheelEvent]().dispose_with(self)
        self._mx_paint = mx.Event1[qt.QPaintEvent]().dispose_with(self)

        self._minimumSizeHint_wrap = QFuncWrap(q_widget := self.q_widget, 'minimumSizeHint', lambda *args, **kwargs: self._minimum_size_hint(*args, **kwargs)).dispose_with(self)
        self._sizeHint_wrap = QFuncWrap(q_widget, 'sizeHint', lambda *args, **kwargs: self._size_hint(*args, **kwargs)).dispose_with(self)
        self._showEvent_wrap = QFuncWrap(q_widget, 'showEvent', lambda *args, **kwargs: self._show_event(*args, **kwargs)).dispose_with(self)
        self._hideEvent_wrap = QFuncWrap(q_widget, 'hideEvent', lambda *args, **kwargs: self._hide_event(*args, **kwargs)).dispose_with(self)
        self._closeEvent_wrap = QFuncWrap(q_widget, 'closeEvent', lambda *args, **kwargs: self._close_event(*args, **kwargs)).dispose_with(self)
        self._changeEvent_wrap = QFuncWrap(q_widget, 'changeEvent', lambda *args, **kwargs: self._change_event(*args, **kwargs)).dispose_with(self)
        self._resizeEvent_wrap = QFuncWrap(q_widget, 'resizeEvent', lambda *args, **kwargs: self._resize_event(*args, **kwargs)).dispose_with(self)
        self._focusInEvent_wrap = QFuncWrap(q_widget, 'focusInEvent', lambda *args, **kwargs: self._focus_in_event(*args, **kwargs)).dispose_with(self)
        self._focusOutEvent_wrap = QFuncWrap(q_widget, 'focusOutEvent', lambda *args, **kwargs: self._focus_out_event(*args, **kwargs)).dispose_with(self)
        self._enterEvent_wrap = QFuncWrap(q_widget, 'enterEvent', lambda *args, **kwargs: self._enter_event(*args, **kwargs)).dispose_with(self)
        self._leaveEvent_wrap = QFuncWrap(q_widget, 'leaveEvent', lambda *args, **kwargs: self._leave_event(*args, **kwargs)).dispose_with(self)
        self._moveEvent_wrap = QFuncWrap(q_widget, 'moveEvent', lambda *args, **kwargs: self._move_event(*args, **kwargs)).dispose_with(self)
        self._keyPressEvent_wrap = QFuncWrap(q_widget, 'keyPressEvent', lambda *args, **kwargs: self._key_press_event(*args, **kwargs)).dispose_with(self)
        self._keyReleaseEvent_wrap = QFuncWrap(q_widget, 'keyReleaseEvent', lambda *args, **kwargs: self._key_release_event(*args, **kwargs)).dispose_with(self)
        self._mouseMoveEvent_wrap = QFuncWrap(q_widget, 'mouseMoveEvent', lambda *args, **kwargs: self._mouse_move_event(*args, **kwargs)).dispose_with(self)
        self._mousePressEvent_wrap = QFuncWrap(q_widget, 'mousePressEvent', lambda *args, **kwargs: self._mouse_press_event(*args, **kwargs)).dispose_with(self)
        self._mouseReleaseEvent_wrap = QFuncWrap(q_widget, 'mouseReleaseEvent', lambda *args, **kwargs: self._mouse_release_event(*args, **kwargs)).dispose_with(self)
        self._wheelEvent_wrap = QFuncWrap(q_widget, 'wheelEvent', lambda *args, **kwargs: self._wheel_event(*args, **kwargs)).dispose_with(self)
        self._paintEvent_wrap = QFuncWrap(q_widget, 'paintEvent', lambda *args, **kwargs: self._paint_event(*args, **kwargs)).dispose_with(self)

        if not self._is_wrap_mode():
            self.h_normal()
            self.v_normal()

    @property
    def mx_show(self) -> mx.IEvent1_rv[qt.QShowEvent]: return self._mx_show
    @property
    def mx_hide(self) -> mx.IEvent1_rv[qt.QHideEvent]: return self._mx_hide
    @property
    def mx_close(self) -> mx.IEvent1_rv[qt.QCloseEvent]: return self._mx_close
    @property
    def mx_change(self) -> mx.IEvent1_rv[qt.QEvent]: return self._mx_change
    @property
    def mx_resize(self) -> mx.IEvent1_rv[qt.QResizeEvent]: return self._mx_resize
    @property
    def mx_focus_in(self) -> mx.IEvent1_rv[qt.QFocusEvent]: return self._mx_focus_in
    @property
    def mx_focus_out(self) -> mx.IEvent1_rv[qt.QFocusEvent]: return self._mx_focus_out
    @property
    def mx_enter(self) -> mx.IEvent1_rv[qt.QEnterEvent]: return self._mx_enter
    @property
    def mx_leave(self) -> mx.IEvent1_rv[qt.QEvent]: return self._mx_leave
    @property
    def mx_move(self) -> mx.IEvent1_rv[qt.QMoveEvent]: return self._mx_move
    @property
    def mx_key_press(self) -> mx.IEvent1_rv[qt.QKeyEvent]: return self._mx_key_press
    @property
    def mx_key_release(self) -> mx.IEvent1_rv[qt.QKeyEvent]: return self._mx_key_release
    @property
    def mx_mouse_move(self) -> mx.IEvent1_rv[qt.QMouseEvent]: return self._mx_mouse_move
    @property
    def mx_mouse_press(self) -> mx.IEvent1_rv[qt.QMouseEvent]: return self._mx_mouse_press
    @property
    def mx_mouse_release(self) -> mx.IEvent1_rv[qt.QMouseEvent]: return self._mx_mouse_release
    @property
    def mx_wheel(self) -> mx.IEvent1_rv[qt.QWheelEvent]: return self._mx_wheel
    @property
    def mx_paint(self) -> mx.IEvent1_rv[qt.QPaintEvent]: return self._mx_paint

    def __dispose__(self):
        if not self._is_wrap_mode():
            self.q_widget.hide()
        super().__dispose__()

    @property
    def q_widget(self) -> qt.QWidget: return self.q_object
    @property
    def q_font(self) -> qt.QFont: return self.q_widget.font()
    @property
    def q_style(self) -> qt.QStyle: return self.q_widget.style()
    @property
    def q_palette(self) -> qt.QPalette: return self.q_widget.palette()
    @property
    def rect(self) -> qt.QRect: return self.q_widget.rect()
    @property
    def size(self) -> qt.QSize: return self.q_widget.size()
    @property
    def visible(self) -> bool: return self.q_widget.isVisible()
    
    def map_to_global(self, point : qt.QPoint) -> qt.QPoint: return self.q_widget.mapToGlobal(point)

    def h_compact(self, size : int = None):
        self._min_width = size
        self._max_width = size
        self.q_widget.setSizePolicy(qt.QSizePolicy.Policy.Fixed, self.q_widget.sizePolicy().verticalPolicy())
        return self

    def v_compact(self, size : int = None):
        self._min_height = size
        self._max_height = size
        self.q_widget.setSizePolicy(self.q_widget.sizePolicy().horizontalPolicy(), qt.QSizePolicy.Policy.Fixed)
        return self

    def h_normal(self, min : int = None, max : int = None):
        self._min_width = min
        self._max_width = max
        
        self.q_widget.setSizePolicy(qt.QSizePolicy.Policy.Preferred, self.q_widget.sizePolicy().verticalPolicy())
        return self

    def v_normal(self, min : int = None, max : int = None):
        self._min_height = min
        self._max_height = max
        self.q_widget.setSizePolicy(self.q_widget.sizePolicy().horizontalPolicy(), qt.QSizePolicy.Policy.Preferred)
        return self

    def h_expand(self, min : int = None, max : int = None):
        self._min_width = min
        self._max_width = max
        
        self.q_widget.setSizePolicy(qt.QSizePolicy.Policy.Expanding, self.q_widget.sizePolicy().verticalPolicy())
        return self

    def v_expand(self, min : int = None, max : int = None):
        self._min_height = min
        self._max_height = max
        self.q_widget.setSizePolicy(self.q_widget.sizePolicy().horizontalPolicy(), qt.QSizePolicy.Policy.Expanding)
        return self

    def show(self):
        self.set_visible(True)
        return self

    def hide(self):
        self.set_visible(False)
        return self

    def enable(self):
        self.set_enabled(True)
        return self

    def disable(self):
        self.set_enabled(False)
        return self

    def clear_focus(self):
        self.q_widget.clearFocus()
        return self

    def set_focus(self):
        self.q_widget.setFocus()
        return self

    def set_font(self, font : qt.QFont | FontDB):
        if isinstance(font, FontDB):
            font = QFontDB.instance().get(font)
        self.q_widget.setFont(font)
        return self

    def set_cursor(self, cursor : qt.QCursor|qt.Qt.CursorShape|qt.QPixmap):
        self.q_widget.setCursor(cursor)
        return self

    def unset_cursor(self):
        self.q_widget.unsetCursor()
        return self

    def set_layout_direction(self, layout_direction : LayoutDirection):
        self.q_widget.setLayoutDirection(layout_direction)
        return self

    def set_mouse_tracking(self, b : bool):
        self.q_widget.setMouseTracking(b)
        return self

    def set_visible(self, visible : bool):
        self.q_widget.setVisible(visible)
        return self

    def set_enabled(self, enabled : bool):
        self.q_widget.setEnabled(enabled)
        return self

    def set_tooltip(self, tooltip : str):
        if (disp := getattr(self, '_QWidget_tooltip_disp', None)) is not None:
            disp.dispose()
        self._QWidget_tooltip_disp = QApplication.instance().mx_language.reflect(lambda lang: self.q_widget.setToolTip(lx.L(tooltip, lang))).dispose_with(self)
        return self

    def set_updates_enabled(self, enable : bool):
        self.q_widget.setUpdatesEnabled(enable)
        return self

    def repaint(self):
        self.q_widget.repaint()
        return self

    def update(self):
        self.q_widget.update()
        return self

    def update_geometry(self):
        self.q_widget.updateGeometry()
        return self

    def _minimum_size_hint(self) -> qt.QSize:
        """inheritable/overridable"""
        size = self._minimumSizeHint_wrap.get_super()()
        if (min_width := self._min_width) is not None:
            size = qt.QSize(min_width, size.height())
        if (min_height := self._min_height) is not None:
            size = qt.QSize(size.width(), min_height)
        return size

    def _size_hint(self) -> qt.QSize:
        """inheritable/overridable"""
        size = self._sizeHint_wrap.get_super()()
        if (max_width := self._max_width) is not None:
            size = qt.QSize(max_width, size.height())
        if (max_height := self._max_height) is not None:
            size = qt.QSize(size.width(), max_height)
        return size

    def _show_event(self, ev : qt.QShowEvent):
        """inheritable"""
        self._showEvent_wrap.get_super()(ev)
        self._mx_show.emit(ev)

    def _hide_event(self, ev : qt.QHideEvent):
        """inheritable"""
        self._hideEvent_wrap.get_super()(ev)
        self._mx_hide.emit(ev)

    def _close_event(self, ev : qt.QHideEvent):
        """inheritable"""
        self._closeEvent_wrap.get_super()(ev)
        self._mx_close.emit(ev)

    def _change_event(self, ev : qt.QEvent):
        """inheritable"""
        self._changeEvent_wrap.get_super()(ev)
        self._mx_change.emit(ev)

    def _resize_event(self, ev : qt.QResizeEvent):
        """inheritable"""
        self._resizeEvent_wrap.get_super()(ev)
        self._mx_resize.emit(ev)

    def _focus_in_event(self, ev : qt.QFocusEvent):
        """inheritable"""
        self._focusInEvent_wrap.get_super()(ev)

    def _focus_out_event(self, ev : qt.QFocusEvent):
        """inheritable"""
        self._focusOutEvent_wrap.get_super()(ev)

    def _enter_event(self, ev : qt.QEnterEvent):
        """inheritable"""
        self._enterEvent_wrap.get_super()(ev)
        self._mx_enter.emit(ev)

    def _leave_event(self, ev : qt.QEvent):
        """inheritable"""
        self._leaveEvent_wrap.get_super()(ev)
        self._mx_leave.emit(ev)

    def _move_event(self, ev : qt.QMoveEvent):
        """inheritable"""
        self._moveEvent_wrap.get_super()(ev)
        self._mx_move.emit(ev)

    def _key_press_event(self, ev : qt.QKeyEvent):
        """inheritable"""
        #if ev.key() != qt.Qt.Key.Key_Tab:
        self._keyPressEvent_wrap.get_super()(ev)
        self._mx_key_press.emit(ev)

    def _key_release_event(self, ev : qt.QKeyEvent):
        """inheritable"""
        self._keyReleaseEvent_wrap.get_super()(ev)
        self._mx_key_release.emit(ev)

    def _mouse_move_event(self, ev : qt.QMouseEvent):
        """inheritable"""
        self._mouseMoveEvent_wrap.get_super()(ev)
        self._mx_mouse_move.emit(ev)

    def _mouse_press_event(self, ev : qt.QMouseEvent):
        """inheritable"""
        self._mousePressEvent_wrap.get_super()(ev)
        self._mx_mouse_press.emit(ev)

    def _mouse_release_event(self, ev : qt.QMouseEvent):
        """inheritable"""
        self._mouseReleaseEvent_wrap.get_super()(ev)
        self._mx_mouse_release.emit(ev)

    def _wheel_event(self, ev : qt.QWheelEvent):
        """inheritable"""
        self._wheelEvent_wrap.get_super()(ev)
        self._mx_wheel.emit(ev)

    def _paint_event(self, ev : qt.QPaintEvent):
        """inheritable"""
        self._paintEvent_wrap.get_super()(ev)
        self._mx_paint.emit(ev)
