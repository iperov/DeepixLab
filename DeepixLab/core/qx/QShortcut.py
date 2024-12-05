from .. import mx, qt
from .QApplication import QApplication
from .QObject import QObject
from .QWidget import QWidget
from .QWindow import QWindow

func_keys = [   qt.Qt.Key.Key_Control, qt.Qt.Key.Key_Shift, qt.Qt.Key.Key_Alt,
                qt.Qt.Key.Key_Insert,
                qt.Qt.Key.Key_Delete,
                qt.Qt.Key.Key_PageUp,
                qt.Qt.Key.Key_PageDown,
                qt.Qt.Key.Key_Left,
                qt.Qt.Key.Key_Up,
                qt.Qt.Key.Key_Right,
                qt.Qt.Key.Key_Down,
                qt.Qt.Key.Key_Home,
                qt.Qt.Key.Key_End,
                qt.Qt.Key.Key_Tab,
                qt.Qt.Key.Key_F1,
                qt.Qt.Key.Key_F2,
                qt.Qt.Key.Key_F3,
                qt.Qt.Key.Key_F4,
                qt.Qt.Key.Key_F5,
                qt.Qt.Key.Key_F6,
                qt.Qt.Key.Key_F7,
                qt.Qt.Key.Key_F8,
                qt.Qt.Key.Key_F9,
                qt.Qt.Key.Key_F10,
                qt.Qt.Key.Key_F11,
                qt.Qt.Key.Key_F12,
            ]

class QShortcut(QObject):
    def __init__(self, key_comb : qt.QKeyCombination):
        """
        Shortcut.
        
        Should be parented in order to work. Can be disposed in order to be disabled.
        
        First parent QWidget in tree will be used to check visibility.
        """
        super().__init__()
        self._key_comb = key_comb
        self._anchor = None
        self._typing_focused = False
        self._pressed = False

        self._anchor_disp_bag = mx.Disposable().dispose_with(self)
        self._window_disp_bag = mx.Disposable().dispose_with(self)
        self._app_disp_bag = mx.Disposable().dispose_with(self)

        self._mx_press = mx.Event0().dispose_with(self)
        self._mx_release = mx.Event0().dispose_with(self)
        
    def __dispose__(self):
        self.release()
        super().__dispose__()

    @property
    def mx_press(self) -> mx.IEvent0_rv: return self._mx_press
    @property
    def mx_release(self) -> mx.IEvent0_rv: return self._mx_release
    
    @property
    def key_comb(self) -> qt.QKeyCombination: return self._key_comb
    
    def set_key_comb(self, key_comb : qt.QKeyCombination):
        self._key_comb = key_comb
        return self
    
    def press(self):
        self.release()
        if not self._pressed:
            self._pressed = True
            self._mx_press.emit()

    def release(self):
        if self._pressed:
            self._pressed = False
            self._mx_release.emit()
    
    def _visible_to_parent_event(self, parent : QObject):
        super()._visible_to_parent_event(parent)
        
        if isinstance(parent, QWidget):    
            # First parent QWidget will be used as anchor
            if self._anchor is None:
                anchor = self._anchor = parent
                try:
                    anchor.mx_hide.listen(lambda *_: self.release()).dispose_with(self._anchor_disp_bag)
                except:
                    import code
                    code.interact(local=dict(globals(), **locals()))

                
        if isinstance(parent, QWindow):
            parent.mx_window_key_press.listen(self._on_window_key_press).dispose_with(self._window_disp_bag)
            parent.mx_window_key_release.listen(self._on_window_key_release).dispose_with(self._window_disp_bag)
            parent.mx_window_leave.listen(lambda: self.release()).dispose_with(self._window_disp_bag)

        elif isinstance(parent, QApplication):
            parent.mx_focus_widget.reflect(self._on_app_mx_focus_widget).dispose_with(self._app_disp_bag)


    def _hiding_from_parent_event(self, parent : QObject):
        if isinstance(parent, QWidget):    
            # First parent QWidget
            if self._anchor is parent:
                self._anchor_disp_bag.dispose_items()
                self._anchor = None
                self.release()
                
        if isinstance(parent, QWindow):
            self._window_disp_bag.dispose_items()

        elif isinstance(parent, QApplication):
            self._app_disp_bag.dispose_items()
        
        super()._hiding_from_parent_event(parent)
        
    def _on_app_mx_focus_widget(self, widget : qt.QWidget):
        self._typing_focused = False

        if isinstance(widget, (qt.QLineEdit, qt.QTextEdit)):
            if not widget.isReadOnly():
                # Disable while focused on typing widgets
                self._typing_focused = True

        if self._typing_focused:
            self.release()

    
        
    def _on_window_key_press(self, ev : qt.QKeyEvent):
        #self._anchor.visible 
        if self._anchor is not None \
            and not self._typing_focused:
            if not ev.isAutoRepeat():
                #print(ev.key(), ev.modifiers(), ev.nativeVirtualKey(), ev.keyCombination())
                if ev.key() in func_keys:
                    key_comb = ev.keyCombination()
                else:
                    # Using native virtual key in order to ignore keyboard language
                    key_comb = qt.QKeyCombination(ev.modifiers(), qt.Qt.Key(ev.nativeVirtualKey()))
                    
                if (key_comb.key() in [qt.Qt.Key.Key_Control, qt.Qt.Key.Key_Shift, qt.Qt.Key.Key_Alt] and \
                    self._key_comb.key() == key_comb.key()) or self._key_comb == key_comb:
                        self.press()

    def _on_window_key_release(self, ev : qt.QKeyEvent):
        if not ev.isAutoRepeat():
            if ev.key() in func_keys:
                key_comb = ev.keyCombination()
            else:
                # Using native virtual key in order to ignore keyboard language
                key_comb = qt.QKeyCombination(ev.modifiers(), qt.Qt.Key(ev.nativeVirtualKey()))
            
            if self._key_comb.key() == key_comb.key():
                # Release by key, not modifier.
                self.release()



