from .. import mx, qt
from ._helpers import q_init
from .QAbstractSpinBox import QAbstractSpinBox
from .QEvent import QEvent1
from .QFontDB import FontDB


class QDoubleSpinBox(QAbstractSpinBox):
    ButtonSymbols = qt.QDoubleSpinBox.ButtonSymbols

    def __init__(self, **kwargs):
        super().__init__(q_abstract_spin_box=q_init('q_double_spin_box', qt.QDoubleSpinBox, **kwargs), **kwargs)
        
        q_double_spin_box = self.get_q_double_spin_box()

        self._mx_value = mx.GetSetProperty[float](self.get_value, self.set_value, QEvent1[float](q_double_spin_box.valueChanged).dispose_with(self) ).dispose_with(self)
        self._moving = False
        
        self.set_font(FontDB.Digital)
        self.set_mouse_tracking(True)

    @property
    def mx_value(self) -> mx.IProperty_v[float]: return self._mx_value

    def get_q_double_spin_box(self) -> qt.QDoubleSpinBox: return self.q_abstract_spin_box

    def get_value(self) -> float: return self.get_q_double_spin_box().value()

    def get_single_step(self) -> float:
        return self.get_q_double_spin_box().singleStep()

    def set_value(self, value : float):
        self.get_q_double_spin_box().setValue(value)
        return self

    def set_single_step(self, step : float):
        self.get_q_double_spin_box().setSingleStep(step)
        return self

    def set_decimals(self, prec : int):
        self.get_q_double_spin_box().setDecimals(prec)
        return self

    def set_minimum(self, min : float):
        self.get_q_double_spin_box().setMinimum(min)
        return self

    def set_maximum(self, max : float):
        self.get_q_double_spin_box().setMaximum(max)
        return self

    def set_button_symbols(self, bs : ButtonSymbols):
        self.get_q_double_spin_box().setButtonSymbols(bs)
        return self

    def _resize_event(self, ev: qt.QResizeEvent):
        super()._resize_event(ev)

        opt = qt.QStyleOptionSpinBox()
        opt.initFrom(self.q_widget)

        style = self.q_style
        up_rect = style.subControlRect( qt.QStyle.ComplexControl.CC_SpinBox, opt, qt.QStyle.SubControl.SC_SpinBoxUp)
        down_rect = style.subControlRect( qt.QStyle.ComplexControl.CC_SpinBox, opt, qt.QStyle.SubControl.SC_SpinBoxDown)
        self._buttons_rect = up_rect.united(down_rect)

    def _mouse_press_event(self, ev: qt.QMouseEvent):
        pos = ev.pos()

        if self._buttons_rect.contains(pos):
            if not self._moving:
                self._moving = True
                self._moving_start_pos = pos
                self._moving_delta = 0
        else:
            super()._mouse_press_event(ev)


    def _mouse_move_event(self, ev: qt.QMouseEvent):
        pos = ev.pos()

        if self._buttons_rect.contains(pos):
            self.set_cursor(qt.Qt.CursorShape.SizeVerCursor)
        else:
            self.unset_cursor()

        if self._moving:
            self._moving_delta -= (ev.pos() - self._moving_start_pos).y() / 5

            delta_int = int(self._moving_delta)
            if delta_int != 0:
                self._moving_delta -= delta_int
                self.set_value( self.get_value() + delta_int*self.get_single_step() )

            qt.QCursor.setPos( self.map_to_global(self._moving_start_pos) )
        else:
            super()._mouse_move_event(ev)

    def _mouse_release_event(self, ev: qt.QMouseEvent):
        super()._mouse_release_event(ev)
        self._finish_moving()

    def _leave_event(self, ev: qt.QEvent):
        super()._leave_event(ev)
        self._finish_moving()

    def _finish_moving(self):
        if self._moving:
            self._moving = False



