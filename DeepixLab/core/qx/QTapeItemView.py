from .. import mx, qt
from ..lib.math import FVec2f
from ..qt import QPoint, QRect, QSize
from .FTapeItemView import FTapeItemView
from .QBox import QVBox
from .QScrollBar import QScrollBar
from .QWidget import QWidget
from .StyleColor import StyleColor


class QTapeItemView(QVBox):
    """"""
    
    def __init__(self):
        super().__init__()
        
        self._m = FTapeItemView()
        self._mx_current_item_id = mx.Property[int](-1, defer=lambda new_value, value, prop: self.apply_model(self._m.set_current_item_id(new_value))).dispose_with(self)
        
        self._q_canvas = _Canvas(self)
        
        self._q_scrollbar = QScrollBar()
        self._q_scrollbar_conn = self._q_scrollbar.mx_value.listen(lambda value: self.apply_model(self._m.scroll_to(value)))
        
        (self.add(self._q_canvas)
             .add(self._q_scrollbar.v_compact()))

    @property
    def mx_current_item_id(self) -> mx.IProperty_v[int]:
        """-1 if item_count==0"""
        return self._mx_current_item_id
    
    @property
    def model(self) -> FTapeItemView: return self._m

        
    def apply_model(self, new_m : FTapeItemView):
        m, self._m = self._m, new_m
        if new_m is m:
            # Nothing was changed
            return
        
        changed_scroll_value   = new_m.scroll_value != m.scroll_value
        
        
        upd =   new_m.item_count != m.item_count or \
                new_m.cli_size != m.cli_size or \
                changed_scroll_value 
        
        upd_geo = new_m.item_size != m.item_size or \
                  new_m.item_spacing != m.item_spacing

        if new_m.scroll_value_max != m.scroll_value_max:
            if new_m.scroll_value_max != 0:
                with self._q_scrollbar_conn.disabled_scope():
                    self._q_scrollbar.set_minimum(0).set_maximum(new_m.scroll_value_max).show()
            else:
                self._q_scrollbar.hide()

        if new_m.current_item_id != m.current_item_id:
            self._mx_current_item_id._set(new_m.current_item_id)
        
        
        if changed_scroll_value:
            with self._q_scrollbar_conn.disabled_scope():
                self._q_scrollbar.set_value(new_m.scroll_value)

        if upd:
            self.update()

        if upd_geo:
            self.update_geometry()
            self._q_canvas.update_geometry()

    def _on_paint_item(self, id : int, qp : qt.QPainter, rect : QRect):
        """overridable. Paint item content in given rect."""
        raise NotImplementedError()


class _Canvas(QWidget):
    # Canvas part of QTapeItemView

    def __init__(self, host : QTapeItemView):
        super().__init__()
        self._host = host
        self._qp = qt.QPainter()
    
    
    def _minimum_size_hint(self) -> QSize: 
        m = self._host._m
        
        return qt.QSize_from_FVec2( m.item_size + FVec2f(m.item_spacing*2, m.item_spacing*2) )
    
    def _leave_event(self, ev: qt.QEvent):
        super()._leave_event(ev)
        self._host.apply_model( self._host._m.mouse_leave())    
        
    def _resize_event(self, ev: qt.QResizeEvent):
        super()._resize_event(ev)
        self._host.apply_model( self._host._m.set_cli_size(ev.size().width(), ev.size().height()))

    def _mouse_press_event(self, ev: qt.QMouseEvent):
        super()._mouse_press_event(ev)
        if ev.button() == qt.Qt.MouseButton.LeftButton:
            self._host.apply_model( self._host._m.mouse_lbtn_down(qt.QPoint_to_FVec2f(ev.pos()), ctrl_pressed=qt.QApplication.keyboardModifiers() == qt.Qt.KeyboardModifier.ControlModifier,
                                                                shift_pressed=qt.QApplication.keyboardModifiers() == qt.Qt.KeyboardModifier.ShiftModifier))

    def _mouse_move_event(self, ev: qt.QMouseEvent):
        super()._mouse_move_event(ev)
        self._host.apply_model( self._host._m.mouse_move( qt.QPoint_to_FVec2f(ev.pos())) )

    def _mouse_release_event(self, ev: qt.QMouseEvent):
        super()._mouse_release_event(ev)
        if ev.button() == qt.Qt.MouseButton.LeftButton:
            self._host.apply_model( self._host._m.mouse_lbtn_up(qt.QPoint_to_FVec2f(ev.pos())) )

    def _wheel_event(self, ev: qt.QWheelEvent):
        super()._wheel_event(ev)
    
        self._host.apply_model( self._host._m.mouse_wheel(qt.QPointF_to_FVec2f(ev.position()),
                                                          ev.angleDelta().y(),
                                                          ctrl_pressed=qt.QApplication.keyboardModifiers() == qt.Qt.KeyboardModifier.ControlModifier,
                                                          shift_pressed=qt.QApplication.keyboardModifiers() == qt.Qt.KeyboardModifier.ShiftModifier) )
        

    def _paint_event(self, ev : qt.QPaintEvent):
        rect = self.rect
        qp = self._qp
        qp.begin(self.q_widget)

        
        host = self._host
        m = host._m
        item_spacing = m.item_spacing
        current_item_id = m.current_item_id
        
        mid_color = StyleColor.Text
        if m.item_count != 0:
            for item_id in range(m.v_item_start, m.v_item_start+m.v_item_count):
                item_rect = qt.QRect_from_FBBox(m.get_item_cli_box(item_id))

                if current_item_id == item_id:
                    select_rect = item_rect.adjusted(-item_spacing, -item_spacing, item_spacing, item_spacing)
                    qp.fillRect(select_rect, mid_color)

                host._on_paint_item(item_id, qp, item_rect)

        else:
            qp.drawText(rect, qt.Qt.AlignmentFlag.AlignCenter, '- @(no_items) -') 

        qp.end()
        

