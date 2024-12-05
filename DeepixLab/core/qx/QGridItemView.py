from __future__ import annotations

from datetime import datetime

from .. import ax, mx, qt, lx
from ..lib.collections import FIndices
from ..lib.math import FVec2f
from ..qt import QRect, QSize
from ._constants import Orientation
from .FGridItemView import FGridItemView
from .QApplication import QApplication
from .QBox import QHBox
from .QScrollBar import QScrollBar
from .QWidget import QWidget
from .StyleColor import StyleColor


class QGridItemView(QHBox):
    """
    Direct Grid Item View 
    Able to show tens of thousand items without lags.
    """

    def __init__(self):
        super().__init__()
        self._m = FGridItemView()
        
        self._fg = ax.FutureGroup().dispose_with(self)
        self._mx_model = mx.Property[FGridItemView](self._m).dispose_with(self)
        self._mx_selected_items = mx.Property[FIndices](FIndices()).dispose_with(self)
        self._mx_marked_items = mx.Property[FIndices](FIndices()).dispose_with(self)
        
        self._q_canvas = _Canvas(self)
        q_scrollbar = self._q_scrollbar = QScrollBar().set_orientation(Orientation.Vertical).h_compact()
        self._q_scrollbar_conn = q_scrollbar.mx_value.listen(lambda value: self.apply_model(self._m.scroll_to_value(value)))

        (self.add(self._q_canvas)
             .add(self._q_scrollbar))

        self._m_update_task()
        
        QApplication.instance().mx_language.reflect(lambda lang: setattr(self, '_lang', lang)).dispose_with(self)
    
        
    @property
    def model(self) -> FGridItemView: return self._m
    
    @property
    def mx_model(self) -> mx.IProperty_rv[FGridItemView]: return self._mx_model
    @property
    def mx_selected_items(self) -> mx.IProperty_rv[FIndices]: return self._mx_selected_items
    @property
    def mx_marked_items(self) -> mx.IProperty_rv[FIndices]: return self._mx_marked_items


    def apply_model(self, new_m : FGridItemView):
        m, self._m = self._m, new_m
        if new_m is m:
            # Nothing was changed
            return

        changed_selected_idxs = not (new_m.selected_items is m.selected_items)
        changed_marked_idxs   = not (new_m.marked_items is m.marked_items)
        changed_scroll_value  = new_m.scroll_value != m.scroll_value

        upd =   new_m.cli_size != m.cli_size or \
                new_m.item_count != m.item_count or \
                changed_scroll_value or \
                new_m.selecting_mode != m.selecting_mode or \
                (new_m.selecting_mode != 0 and new_m.mouse_cli_pt != m.mouse_cli_pt) or \
                changed_selected_idxs or changed_marked_idxs

        upd_geo = new_m.item_size != m.item_size or \
                  new_m.item_spacing != m.item_spacing

        if new_m.scroll_value_max != m.scroll_value_max:
            if new_m.scroll_value_max != 0:
                with self._q_scrollbar_conn.disabled_scope():
                    self._q_scrollbar.set_minimum(0).set_maximum(new_m.scroll_value_max).show()
            else:
                self._q_scrollbar.hide()

        if changed_scroll_value:
            with self._q_scrollbar_conn.disabled_scope():
                self._q_scrollbar.set_value(new_m.scroll_value)

        if changed_selected_idxs:
            self._mx_selected_items.set( new_m.selected_items )
        if changed_marked_idxs:
            self._mx_marked_items.set( new_m.marked_items )

        if upd:
            self.update()

        if upd_geo:
            self.update_geometry()
            self._q_canvas.update_geometry()
            
        self._mx_model.set(new_m)

    def _on_paint_item(self, item_id : int, qp : qt.QPainter, rect : qt.QRect):
        """overridable. Paint item content in given rect."""
        raise NotImplementedError()
    
    
    @ax.task
    def _m_update_task(self):
        yield ax.attach_to(self._fg)

        t = datetime.now().timestamp()
        while True:
            dt = datetime.now().timestamp() - t

            self.apply_model(self._m.update(dt))
            t += dt
            yield ax.sleep(1.0 / 15.0)


class _Canvas(QWidget):
    # Canvas part of QGridItemView

    def __init__(self, host : QGridItemView):
        super().__init__()
        self._host = host
        self._qp = qt.QPainter()
        
    def _minimum_size_hint(self) -> QSize: 
        m = self._host._m
        return qt.QSize_from_FVec2(m.item_size + FVec2f(m.item_spacing, m.item_spacing))

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
                                                        shift_pressed=qt.QApplication.keyboardModifiers() == qt.Qt.KeyboardModifier.ShiftModifier,
                                                        ) )


    def _paint_event(self, ev: qt.QPaintEvent):
        qp = self._qp
        qp.begin(self.q_widget)

        m = self._host._m
        

        if m.item_count == 0:
            rect = QRect(0, 0, m.cli_size.x, m.cli_size.y)
            qp.drawText(rect, qt.Qt.AlignmentFlag.AlignCenter, lx.L('- @(no_items) -', self._host._lang) )
        else:
            selection_frame_q_color = StyleColor.SelectColor
            marked_frame_q_color = qt.QColor(selection_frame_q_color)
            marked_frame_q_color.setAlpha(110)
            
            if m.v_item_count != 0:

                selection_aabb = m.selection_aabb

                for item_id in range(m.v_item_start, m.v_item_start+m.v_item_count):
                    item_rect = qt.QRect_from_FBBox(m.get_item_cli_box(item_id))
                    
                    idx_in_aabb = False
                    if selection_aabb is not None:
                        row = item_id // m.v_col_count
                        col = item_id % m.v_col_count

                        row_aa, row_bb, col_aa, col_bb = selection_aabb
                        idx_in_aabb = row >= row_aa and row <= row_bb and col >= col_aa and col <= col_bb

                    selected = m.is_selected(item_id)
                    marked = m.is_marked(item_id)

                    selected = (m._selecting_mode == 0 and selected) or \
                            (m._selecting_mode == 1 and ((not selected and     idx_in_aabb) or \
                                                                (selected and not idx_in_aabb))) \
                            or (m._selecting_mode == 2 and (selected or idx_in_aabb) )

                    if selected:
                        margin = m.item_spacing // 2
                        item_outter_rect = item_rect.marginsAdded(qt.QMargins(margin,margin,margin,margin))
                        qp.fillRect(item_outter_rect, selection_frame_q_color)
                        
                    self._host._on_paint_item(item_id, qp, item_rect)

                    if marked:
                        qp.fillRect(item_rect, marked_frame_q_color)

                if m._selecting_mode != 0:
                    qp.fillRect(qt.QRect_from_FBBox(m.selection_cli_box), qt.QColor(255,255,255,127))

        qp.end()
