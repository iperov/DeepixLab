from __future__ import annotations

import itertools
from collections import deque

from core import ax, mx, qt, qx
from core.lib.image import FImage
from core.lib.math import FAffMat2, FRectf

from .FTransformEditor import FTransformEditor


class QxTransformEditor(qx.QHBox):
    """Generic image transform editor"""

    def __init__(self, image : FImage):
        """
            image       HWC
        """
        super().__init__()

        self._mx_quit_ev = mx.Event1[bool]().dispose_with(self)

        H,W,_ = image.shape

        self._image = image
        self._image_pixmap = qt.QPixmap_from_np(image.bgr().HWC())

        self._m = FTransformEditor(W,H)
        self._m_undo = deque([self._m])
        self._m_redo = deque()

        self._q_left_panel_vbox = qx.QVBox()
        self._q_central_panel_vbox = qx.QVBox()

        self.add(qx.QSplitter()
                    .add( qx.QVScrollArea().h_compact().set_widget(
                        self._q_left_panel_vbox.set_spacing(1).v_compact() ))
                    .add(self._q_central_panel_vbox) )

        self._mx_settings.reflect(lambda settings, enter, bag=mx.Disposable().dispose_with(self):
                                   self.__ref_settings(settings, enter, bag))


    def __ref_settings(self, settings : qx.QSettings, enter : bool, bag : mx.Disposable):
        if not enter:
            bag.dispose_items()
        else:
            self.__settings = settings

            self._q_canvas = _QCanvas(self).dispose_with(bag)

            shortcut_exit = qx.QShortcut(qt.QKeyCombination(qt.Qt.KeyboardModifier.ControlModifier, qt.Qt.Key.Key_Q)).set_parent(self._q_canvas)
            shortcut_exit.mx_press.listen(lambda: self._mx_quit_ev.emit(False) )

            shortcut_save_and_exit = qx.QShortcut(qt.QKeyCombination(qt.Qt.KeyboardModifier.ControlModifier, qt.Qt.Key.Key_S)).set_parent(self._q_canvas)
            shortcut_save_and_exit.mx_press.listen(lambda: self._mx_quit_ev.emit(True) )

            undo_redo_fg = ax.FutureGroup().dispose_with(self)
            undo_shortcut = qx.QShortcut( qt.QKeyCombination(qt.Qt.KeyboardModifier.ControlModifier, qt.Qt.Key.Key_Z)).set_parent(self._q_canvas)
            undo_shortcut.mx_press.listen(lambda: self._undo_model(undo_redo_fg))
            undo_shortcut.mx_release.listen(lambda: undo_redo_fg.cancel_all())

            redo_shortcut = qx.QShortcut( qt.QKeyCombination(qt.Qt.KeyboardModifier.ControlModifier|qt.Qt.KeyboardModifier.ShiftModifier, qt.Qt.Key.Key_Z)).set_parent(self._q_canvas)
            redo_shortcut.mx_press.listen(lambda: self._redo_model(undo_redo_fg))
            redo_shortcut.mx_release.listen(lambda: undo_redo_fg.cancel_all())

            q_exit_btn = qx.QPushButton().set_text(f'@(Quit) {qx.hfmt.colored_shortcut_keycomb(shortcut_exit)}')
            q_exit_btn.mx_clicked.listen(lambda: shortcut_exit.press())

            q_exit_save_btn = qx.QPushButton().set_text(f'@(Quit) and save {qx.hfmt.colored_shortcut_keycomb(shortcut_save_and_exit)}')
            q_exit_save_btn.mx_clicked.listen(lambda: shortcut_save_and_exit.press())

            self._q_left_panel_vbox.add(
                qx.QVBox().set_spacing(4).dispose_with(bag)
                    .add(qx.QCollapsibleVBox().set_text('@(File)').inline(lambda collapsible: collapsible.content_vbox

                        .add(q_exit_btn)
                        .add(q_exit_save_btn))))

            self._q_central_panel_vbox.add(self._q_canvas)

            self._cursor_rotate = qx.QIconDB.instance().cursor(qx.IconDB.refresh_outline)


    @property
    def mx_quit_ev(self) -> mx.IEvent1_rv[bool]:
        """event called when QxMaskEditor done working, so object can be disposed"""
        return self._mx_quit_ev

    @property
    def model(self) -> FTransformEditor: return self._m

    def apply_model(self, new_m : FTransformEditor):
        m, self._m = self._m, new_m

        changed_w_img_rect = new_m.w_img_rect != m.w_img_rect
        changed_hovering_ctrl_circle_edge   = new_m.is_hovering_ctrl_circle_edge() != m.is_hovering_ctrl_circle_edge()
        changed_hovering_ctrl_inside_circle = new_m.is_hovering_ctrl_circle_inside() != m.is_hovering_ctrl_circle_inside()

        upd_qvp = ( new_m.cli_size != m.cli_size or
                    new_m.view_proj != m.view_proj or
                    changed_w_img_rect or
                    changed_hovering_ctrl_circle_edge or
                    changed_hovering_ctrl_inside_circle
                    )

        upd_cursor = changed_hovering_ctrl_circle_edge \
                    or changed_hovering_ctrl_inside_circle

        if new_m.is_changed_for_undo(m):
            self._m_undo.append(m)
            self._m_redo = deque()

        if upd_cursor:
            self._update_cursor()

        if upd_qvp:
            self._q_canvas.update()


    def _update_cursor(self):
        m = self._m

        if m.is_hovering_ctrl_circle_edge():
            cursor = qt.Qt.CursorShape.ClosedHandCursor
            cursor = self._cursor_rotate
        elif m.is_hovering_ctrl_circle_inside():
            cursor = qt.Qt.CursorShape.SizeAllCursor
        else:
            cursor = qt.Qt.CursorShape.ArrowCursor

        self._q_canvas.set_cursor(cursor)


    @ax.task
    def _undo_model(self, fg : ax.FutureGroup):
        yield ax.attach_to(fg, cancel_all=True)

        for i in itertools.count():

            if len(self._m_undo) > 1:
                if len(self._m_redo) == 0:
                    self._m_redo.appendleft(self._m)

                self._m_redo.appendleft(self._m_undo.pop())
                self._m = self._m_redo[0].recover_from_undo_redo(self._m)

                self._update_cursor()
                self._q_canvas.update()

            yield ax.sleep(0.4 if i == 0 else 0.1)

    @ax.task
    def _redo_model(self, fg : ax.FutureGroup):
        yield ax.attach_to(fg, cancel_all=True)

        for i in itertools.count():
            if len(self._m_redo) != 0:
                self._m_undo.append(self._m_redo.popleft())
                self._m = self._m_redo[0].recover_from_undo_redo(self._m)

                if len(self._m_redo) == 1:
                    self._m_redo = deque()

                self._update_cursor()
                self._q_canvas.update()

            yield ax.sleep(0.4 if i == 0 else 0.1)






class _QCanvas(qx.QWidget):
    def __init__(self, host : QxTransformEditor):
        super().__init__()
        self._host = host
        self._qp = qt.QPainter()

        self.set_mouse_tracking(True)

    def _minimum_size_hint(self) -> qt.QSize: return qt.QSize(256,256)

    def _leave_event(self, ev: qt.QEvent):
        super()._leave_event(ev)
        self._host.apply_model( self._host._m.mouse_leave())

    def _resize_event(self, ev: qt.QResizeEvent):
        super()._resize_event(ev)
        self._host.apply_model( self._host._m.set_cli_size(ev.size().width(), ev.size().height()))

    def _mouse_press_event(self, ev: qt.QMouseEvent):
        super()._mouse_press_event(ev)
        if ev.button() == qt.Qt.MouseButton.LeftButton:
            self._host.apply_model( self._host._m.mouse_lbtn_down(qt.QPoint_to_FVec2f(ev.pos()), ctrl_pressed=qt.QApplication.keyboardModifiers() == qt.Qt.KeyboardModifier.ControlModifier, shift_pressed=qt.QApplication.keyboardModifiers() == qt.Qt.KeyboardModifier.ShiftModifier))
        elif ev.button() == qt.Qt.MouseButton.MiddleButton:
            self._host.apply_model( self._host._m.mouse_mbtn_down(qt.QPoint_to_FVec2f(ev.pos()), ctrl_pressed=qt.QApplication.keyboardModifiers() == qt.Qt.KeyboardModifier.ControlModifier, shift_pressed=qt.QApplication.keyboardModifiers() == qt.Qt.KeyboardModifier.ShiftModifier))

    def _mouse_move_event(self, ev: qt.QMouseEvent):
        super()._mouse_move_event(ev)
        self._host.apply_model( self._host._m.mouse_move(qt.QPoint_to_FVec2f(ev.pos()), ctrl_pressed=qt.QApplication.keyboardModifiers() == qt.Qt.KeyboardModifier.ControlModifier, shift_pressed=qt.QApplication.keyboardModifiers() == qt.Qt.KeyboardModifier.ShiftModifier))

    def _mouse_release_event(self, ev: qt.QMouseEvent):
        super()._mouse_release_event(ev)
        if ev.button() == qt.Qt.MouseButton.LeftButton:
            self._host.apply_model( self._host._m.mouse_lbtn_up(qt.QPoint_to_FVec2f(ev.pos()), ctrl_pressed=qt.QApplication.keyboardModifiers() == qt.Qt.KeyboardModifier.ControlModifier, shift_pressed=qt.QApplication.keyboardModifiers() == qt.Qt.KeyboardModifier.ShiftModifier))
        elif ev.button() == qt.Qt.MouseButton.MiddleButton:
            self._host.apply_model( self._host._m.mouse_mbtn_up(qt.QPoint_to_FVec2f(ev.pos()), ctrl_pressed=qt.QApplication.keyboardModifiers() == qt.Qt.KeyboardModifier.ControlModifier, shift_pressed=qt.QApplication.keyboardModifiers() == qt.Qt.KeyboardModifier.ShiftModifier))

    def _wheel_event(self, ev: qt.QWheelEvent):
        super()._wheel_event(ev)
        self._host.apply_model( self._host._m.mouse_wheel(qt.QPointF_to_FVec2f(ev.position()), ev.angleDelta().y(), ctrl_pressed=qt.QApplication.keyboardModifiers() == qt.Qt.KeyboardModifier.ControlModifier, shift_pressed=qt.QApplication.keyboardModifiers() == qt.Qt.KeyboardModifier.ShiftModifier))


    def _paint_event(self, ev : qt.QPaintEvent):
        host = self._host
        model = host.model

        w_stencil_rect = model.w_stencil_rect
        cli_stencil_rect = model.cli_stencil_rect
        cli_img_rect = model.cli_img_rect
        cli_ctrl_circle = model.cli_ctrl_circle
        img_size = model.img_size
        q_w_stencil_rect = qt.QRect(0, 0, w_stencil_rect.width, w_stencil_rect.height)

        select_pen = qt.QPen()
        select_pen.setWidth(2)
        select_pen.setColor(qx.StyleColor.SelectColor)

        select_lighter_pen = qt.QPen()
        select_lighter_pen.setWidth(2)
        select_lighter_pen.setColor(qx.StyleColor.SelectColor.lighter())

        qp = self._qp
        qp.begin(self.q_widget)
        qp.setRenderHint(qt.QPainter.RenderHint.SmoothPixmapTransform|qt.QPainter.RenderHint.Antialiasing)

        qp.setTransform(qt.QTransform_from_mat( FAffMat2.estimate( FRectf(w_stencil_rect.size), cli_stencil_rect)))
        qp.fillRect(q_w_stencil_rect, qt.QColor(0,0,0))

        qp.setClipping(True)
        qp.setClipRect(q_w_stencil_rect)

        qp.setTransform(qt.QTransform_from_mat( FAffMat2.estimate( FRectf(img_size), cli_img_rect)))
        qp.drawPixmap(0,0, host._image_pixmap)
        qp.setClipping(False)

        qp.resetTransform()
        qp.setPen(select_pen)

        if model.is_hovering_ctrl_circle_inside():
            color = qt.QColor(qx.StyleColor.SelectColor)
            color.setAlpha(127)
            brush = qt.QBrush()
            brush.setStyle(qt.Qt.BrushStyle.SolidPattern)
            brush.setColor(color)

            qp.setBrush(brush)
            qp.drawEllipse( qt.QPoint_from_FVec2(cli_ctrl_circle.pos), cli_ctrl_circle.radius, cli_ctrl_circle.radius )


        if model.is_hovering_ctrl_circle_edge():
            qp.setPen(select_lighter_pen)

        qp.setBrush(qt.QBrush())
        qp.drawEllipse( qt.QPoint_from_FVec2(cli_ctrl_circle.pos), cli_ctrl_circle.radius, cli_ctrl_circle.radius, )

        # Draw corners
        qp.end()

