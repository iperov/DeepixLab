from __future__ import annotations

import itertools
from collections import deque
from enum import Enum, auto
from pathlib import Path
from typing import Deque

from core import ax, mx, qt, qx
from core.lib.image import FImage
from core.lib.math import FLine2f, FVec2f

from .FMaskEditor import FMaskEditor


class QxMaskEditorCanvas(qx.QVBox):

    class ViewMode(Enum):
        Image_and_overlay = auto()
        Mask = auto()

    def __init__(self, image : FImage, mask : FImage = None, q_left_panel_vbox : qx.QVBox = None):
        """
            image       HWC
            mask        HW1
        """
        super().__init__()
        self._image = image
        self._mask = mask
        self._q_left_panel_vbox = q_left_panel_vbox
        self._q_cursor_base_image = qt.QImage(str(Path(__file__).parent / 'assets' / 'cursors' / 'cross_base.png'))
        self._q_cursor_overlay_image = qt.QImage(str(Path(__file__).parent / 'assets' / 'cursors' / 'cross_overlay.png'))

        m = FMaskEditor(image.width, image.height)
        if mask is not None:
            m = m.set_mask_image(mask)
        self._m = m
        self._m_undo : Deque[FMaskEditor] = deque([m])
        self._m_redo : Deque[FMaskEditor] = deque()

        self._mx_model = mx.Property[FMaskEditor](m).dispose_with(self)

        self._canvas_initialized = False
        self._mx_settings.reflect(lambda settings, enter, bag=mx.Disposable().dispose_with(self):
                                  self.__ref_settings(settings, enter, bag))

    @property
    def mx_model(self) -> mx.IProperty_rv[FMaskEditor]:
        return self._mx_model

    def __ref_settings(self, settings : qx.QSettings, enter : bool, bag : mx.Disposable):
        if enter:
            self.__settings = settings
            state = settings.state

            holder = qx.QVBox().dispose_with(bag)

            self._mx_show_lines_size = mx.Flag( state.get('show_lines_size', False) ).dispose_with(bag)
            self._mx_show_lines_size.listen(lambda b: state.set('show_lines_size', b))


            shortcut_red_overlay = qx.QShortcut( qt.QKeyCombination(qt.Qt.Key.Key_F1)).set_parent(holder)
            shortcut_red_overlay.mx_press.listen( lambda: (self._set_view_mode(self.ViewMode.Image_and_overlay), self._set_overlay_color(qt.QColor(255,0,0))))
            btn_red_overlay = qx.QPushButton().set_text(f"@(Red) {qx.hfmt.colored_shortcut_keycomb(shortcut_red_overlay)}")
            btn_red_overlay.inline(lambda btn: (btn.mx_pressed.listen(lambda: shortcut_red_overlay.press()), btn.mx_released.listen(lambda: shortcut_red_overlay.release())))

            shortcut_green_overlay = qx.QShortcut( qt.QKeyCombination(qt.Qt.Key.Key_F2)).set_parent(holder)
            shortcut_green_overlay.mx_press.listen( lambda: (self._set_view_mode(self.ViewMode.Image_and_overlay), self._set_overlay_color(qt.QColor(0,255,0))))
            btn_green_overlay = qx.QPushButton().set_text(f"@(Green) {qx.hfmt.colored_shortcut_keycomb(shortcut_green_overlay)}")
            btn_green_overlay.inline(lambda btn: (btn.mx_pressed.listen(lambda: shortcut_green_overlay.press()), btn.mx_released.listen(lambda: shortcut_green_overlay.release())))

            shortcut_blue_overlay = qx.QShortcut( qt.QKeyCombination(qt.Qt.Key.Key_F3)).set_parent(holder)
            shortcut_blue_overlay.mx_press.listen( lambda: (self._set_view_mode(self.ViewMode.Image_and_overlay), self._set_overlay_color(qt.QColor(0,0,255))))
            btn_blue_overlay = qx.QPushButton().set_text(f"@(Blue) {qx.hfmt.colored_shortcut_keycomb(shortcut_blue_overlay)}")
            btn_blue_overlay.inline(lambda btn: (btn.mx_pressed.listen(lambda: shortcut_blue_overlay.press()), btn.mx_released.listen(lambda: shortcut_blue_overlay.release())))

            shortcut_view_mask_mode = qx.QShortcut( qt.QKeyCombination(qt.Qt.Key.Key_F4)).set_parent(holder)
            shortcut_view_mask_mode.mx_press.listen( lambda: self._set_view_mode(self.ViewMode.Mask))
            btn_view_mask_mode = qx.QPushButton().set_text(f"@(BW_mask) {qx.hfmt.colored_shortcut_keycomb(shortcut_view_mask_mode)}")
            btn_view_mask_mode.inline(lambda btn: (btn.mx_pressed.listen(lambda: shortcut_view_mask_mode.press()), btn.mx_released.listen(lambda: shortcut_view_mask_mode.release())))

            shortcut_toggle_show_lines_size = qx.QShortcut( qt.QKeyCombination(qt.Qt.Key.Key_Agrave)).set_parent(holder)
            shortcut_toggle_show_lines_size.mx_press.listen( lambda: (self._mx_show_lines_size.toggle(), self._q_canvas.update())  )

            shortcut_dec_opacity = qx.QShortcut( qt.QKeyCombination(qt.Qt.Key.Key_Q)).set_parent(holder)
            shortcut_dec_opacity.mx_press.listen( lambda: self._set_overlay_opacity(self._overlay_opacity-0.25))
            btn_dec_opacity = qx.QPushButton().set_alignment(qx.Align.CenterF).set_text(f"@(Decrease_opacity)<br>{qx.hfmt.colored_shortcut_keycomb(shortcut_dec_opacity)}")
            btn_dec_opacity.inline(lambda btn: (btn.mx_pressed.listen(lambda: shortcut_dec_opacity.press()), btn.mx_released.listen(lambda: shortcut_dec_opacity.release()),) )

            shortcut_inc_opacity = qx.QShortcut( qt.QKeyCombination(qt.Qt.Key.Key_E)).set_parent(holder)
            shortcut_inc_opacity.mx_press.listen( lambda: self._set_overlay_opacity(self._overlay_opacity+0.25))
            btn_inc_opacity = qx.QPushButton().set_alignment(qx.Align.CenterF).set_text(f"@(Increase_opacity)<br>{qx.hfmt.colored_shortcut_keycomb(shortcut_inc_opacity)}")
            btn_inc_opacity.inline(lambda btn: (btn.mx_pressed.listen(lambda: shortcut_inc_opacity.press()), btn.mx_released.listen(lambda: shortcut_inc_opacity.release()),) )

            shortcut_fill_poly = qx.QShortcut( qt.QKeyCombination(qt.Qt.Key.Key_W)).set_parent(holder)
            shortcut_fill_poly.mx_press.listen(lambda: self.apply_model( self._m.apply_state_poly(FMaskEditor.PolyApplyType.INCLUDE) ))
            btn_fill_poly = qx.QPushButton().set_text(f"@(Fill) {qx.hfmt.colored_shortcut_keycomb(shortcut_fill_poly)}")
            btn_fill_poly.inline(lambda btn: (btn.mx_pressed.listen(lambda: shortcut_fill_poly.press()), btn.mx_released.listen(lambda: shortcut_fill_poly.release())))

            shortcut_cut_poly = qx.QShortcut( qt.QKeyCombination(qt.Qt.Key.Key_S)).set_parent(holder)
            shortcut_cut_poly.mx_press.listen(lambda: self.apply_model( self._m.apply_state_poly(FMaskEditor.PolyApplyType.EXCLUDE) ))
            btn_cut_poly = qx.QPushButton().set_text(f"@(Cut) {qx.hfmt.colored_shortcut_keycomb(shortcut_cut_poly)}")
            btn_cut_poly.inline(lambda btn: (btn.mx_pressed.listen(lambda: shortcut_cut_poly.press()), btn.mx_released.listen(lambda: shortcut_cut_poly.release())))

            shortcut_delete_poly = qx.QShortcut( qt.QKeyCombination(qt.Qt.Key.Key_X)).set_parent(holder)
            shortcut_delete_poly.mx_press.listen(lambda: self.apply_model( self._m.delete_state_poly() ))
            btn_delete_poly = qx.QPushButton().set_text(f"@(Delete) {qx.hfmt.colored_shortcut_keycomb(shortcut_delete_poly)}")
            btn_delete_poly.inline(lambda btn: (btn.mx_pressed.listen(lambda: shortcut_delete_poly.press()), btn.mx_released.listen(lambda: shortcut_delete_poly.release())))

            edit_points_shortcut = qx.QShortcut( qt.QKeyCombination(qt.Qt.Key.Key_Control)).set_parent(holder)# qt.Qt.KeyboardModifier.ControlModifier,
            edit_points_shortcut.mx_press.listen(lambda: ( edit_points_btn_conn.disable(), btn_edit_points.set_checked(True), edit_points_btn_conn.enable(),
                                                          ( self.apply_model(self._m.set_state(state.set_edit_mode(FMaskEditor.FStateEditPoly.EditMode.PT_ADD_DEL)) )
                                                            if isinstance(state := self._m.state, FMaskEditor.FStateEditPoly) else ... ) ))
            edit_points_shortcut.mx_release.listen(lambda: ( edit_points_btn_conn.disable(), btn_edit_points.set_checked(False), edit_points_btn_conn.enable(),
                                                              ( self.apply_model(self._m.set_state(state.set_edit_mode(FMaskEditor.FStateEditPoly.EditMode.PT_MOVE)) )
                                                              if isinstance(state := self._m.state, FMaskEditor.FStateEditPoly) else ... ) ))
            btn_edit_points = qx.QPushButton().set_checkable(True).set_checked(False).set_text(f"@(Add_delete_points) {qx.hfmt.colored_shortcut_keycomb(edit_points_shortcut)}")
            edit_points_btn_conn = btn_edit_points.mx_toggled.listen(lambda toggled: edit_points_shortcut.press() if toggled else edit_points_shortcut.release() )


            perp_move_shortcut = qx.QShortcut( qt.QKeyCombination(qt.Qt.Key.Key_Shift)).set_parent(holder)
            perp_move_shortcut.mx_press.listen(lambda: ( perp_move_btn_conn.disable(), btn_perp_move.set_checked(True), perp_move_btn_conn.enable(),
                                                          ( self.apply_model(self._m.set_state(state.set_edit_mode(FMaskEditor.FStateEditPoly.EditMode.PT_MOVE_PERP)) )
                                                            if isinstance(state := self._m.state, FMaskEditor.FStateEditPoly) else ... ) ))
            perp_move_shortcut.mx_release.listen(lambda: ( perp_move_btn_conn.disable(), btn_perp_move.set_checked(False), perp_move_btn_conn.enable(),
                                                              ( self.apply_model(self._m.set_state(state.set_edit_mode(FMaskEditor.FStateEditPoly.EditMode.PT_MOVE)) )
                                                              if isinstance(state := self._m.state, FMaskEditor.FStateEditPoly) else ... ) ))
            btn_perp_move = qx.QPushButton().set_checkable(True).set_checked(False).set_text(f"@(Perpendicular_constraint) {qx.hfmt.colored_shortcut_keycomb(perp_move_shortcut)}")
            perp_move_btn_conn = btn_perp_move.mx_toggled.listen(lambda toggled: perp_move_shortcut.press() if toggled else perp_move_shortcut.release() )


            shortcut_half_edge = qx.QShortcut( qt.QKeyCombination(qt.Qt.Key.Key_1)).set_parent(holder)
            shortcut_half_edge.mx_press.listen(lambda: self.apply_model( self._m.half_edge() ))
            btn_half_edge = qx.QPushButton().set_text(f"@(Half_edge) {qx.hfmt.colored_shortcut_keycomb(shortcut_half_edge)}")
            btn_half_edge.inline(lambda btn: (btn.mx_pressed.listen(lambda: shortcut_half_edge.press()), btn.mx_released.listen(lambda: shortcut_half_edge.release())))


            shortcut_smooth_corner = qx.QShortcut( qt.QKeyCombination(qt.Qt.Key.Key_2)).set_parent(holder)
            shortcut_smooth_corner.mx_press.listen(lambda: self.apply_model( self._m.smooth_corner() ))
            btn_smooth_corner = qx.QPushButton().set_text(f"@(Smooth_corner) {qx.hfmt.colored_shortcut_keycomb(shortcut_smooth_corner)}")
            btn_smooth_corner.inline(lambda btn: (btn.mx_pressed.listen(lambda: shortcut_smooth_corner.press()), btn.mx_released.listen(lambda: shortcut_smooth_corner.release())))


            center_on_cursor_shortcut = qx.QShortcut( qt.QKeyCombination(qt.Qt.Key.Key_C)).set_parent(holder)
            center_on_cursor_shortcut.mx_press.listen(lambda: self.apply_model( self._m.center_on_cursor()))
            btn_center_on_cursor = qx.QPushButton().set_text(f"@(Center_at_cursor) {qx.hfmt.colored_shortcut_keycomb(center_on_cursor_shortcut)}")
            btn_center_on_cursor.inline(lambda btn: (btn.mx_pressed.listen(lambda: center_on_cursor_shortcut.press()), btn.mx_released.listen(lambda: center_on_cursor_shortcut.release())))


            shortcut_copy_image = qx.QShortcut( qt.QKeyCombination(qt.Qt.KeyboardModifier.ControlModifier|qt.Qt.KeyboardModifier.ShiftModifier, qt.Qt.Key.Key_C)).set_parent(holder)
            shortcut_copy_image.mx_press.listen(lambda: qx.QApplication.instance().q_clipboard.set_image(qt.QImage_from_FImage(self._image.bgr())))
            btn_copy_image = qx.QPushButton().set_text(f"@(Copy_image) {qx.hfmt.colored_shortcut_keycomb(shortcut_copy_image)}")
            btn_copy_image.inline(lambda btn: (btn.mx_pressed.listen(lambda: shortcut_copy_image.press()), btn.mx_released.listen(lambda: shortcut_copy_image.release())))

            shortcut_copy_mask = qx.QShortcut( qt.QKeyCombination(qt.Qt.KeyboardModifier.ControlModifier, qt.Qt.Key.Key_C)).set_parent(holder)
            shortcut_copy_mask.mx_press.listen(lambda: qx.QApplication.instance().q_clipboard.set_image( qt.QImage_from_FImage(self._m.mask_image.ch1()) ))
            btn_copy_mask = qx.QPushButton().set_text(f"@(Copy_mask) {qx.hfmt.colored_shortcut_keycomb(shortcut_copy_mask)}")
            btn_copy_mask.inline(lambda btn: (btn.mx_pressed.listen(lambda: shortcut_copy_mask.press()), btn.mx_released.listen(lambda: shortcut_copy_mask.release())))

            shortcut_paste_mask = qx.QShortcut( qt.QKeyCombination(qt.Qt.KeyboardModifier.ControlModifier, qt.Qt.Key.Key_V)).set_parent(holder)
            shortcut_paste_mask.mx_press.listen(lambda: self.apply_model(self._m.set_mask_image(qt.QImage_to_FImage(image).ch1())) if (image := qx.QApplication.instance().q_clipboard.get_image()) is not None else ... )
            btn_paste_mask = qx.QPushButton().set_text(f"@(Paste_mask) {qx.hfmt.colored_shortcut_keycomb(shortcut_paste_mask)}")
            btn_paste_mask.inline(lambda btn: (btn.mx_pressed.listen(lambda: shortcut_paste_mask.press()), btn.mx_released.listen(lambda: shortcut_paste_mask.release())))


            undo_redo_fg = ax.FutureGroup().dispose_with(self)
            shortcut_undo = qx.QShortcut( qt.QKeyCombination(qt.Qt.KeyboardModifier.ControlModifier, qt.Qt.Key.Key_Z)).set_parent(holder)
            shortcut_undo.mx_press.listen(lambda: self._undo_m(undo_redo_fg))
            shortcut_undo.mx_release.listen(lambda: undo_redo_fg.cancel_all())
            btn_undo = qx.QPushButton().set_text(f"@(Undo) {qx.hfmt.colored_shortcut_keycomb(shortcut_undo)}")
            btn_undo.inline(lambda btn: (btn.mx_pressed.listen(lambda: shortcut_undo.press()), btn.mx_released.listen(lambda: shortcut_undo.release())))

            shortcut_redo = qx.QShortcut( qt.QKeyCombination(qt.Qt.KeyboardModifier.ControlModifier|qt.Qt.KeyboardModifier.ShiftModifier, qt.Qt.Key.Key_Z)).set_parent(holder)
            shortcut_redo.mx_press.listen(lambda: self._redo_m(undo_redo_fg))
            shortcut_redo.mx_release.listen(lambda: undo_redo_fg.cancel_all())
            btn_redo = qx.QPushButton().set_text(f"@(Redo) {qx.hfmt.colored_shortcut_keycomb(shortcut_redo)}")
            btn_redo.inline(lambda btn: (btn.mx_pressed.listen(lambda: shortcut_redo.press()), btn.mx_released.listen(lambda: shortcut_redo.release())))



            self._q_canvas = QxFMaskEditorCanvas(self).dispose_with(bag)

            self._q_left_panel_vbox.add(
                holder .add(qx.QVBox().set_spacing(4)

                        .add(qx.QCollapsibleVBox().set_text('@(Overlay)').inline(lambda collapsible: collapsible.content_vbox
                            .add(qx.QHBox().add(btn_red_overlay).add(btn_green_overlay).add(btn_blue_overlay).add(btn_view_mask_mode))

                            .add(qx.QHBox()
                                    .add(qx.QCheckBoxMxFlag(self._mx_show_lines_size))
                                    .add(qx.QLabel().set_text(f"@(Show_lines_size) {qx.hfmt.colored_shortcut_keycomb(shortcut_toggle_show_lines_size)}"))
                                , align=qx.Align.CenterF)

                            .add_spacer(4)

                            .add(qx.QHBox()
                                    .add(btn_dec_opacity).add(btn_inc_opacity))
                              ))

                        .add(qx.QCollapsibleVBox().set_text('@(Polygon)').inline(lambda collapsible: collapsible.content_vbox
                            .add(qx.QHBox().add(btn_half_edge).add(btn_smooth_corner))
                            .add(qx.QHBox()
                                .add(btn_fill_poly)
                                .add(btn_cut_poly)
                                .add(btn_delete_poly))
                            .add_spacer(4)
                            .add(btn_edit_points)
                            .add(btn_perp_move)


                            ))

                        .add(qx.QCollapsibleVBox().set_text('@(Navigation)').inline(lambda collapsible: collapsible.content_vbox
                                .add(btn_center_on_cursor)))

                        .add(qx.QCollapsibleVBox().set_text('@(Action)').inline(lambda collapsible: collapsible.content_vbox
                                .add(btn_copy_image)
                                .add(btn_copy_mask)
                                .add(btn_paste_mask)
                                #.add(btn_clear_mask)
                                .add_spacer(4)
                                .add(qx.QHBox()
                                    .add(btn_undo).add(btn_redo)
                                )))
                            ))

            self.add(self._q_canvas)

            mx.CallOnDispose(lambda: setattr(self,'_canvas_initialized', False)).dispose_with(bag)
            self._canvas_initialized = True

            self._view_mode = self.ViewMode.Image_and_overlay
            self._overlay_q_color = qt.QColor(0,255,0)
            self._overlay_opacity = 0.25


            #if (view_mode := state.get('view_mode', None)) is not None:
            #    self._set_view_mode(self.ViewMode(view_mode))
            if (overlay_color := state.get('overlay_color', None)) is not None:
                self._set_overlay_color(qt.QColor(*overlay_color))
            if (overlay_opacity := state.get('overlay_opacity', None)) is not None:
                if overlay_opacity == 0:
                    overlay_opacity = 0.25
                self._set_overlay_opacity(overlay_opacity)

            self._update_cursor()
            self._update_overlay_pixmap()
        else:

            bag.dispose_items()

    @property
    def model(self) -> FMaskEditor: return self._m

    @property
    def image(self) -> FImage:
        """original image"""
        return self._image

    def apply_model(self, new_m : FMaskEditor):
        m, self._m = self._m, new_m



        m_state_type = type(m.state)
        new_m_state_type = type(new_m.state)
        m_drag_type = type(m.drag)
        new_m_drag_type = type(new_m.drag)

        changed_mask_image = not (new_m.mask_image is m.mask_image)
        changed_mouse_cli_pt = new_m.mouse_cli_pt != m.mouse_cli_pt

        upd =   (   new_m.cli_size != m.cli_size or
                    new_m.image_size != m.image_size or
                    new_m.view_proj != m.view_proj or
                    new_m_state_type != m_state_type or
                    changed_mask_image or

                    ( isinstance(new_m.state, FMaskEditor.FStatePoly) and
                      isinstance(    m.state, FMaskEditor.FStatePoly) and
                        not (new_m.state.poly is m.state.poly)
                    ) or

                    ( isinstance(new_m.state, FMaskEditor.FStatePoly) and
                      changed_mouse_cli_pt  ) or

                    ( isinstance(new_m.state, FMaskEditor.FStateEditPoly) and
                      new_m.state.edit_mode != m.state.edit_mode  )
                )

        if new_m.is_changed_for_undo(m):
            self._m_undo.append(m)
            self._m_redo = deque()

        if self._canvas_initialized:
            if new_m.is_activated_center_on_cursor(m):
                qt.QCursor.setPos ( self._q_canvas.map_to_global(qt.QPoint_from_FVec2(new_m.view_proj.vp_view_pos)) )

            if new_m_state_type != m_state_type or \
               new_m_drag_type != m_drag_type:
                self._update_cursor()

            if changed_mask_image:
                self._update_overlay_pixmap()

            if upd:
                self._q_canvas.update()

        self._mx_model.set(new_m)

    def _update_overlay_pixmap(self):
        m = self._m
        mask_image = m.mask_image

        if self._view_mode == self.ViewMode.Image_and_overlay:
            bgr = FImage.full_u8(mask_image.width, mask_image.height, self._overlay_q_color.getRgb()[2::-1] )
            mask = FImage.from_bgr_a(bgr, mask_image.ch1().f32().apply(lambda img: img*self._overlay_opacity).u8())
        else:
            mask = mask_image

        self._q_overlay_pixmap = qt.QPixmap_from_np(mask.HWC())


    def _set_view_mode(self, view_mode : ViewMode):
        if self._view_mode != view_mode:
            self._view_mode = view_mode
            self.__settings.state.set('view_mode', view_mode.value)
            self._update_overlay_pixmap()
            self._q_canvas.update()

    def _set_overlay_color(self, color : qt.QColor):
        if self._overlay_q_color != color:
            self._overlay_q_color = color
            self.__settings.state.set('overlay_color', color.getRgb())
            self._update_cursor()
            self._update_overlay_pixmap()
            self._q_canvas.update()

    def _set_overlay_opacity(self, opacity : float):
        opacity = min(max(0.0, opacity), 1.0)
        if self._overlay_opacity != opacity:
            self._overlay_opacity = opacity
            self.__settings.state.set('overlay_opacity', opacity)
            self._update_overlay_pixmap()
            self._q_canvas.update()


    def _update_cursor(self):
        m = self._m

        # if isinstance(m.drag, m.FDragViewProj):
        #     cursor = qt.Qt.CursorShape.ClosedHandCursor
        # else:
        pixmap = qt.QPixmap(self._q_cursor_base_image)
        qp = qt.QPainter(pixmap)
        qp.drawImage(0,0, qt.QImage_colorized(self._q_cursor_overlay_image, self._overlay_q_color))
        qp.end()

        cursor = qt.QCursor(pixmap)

        self._q_canvas.set_cursor(cursor)


    @ax.task
    def _undo_m(self, fg : ax.FutureGroup):
        yield ax.attach_to(fg, cancel_all=True)

        for i in itertools.count():

            if len(self._m_undo) > 1:
                if len(self._m_redo) == 0:
                    self._m_redo.appendleft(self._m)

                self._m_redo.appendleft(self._m_undo.pop())
                self._m = self._m_redo[0].recover_from_undo_redo(self._m)

                self._update_cursor()
                self._update_overlay_pixmap()
                self._q_canvas.update()

                self._mx_model.set(self._m)

            yield ax.sleep(0.4 if i == 0 else 0.1)

    @ax.task
    def _redo_m(self, fg : ax.FutureGroup):
        yield ax.attach_to(fg, cancel_all=True)

        for i in itertools.count():

            if len(self._m_redo) != 0:
                self._m_undo.append(self._m_redo.popleft())
                self._m = self._m_redo[0].recover_from_undo_redo(self._m)

                if len(self._m_redo) == 1:
                    self._m_redo = deque()

                self._update_cursor()
                self._update_overlay_pixmap()
                self._q_canvas.update()

                self._mx_model.set(self._m)

            yield ax.sleep(0.4 if i == 0 else 0.1)


class QxFMaskEditorCanvas(qx.QWidget):
    def __init__(self, host : QxMaskEditorCanvas):
        super().__init__()
        self._host = host
        self._qp = qt.QPainter()
        self._q_image_pixmap = qt.QPixmap_from_np(host._image.bgr().HWC())
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

    def _paint_event(self, ev: qt.QPaintEvent):
        host = self._host
        model = host._m

        mouse_cli_pt = model.mouse_cli_pt
        mouse_world_pt = model.mouse_world_pt
        state = model.state
        overlay_q_color =  host._overlay_q_color
        w2cli_mat = model.w2cli_mat

        qp = self._qp
        qp.begin(self.q_widget)
        qp.setRenderHint(qt.QPainter.RenderHint.SmoothPixmapTransform)

        qp.setTransform(qt.QTransform_from_mat(w2cli_mat))
        qp.drawPixmap(0,0, self._q_image_pixmap)
        qp.drawPixmap(0,0, host._q_overlay_pixmap)

        # Overlays
        qp.resetTransform()
        qp.setRenderHint(qt.QPainter.RenderHint.Antialiasing)

        if isinstance(state, FMaskEditor.FStatePoly):
            poly_line_path = qt.QPainterPath()
            x2_lines = []
            overlay_path = qt.QPainterPath()
            overlay_black_path = qt.QPainterPath()

            overlay_lines_size = []

            state_cli_poly = state.cli_poly
            state_poly = state.poly

            is_draw_poly = isinstance(state, FMaskEditor.FStateDrawPoly)
            is_edit_poly = isinstance(state, FMaskEditor.FStateEditPoly)

            poly_line_path.addPolygon(qt.QPolygon([ qt.QPoint_from_FVec2(pt) for pt in state_cli_poly.points ]))
            if is_draw_poly and mouse_cli_pt is not None:
                # Line from last point to mouse
                poly_line_path.lineTo( qt.QPoint_from_FVec2(mouse_cli_pt) )

            if state_cli_poly.points_count >= 2:
                # Closing poly line
                poly_line_path.lineTo( qt.QPoint_from_FVec2(state_cli_poly.points[0]) )

            poly_pt_id_at_mouse = state.poly_pt_id_at_mouse
            pt_select_rad = model.pt_select_radius
            pt_rad = (pt_select_rad // 2) +1

            mouse_at_poly = state.is_mouse_at_poly()

            is_show_lines_size = host._mx_show_lines_size.get() and (is_draw_poly or mouse_at_poly or isinstance(model.drag, model.FDragStatePolyMovePt))

            if is_show_lines_size:
                points = state_poly.points

                if is_draw_poly and mouse_world_pt is not None:
                    points = points + (mouse_world_pt,)
                if len(points) >= 3:
                    points = points + (points[0],)

                overlay_lines_size.extend( ( FLine2f(p0, p1) for p0, p1 in zip(points[:-1], points[1:]) ) )


            if is_draw_poly:
                if state_cli_poly.points_count >= 3 and poly_pt_id_at_mouse == 0:
                    # Circle around first poly point
                    overlay_path.addEllipse(qt.QPoint_from_FVec2(state_cli_poly.points[0]), pt_rad, pt_rad)
                    overlay_black_path.addEllipse(qt.QPoint_from_FVec2(state_cli_poly.points[0]), pt_rad+1, pt_rad+1)
            elif is_edit_poly:


                if poly_pt_id_at_mouse is None and (edge_id_pt := state.poly_edge_id_pt_at_mouse) is not None:
                    edge_id, pt, cli_pt = edge_id_pt
                    x2_lines.append( w2cli_mat.map(state_poly.edges[edge_id]) )

                for pt_id, cli_pt in enumerate(state_cli_poly.points):
                    cli_qpt = qt.QPoint_from_FVec2(cli_pt)

                    if state.edit_mode == FMaskEditor.FStateEditPoly.EditMode.PT_ADD_DEL:
                        if pt_id == poly_pt_id_at_mouse:
                            poly_line_path.moveTo( qt.QPoint_from_FVec2(cli_pt + FVec2f(-pt_rad,pt_rad)) )
                            poly_line_path.lineTo( qt.QPoint_from_FVec2(cli_pt + FVec2f(pt_rad,-pt_rad)) )
                            poly_line_path.moveTo( qt.QPoint_from_FVec2(cli_pt + FVec2f(-pt_rad,-pt_rad)) )
                            poly_line_path.lineTo( qt.QPoint_from_FVec2(cli_pt + FVec2f(pt_rad,pt_rad)) )

                    if mouse_at_poly or poly_pt_id_at_mouse == pt_id:

                        t = pt_select_rad if poly_pt_id_at_mouse == pt_id else pt_rad

                        overlay_path.addEllipse(cli_qpt, t, t)
                        overlay_black_path.addEllipse(cli_qpt, t+1, t+1)

                if poly_pt_id_at_mouse is None and state.edit_mode == FMaskEditor.FStateEditPoly.EditMode.PT_ADD_DEL:

                    if (edge_id_pt := state.poly_edge_id_pt_at_mouse) is not None:
                        edge_id, pt, cli_pt = edge_id_pt
                        cli_qpt = qt.QPoint_from_FVec2(cli_pt)

                        edge = state_poly.edges[edge_id]

                        if host._mx_show_lines_size.get():
                            overlay_lines_size.append( FLine2f(edge.p0, pt))
                            overlay_lines_size.append( FLine2f(pt, edge.p1))

                        overlay_path.addEllipse(cli_qpt, pt_rad, pt_rad)
                        overlay_black_path.addEllipse(cli_qpt, pt_rad+1, pt_rad+1)
                        overlay_path.moveTo( qt.QPoint_from_FVec2(cli_pt + FVec2f(0,-pt_rad)) )
                        overlay_path.lineTo( qt.QPoint_from_FVec2(cli_pt + FVec2f(0,pt_rad)) )
                        overlay_path.moveTo( qt.QPoint_from_FVec2(cli_pt + FVec2f(-pt_rad,0)) )
                        overlay_path.lineTo( qt.QPoint_from_FVec2(cli_pt + FVec2f(pt_rad,0)) )

            qp.setBrush(qt.QBrush())
            qp.setPen(qt.QPen(overlay_q_color, 2.0))
            qp.drawPath(poly_line_path)

            qp.setPen(qt.QPen(overlay_q_color, 4.0))
            for line in x2_lines:
                qp.drawLine(qt.QLine_from_FLine2f(line))

            qp.setPen(overlay_q_color)
            qp.drawPath(overlay_path)
            qp.setPen(qt.QColor(0,0,0))
            qp.drawPath(overlay_black_path)

            font = qx.QFontDB.instance().fixed_width()
            qp.setFont(font)
            qp.setPen(qt.QColor(qx.StyleColor.Text))

            fm = qt.QFontMetrics(font, qp.device())

            for world_line in overlay_lines_size:
                text = f'{world_line.length:.0f}'
                rect = fm.boundingRect(qt.QRect(), 0, text)

                w = rect.size().width()
                h = rect.size().height()
                cli_line_pm = w2cli_mat.map(world_line.pm)

                rect = qt.QRect(cli_line_pm.x - w//2, cli_line_pm.y - h//2, w, h)

                qp.fillRect(rect, qt.QColor(0,0,0,150))
                qp.drawText(rect, 0, text)


        qp.end()










