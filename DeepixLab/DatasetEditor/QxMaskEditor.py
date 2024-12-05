from __future__ import annotations

import itertools

from common.FSIP import MxPairType, QxPairType
from core import ax, mx, qt, qx
from core.lib.dataset.FSIP import IFSIP_v
from core.lib.image import FImage
from core.lib.math import FVec2f

from .FMaskEditor import FMaskEditor
from .QxMaskEditorCanvas import QxMaskEditorCanvas


class QxMaskEditor(qx.QVBox):
    def __init__(self,  fsip : IFSIP_v,
                        item_id : int,
                        pair_type : str|None,
                        ):
        super().__init__()
        self._fsip = fsip

        self._thread_pool = ax.ThreadPool().dispose_with(self)
        self._mx_quit_ev = mx.Event0().dispose_with(self)

        self._mx_error = mx.TextEmitter().dispose_with(self)
        qx.QPopupMxTextEmitter(self._mx_error, title='@(Error)').set_parent(self)

        self._q_left_panel_vbox = qx.QVBox()
        self._q_central_panel_vbox = qx.QVBox()

        self.add( qx.QSplitter()
                    .add( qx.QVScrollArea().h_compact().set_widget(
                            self._q_left_panel_vbox.set_spacing(4).v_compact()))
                    .add(self._q_central_panel_vbox) )

        self._mx_settings.reflect(lambda settings, enter, bag=mx.Disposable().dispose_with(self):
                                   self.__ref_settings(settings, enter, bag, item_id, pair_type))

    @property
    def mx_quit_ev(self) -> mx.IEvent0_rv:
        """event called when QxMaskEditor done working, so object can be disposed"""
        return self._mx_quit_ev
    @property
    def current_item_id(self) -> int: return self._q_tape.mx_current_item_id.get()

    def __ref_settings(self, settings : qx.QSettings, enter : bool, bag : mx.Disposable,
                             item_id : int, pair_type : str|None,):
        if not enter:
            bag.dispose_items()
        else:
            fsip = self._fsip
            state = settings.state

            self._canvas_initialized = False
            self._tape_fg = ax.FutureGroup().dispose_with(bag)

            self._mx_keep_view = mx.Flag( state.get('keep_view', False) ).dispose_with(bag)
            self._mx_keep_view.listen(lambda b: state.set('keep_view', b))


            self._mx_pair_type = MxPairType(fsip.root).dispose_with(bag)
            if pair_type is not None:
                self._mx_pair_type.set(pair_type)
            self._mx_pair_type.listen(lambda pair_type, enter: (self._f_rebuild_canvas(),
                                                                self._q_tape.update_items(),) if enter else self._save() )

            self._q_canvas_holder = qx.QVBox()

            q_tape = self._q_tape = qx.QCachedTapeItemView(fut_get_item_pixmap=lambda idx, size: self._tape_get_item_pixmap(idx, size))
            q_tape.apply_model(q_tape.model.set_item_count(fsip.item_count).set_item_size(64, 64+1+16).set_current_item_id(item_id) )
            q_tape.mx_current_item_id.listen(lambda item_id: (self._save(),self._f_rebuild_canvas())  )


            # Shortcuts
            shortcut_exit = qx.QShortcut(qt.QKeyCombination(qt.Qt.KeyboardModifier.ControlModifier, qt.Qt.Key.Key_Q)).set_parent(self._q_canvas_holder).inline(lambda shortcut:
                                    shortcut.mx_press.listen(lambda: self._mx_quit_ev.emit()))

            shortcut_save = qx.QShortcut(qt.QKeyCombination(qt.Qt.KeyboardModifier.ControlModifier, qt.Qt.Key.Key_S)).set_parent(self._q_canvas_holder).inline(lambda shortcut:
                                    shortcut.mx_press.listen(lambda: self._save(force=True)))

            shortcut_prev_pair_type = qx.QShortcut(qt.QKeyCombination(qt.Qt.KeyboardModifier.ControlModifier, qt.Qt.Key.Key_Tab)).set_parent(self._q_canvas_holder).inline(lambda shortcut: shortcut.mx_press.listen(lambda: self._mx_pair_type.set_prev()))
            shortcut_next_pair_type = qx.QShortcut(qt.QKeyCombination(qt.Qt.Key.Key_Tab)).set_parent(self._q_canvas_holder).inline(lambda shortcut: shortcut.mx_press.listen(lambda: self._mx_pair_type.set_next()))

            shortcut_prev = qx.QShortcut(qt.QKeyCombination(qt.Qt.Key.Key_A)).set_parent(self._q_canvas_holder).inline(lambda shortcut: (
                                    shortcut.mx_press.listen(lambda: self._save_and_next(-1)),
                                    shortcut.mx_release.listen(lambda: self._tape_fg.cancel_all())))

            shortcut_next = qx.QShortcut(qt.QKeyCombination(qt.Qt.Key.Key_D)).set_parent(self._q_canvas_holder).inline(lambda shortcut: (
                                    shortcut.mx_press.listen(lambda: self._save_and_next(1)),
                                    shortcut.mx_release.listen(lambda: self._tape_fg.cancel_all())))

            shortcut_prev_mask = qx.QShortcut(qt.QKeyCombination(qt.Qt.KeyboardModifier.ControlModifier, qt.Qt.Key.Key_A)).set_parent(self._q_canvas_holder).inline(lambda shortcut: (
                                    shortcut.mx_press.listen(lambda: self._save_and_next_mask(forward=False)),
                                    shortcut.mx_release.listen(lambda: self._tape_fg.cancel_all())))

            shortcut_next_mask = qx.QShortcut(qt.QKeyCombination(qt.Qt.KeyboardModifier.ControlModifier, qt.Qt.Key.Key_D)).set_parent(self._q_canvas_holder).inline(lambda shortcut: (
                                    shortcut.mx_press.listen(lambda: self._save_and_next_mask(forward=True)),
                                    shortcut.mx_release.listen(lambda: self._tape_fg.cancel_all())))

            self._unsaved_icon = qx.QIconWidget().set_icon(qx.IconDB.save_outline, qx.StyleColor.Text)

            # Fill left panel
            self._q_left_panel_vbox.add(
                qx.QVBox().set_spacing(4).dispose_with(bag)

                    .add(qx.QCollapsibleVBox().set_text('@(File)').inline(lambda collapsible: collapsible.content_vbox
                        .add(qx.QPushButton().set_text(f"@(Quit) {qx.hfmt.colored_shortcut_keycomb(shortcut_exit)}")
                                    .inline(lambda btn: (btn.mx_pressed.listen(lambda: shortcut_exit.press()), btn.mx_released.listen(lambda: shortcut_exit.release()))))

                        .add_spacer(4)

                        .add(qx.QHBox()
                            .add(qx.QPushButton().set_text(f"@(Save) {qx.hfmt.colored_shortcut_keycomb(shortcut_save)}")
                                            .inline(lambda btn: (btn.mx_pressed.listen(lambda: shortcut_save.press()), btn.mx_released.listen(lambda: shortcut_save.release()))))
                            .add(self._unsaved_icon.h_compact().hide()))

                        .add( qx.QHBox()
                                .add(qx.QPushButton().set_text(f"@(and_previous) {qx.hfmt.colored_shortcut_keycomb(shortcut_prev)}")
                                    .inline(lambda btn: (btn.mx_pressed.listen(lambda: shortcut_prev.press()), btn.mx_released.listen(lambda: shortcut_prev.release()))))
                                .add(qx.QPushButton().set_text(f"@(and_next) {qx.hfmt.colored_shortcut_keycomb(shortcut_next)}")
                                    .inline(lambda btn: (btn.mx_pressed.listen(lambda: shortcut_next.press()), btn.mx_released.listen(lambda: shortcut_next.release())))) )

                        .add( qx.QHBox()
                                .add(qx.QPushButton().set_text(f"@(with_mask) {qx.hfmt.colored_shortcut_keycomb(shortcut_prev_mask)}")
                                    .inline(lambda btn: (btn.mx_pressed.listen(lambda: shortcut_prev_mask.press()), btn.mx_released.listen(lambda: shortcut_prev_mask.release()))))
                                .add(qx.QPushButton().set_text(f"@(with_mask) {qx.hfmt.colored_shortcut_keycomb(shortcut_next_mask)}")
                                    .inline(lambda btn: (btn.mx_pressed.listen(lambda: shortcut_next_mask.press()), btn.mx_released.listen(lambda: shortcut_next_mask.release())))) )
                        .add_spacer(4)

                        .add(qx.QCheckBoxMxFlag(self._mx_keep_view).set_text('@(Keep_view)'), align=qx.Align.CenterF)

                        .add_spacer(4)

                        .add(qx.QPushButton().set_text(f"@(Delete_mask_and_file)")
                                            .inline(lambda btn: btn.mx_pressed.listen(lambda: self._delete_mask_and_file())))
                    ))

                    .add(qx.QCollapsibleVBox().set_text('@(Mask)').inline(lambda collapsible: collapsible.content_vbox
                        .add(QxPairType(self._mx_pair_type))

                        .add( qx.QHBox()
                            .add(qx.QPushButton().set_text(f"@(Previous) {qx.hfmt.colored_shortcut_keycomb(shortcut_prev_pair_type)}")
                                                                    .inline(lambda btn: btn.mx_clicked.listen(lambda: shortcut_prev_pair_type.press())))
                            .add(qx.QPushButton().set_text(f"@(Next) {qx.hfmt.colored_shortcut_keycomb(shortcut_next_pair_type)}")
                                            .inline(lambda btn: btn.mx_clicked.listen(lambda: shortcut_next_pair_type.press()))))

                        ))
                )


            # Fill central
            self._q_central_panel_vbox.add(
                qx.QSplitter().dispose_with(bag)
                    .set_orientation(qx.Orientation.Vertical).set_default_sizes([9999,1])
                    .add(qx.QVBox()
                            .add(self._q_canvas_holder))
                    .add(qx.QVBox()
                            .add(self._q_tape)))

            self._f_rebuild_canvas = lambda bag=mx.Disposable().dispose_with(bag): self._rebuild_canvas(bag=bag)
            self._f_rebuild_canvas()



    def _rebuild_canvas(self, bag : mx.Disposable):
        bag.dispose_items()
        mx.CallOnDispose(lambda: setattr(self, '_canvas_initialized', False)).dispose_with(bag)

        fsip = self._fsip
        if (item_id := self._q_tape.mx_current_item_id.get()) is not None:
            if (pair_type := self._mx_pair_type.get()) not in [None, MxPairType.NO_PAIR]:

                err = None
                try:
                    item = FImage.from_file(fsip.get_item_path(item_id))
                    pair = FImage.from_file(fsip.get_pair_path(item_id, pair_type)) \
                            if fsip.has_pair(item_id, pair_type) else None
                except Exception as e:
                    err = e

                if err is None:
                    self._q_me_canvas = q_me_canvas = QxMaskEditorCanvas(item, mask=pair, q_left_panel_vbox=self._q_left_panel_vbox).dispose_with(bag)
                    self._q_me_canvas_mx_mask_image = mx.Property[FImage](q_me_canvas.model.mask_image).dispose_with(bag)
                    mx.CallOnDispose(lambda: setattr(self, '_last_model', q_me_canvas.model)).dispose_with(bag)
                    last_model : FMaskEditor = getattr(self, '_last_model', None)

                    if self._mx_keep_view.get() and last_model is not None:
                        model = q_me_canvas.model

                        image_aspect = FVec2f(model.image_size) / last_model.image_size
                        w_view_pos = last_model.view_proj.w_view_pos * image_aspect
                        view_scale = last_model.view_proj.scale * image_aspect

                        model = model.set_view_proj(model.view_proj.set_w_view_pos(w_view_pos)
                                                                   .set_scale(view_scale))
                        q_me_canvas.apply_model(model)

                    mx.CallOnDispose(lambda: self._unsaved_icon.set_visible(False)).dispose_with(bag)
                    self._q_me_canvas_mx_mask_image.listen(upd_func := lambda *_: self._unsaved_icon.set_visible(q_me_canvas.model.mask_image != self._q_me_canvas_mx_mask_image.get()) )
                    q_me_canvas.mx_model.reflect(upd_func).dispose_with(bag)

                    self._canvas_initialized = True

                    self._q_canvas_holder.add(self._q_me_canvas)
                else:
                    self._mx_error.emit(str(err))
                    self._q_canvas_holder.add(qx.QLabel().dispose_with(bag).set_font(qx.FontDB.FixedWidth).set_align(qx.Align.CenterF).set_text(f'@(Error)'))
            else:
                self._q_canvas_holder.add(qx.QLabel().dispose_with(bag).set_align(qx.Align.CenterF).set_text('@(No_mask_selected)'))
        else:
            self._q_canvas_holder.add(qx.QLabel().dispose_with(bag).set_align(qx.Align.CenterF).set_text('@(No_image_selected)'))

    def _delete_mask_and_file(self):
        if (item_id := self._q_tape.mx_current_item_id.get()) is not None:
            if (pair_type := self._mx_pair_type.get()) not in [None, MxPairType.NO_PAIR]:
                try:
                    self._fsip.delete_pair(item_id, pair_type)
                except Exception as e:
                    self._mx_error.emit(str(e))
                    return False

                self._q_tape.update_item(item_id)
                self._f_rebuild_canvas()

    def _save(self, force = False):
        """"""
        if self._canvas_initialized:
            q_me_canvas = self._q_me_canvas
            model = q_me_canvas.model
            mask_image = model.mask_image

            if (item_id := self._q_tape.mx_current_item_id.get()) is not None and \
               (pair_type := self._mx_pair_type.get()) not in [None, MxPairType.NO_PAIR]:

                    if self._q_me_canvas_mx_mask_image.get() != mask_image or force:
                        # Save if mask changed or force
                        try:
                            fsip = self._fsip
                            if (pair_path := fsip.get_pair_path(item_id, pair_type)) is None:
                                pair_path = fsip.add_pair_path(item_id, pair_type, '.png')

                            mask_image.save(pair_path)
                        except Exception as e:
                            self._mx_error.emit(str(e))
                            return False

                        self._q_me_canvas_mx_mask_image.set(mask_image)
                        self._q_tape.update_item(item_id)
        return True

    def _save_and_goto(self, item_id : int) -> bool:
        if (r := self._save()):
            self._q_tape.mx_current_item_id.set(item_id)
        return r


    @ax.task
    def _save_and_next(self, diff : int):
        yield ax.attach_to(self._tape_fg, cancel_all=True)

        for i in itertools.count():
            if (current_item_id := self._q_tape.mx_current_item_id.get()) is not None:
                if not (self._save_and_goto(current_item_id+diff)):
                    yield ax.cancel()

            if i == 0:
                yield ax.sleep(0.5)
            else:
                yield ax.sleep(0.05)

    @ax.task
    def _save_and_next_mask(self, forward : bool):
        yield ax.attach_to(self._tape_fg, cancel_all=True)

        fsip = self._fsip

        for i in itertools.count():

            if (item_id := self._q_tape.mx_current_item_id.get()) is not None and \
               (pair_type := self._mx_pair_type.get()) not in [None, MxPairType.NO_PAIR]:

                for next_item_id in range(item_id+1, fsip.item_count) if forward else \
                                    range(item_id-1, -1,-1):
                    if fsip.has_pair(next_item_id, pair_type):
                        self._save_and_goto(next_item_id)
                        break

            if i == 0:
                yield ax.sleep(0.5)
            else:
                yield ax.sleep(0.05)


    @ax.task
    def _tape_get_item_pixmap(self, item_id : int, size : qt.QSize) -> qt.QPixmap:
        yield ax.switch_to(self._thread_pool)

        fsip = self._fsip

        w, h = size.width(),size.height()

        caption_bg_color = qx.StyleColor.Midlight

        item_path = fsip.get_item_path(item_id)
        try:

            pixmap = qt.QPixmap_from_FImage(FImage.from_file(item_path))

            if (pair_type := self._mx_pair_type.get()) not in [None, MxPairType.NO_PAIR] and \
                fsip.has_pair(item_id, pair_type):
                caption_bg_color = qt.QColor(0,50,100)

        except Exception as e:
            pixmap = qx.QIconDB.instance().pixmap(qx.IconDB.alert_circle_outline, qt.QColor(255,0,0))

        caption = item_path.name

        out_pixmap = qt.QPixmap(w, h)
        out_pixmap.fill(qt.QColor(0,0,0,0))
        qp = qt.QPainter(out_pixmap)


        image_rect = qt.QRect(0, 0, w, w)
        cap_rect = qt.QRect(0, h-16, w, 16)
        qp.fillRect(image_rect, qx.StyleColor.Midlight)
        qp.fillRect(cap_rect, caption_bg_color)

        fitted_image_rect = qt.QRect_fit_in(pixmap.rect(), image_rect)
        qp.drawPixmap(fitted_image_rect, pixmap)

        font = qx.QFontDB.instance().fixed_width()
        fm = qt.QFontMetrics(font)
        qp.setFont(font)
        qp.setPen(qx.StyleColor.Text)
        caption_text = fm.elidedText(caption, qt.Qt.TextElideMode.ElideLeft, cap_rect.width())
        qp.drawText(cap_rect, qt.Qt.AlignmentFlag.AlignCenter, caption_text)


        qp.end()

        return out_pixmap

