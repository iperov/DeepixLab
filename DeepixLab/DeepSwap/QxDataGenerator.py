import itertools

from common.FSIP import QxFSIPRefList
from core import ax, mx, qt, qx

from .MxDataGenerator import MxDataGenerator


class QxDataGenerator(qx.QVBox):
    def __init__(self, data_gen : MxDataGenerator):
        super().__init__()
        self._data_gen = data_gen
        self._fg = ax.FutureGroup().dispose_with(self)
        self._preview_tg = ax.FutureGroup().dispose_with(self)

        self._q_transform_vbox = qx.QVBox()
        self._q_btn_generate_preview = qx.QPushButton()

        wnd = self._preview_wnd = qx.QWindow().dispose_with(self).set_parent(self).set_title('@(Preview)')
        wnd.set_window_flags(qx.WindowType.Tool | qx.WindowType.WindowStaysOnTopHint | qx.WindowType.MSWindowsFixedSizeDialogHint)
        wnd.mx_close.listen(lambda *_: self._preview_tg.cancel_all() ).dispose_with(self)
        wnd.hide()

        data_gen.mx_error.listen(lambda text: wnd.q_info_bar.add_text(f"<span style='color: red'>@(Error)</span>: {text}") if (wnd := self.get_first_parent_by_class(qx.QMainWIndow)) is not None else ...).dispose_with(self)

        (self
            .add(QxFSIPRefList(data_gen.mx_fsip_ref_list))
            .add(qx.QMxProgress(data_gen.mx_reloading_progress, hide_inactive=True))
            .add(qx.QPushButton().set_text('^ @(Apply_and_reload) ^')
                                 .inline(lambda btn: btn.mx_clicked.listen(lambda: data_gen.apply_and_reload())))
            .add_spacer(4)

            .add( qx.QVBox().set_spacing(1)
                    .add( qx.QHBox()
                        .add( qx.QCheckBoxMxFlag(self._data_gen.mx_dcs) )
                        .add(qx.QLabel().set_text('@(Decrease_chance_similar)') ))

                    .add(qx.QMxProgress(self._data_gen.mx_dcs_progress, hide_inactive=True))
                , align=qx.Align.CenterF )

            .add_spacer(4)

            .add(qx.QGrid().set_spacing(1)
                    .row(0)
                        .add(None)
                        .add(qx.QLabel().set_text('@(Random_augmentations)'), align=qx.Align.CenterF)

                    .next_row()
                        .add(qx.QLabel().set_text('@(Flip)'), align=qx.Align.RightF)
                        .add(qx.QHFrame().add(qx.QCheckBoxMxFlag(data_gen.mx_rnd_flip), align=qx.Align.CenterF))
                    
                    .next_row()
                        .add(qx.QLabel().set_text('Channel exposure'), align=qx.Align.CenterF)
                        
                        .add( qx.QHFrame().set_spacing(1).add(qx.QGrid()
                             
                                .row(0)
                                    .add(qx.QLabel().set_text('R'), align=qx.Align.CenterF)
                                    .add(qx.QLabel().set_text('G'), align=qx.Align.CenterF)
                                    .add(qx.QLabel().set_text('B'), align=qx.Align.CenterF)
                                .next_row()
                                    .add(qx.QDoubleSpinBoxMxNumber(data_gen.mx_rnd_r_exposure_var))
                                    .add(qx.QDoubleSpinBoxMxNumber(data_gen.mx_rnd_g_exposure_var))
                                    .add(qx.QDoubleSpinBoxMxNumber(data_gen.mx_rnd_b_exposure_var))
                                .grid()
                             ))

                    
                    .grid(), align=qx.Align.CenterF)

            .add_spacer(4)
            .add(qx.QLabel().set_text('@(Random_transformations)'), align=qx.Align.CenterF)


            .add(self._q_transform_vbox, align=qx.Align.CenterF)

            .add_spacer(8)
            .add(qx.QVBox()
                    .add(self._q_btn_generate_preview.v_compact().set_checkable(True).set_text('@(Generate_preview)'))
                , align=qx.Align.CenterF)
        )
        
        self._q_transform_vbox.add(vbox := qx.QVBox().dispose_with(self))

        vbox.add(grid := qx.QGrid().set_spacing(1).v_compact(), align=qx.Align.RightF)

        row = grid.row(0)
        row.add(None)
        row.add(qx.QLabel().set_align(qx.Align.CenterF).set_text('@(Offset)'), align=qx.Align.BottomF)
        row.add(qx.QLabel().set_align(qx.Align.CenterF).set_text('@(Random)'), align=qx.Align.BottomF)

        row = row.next_row()
        (row    .add(qx.QLabel().set_text('@(Translation_X)'), align=qx.Align.RightF)
                .add(qx.QDoubleSpinBoxMxNumber(self._data_gen.mx_offset_tx) )
                .add(qx.QDoubleSpinBoxMxNumber(self._data_gen.mx_rnd_tx_var) ))
        row = row.next_row()
        (row    .add(qx.QLabel().set_text('@(Translation_Y)'), align=qx.Align.RightF)
                .add(qx.QDoubleSpinBoxMxNumber(self._data_gen.mx_offset_ty))
                .add(qx.QDoubleSpinBoxMxNumber(self._data_gen.mx_rnd_ty_var)))

        row = row.next_row()
        row.add(qx.QLabel().set_text('@(Scale)'), align=qx.Align.RightF)
        row.add(qx.QDoubleSpinBoxMxNumber(self._data_gen.mx_offset_scale))
        row.add(qx.QDoubleSpinBoxMxNumber(self._data_gen.mx_rnd_scale_var))

        row = row.next_row()
        row.add(qx.QLabel().set_text('@(Rotation)'), align=qx.Align.RightF)
        row.add(qx.QDoubleSpinBoxMxNumber(self._data_gen.mx_offset_rot_deg))
        row.add(qx.QDoubleSpinBoxMxNumber(self._data_gen.mx_rnd_rot_deg_var))

        vbox.add_spacer(8)
        vbox.add( qx.QGrid().set_spacing(1).v_compact()
                    .row(0)
                        .add(qx.QLabel().set_text('@(Transform_intensity)'), align=qx.Align.RightF, col_span=1)
                        .add(qx.QDoubleSpinBoxMxNumber(self._data_gen.mx_transform_intensity))
                    .next_row()
                        .add(qx.QLabel().set_text('@(Image_deform_intensity)'), align=qx.Align.RightF, col_span=1)
                        .add(qx.QDoubleSpinBoxMxNumber(self._data_gen.mx_image_deform_intensity))

                    .grid(), align=qx.Align.RightF )


        mx.CallOnDispose(lambda: self._preview_tg.cancel_all()).dispose_with(self)
        

        self._q_btn_generate_preview.mx_toggled.listen(lambda checked, bag=mx.Disposable().dispose_with(self):
                                                        self._gen_preview_task(bag).call_on_finish(lambda fut: self._q_btn_generate_preview.set_checked(False))
                                                        if checked else self._preview_tg.cancel_all() )


    @ax.task
    def _gen_preview_task(self, bag : mx.Disposable):
        yield ax.attach_to(self._preview_tg, cancel_all=True)

        wnd = self._preview_wnd
        wnd.show()

        W = H = 256

        for i in itertools.count():
            yield ax.wait(t := self._data_gen.generate(1,(H,W,3),(H,W,3),asap=True))

            if i == 0:
                bag.dispose_items()

                image_input_pixmap_widget = qx.QPixmapWidget().h_compact(W).v_compact(H)
                image_target_pixmap_widget = qx.QPixmapWidget().h_compact(W).v_compact(H)
                image_target_guide_pixmap_widget = qx.QPixmapWidget().h_compact(W).v_compact(H)

                wnd.add(qx.QTabWidget().dispose_with(bag).set_tab_position(qx.QTabWidget.TabPosition.North)
                            .add_tab(lambda tab: tab.set_title('@(Input_image)').add(image_input_pixmap_widget))
                            .add_tab(lambda tab: tab.set_title('@(Target_image)').add(image_target_pixmap_widget))
                            .add_tab(lambda tab: tab.set_title('@(Target_guide)').add(image_target_guide_pixmap_widget))
                            )

            if t.succeeded:
                result = t.result

                image_input_pixmap_widget.set_pixmap( qt.QPixmap_from_FImage(result.input_image[0]) )
                image_target_pixmap_widget.set_pixmap( qt.QPixmap_from_FImage(result.target_image[0]) )
                image_target_guide_pixmap_widget.set_pixmap( qt.QPixmap_from_FImage(result.target_guide[0]) )
            else:
                error = t.error
                bag.dispose_items()

                wnd.add(qx.QVBox().dispose_with(bag)
                            .add( qx.QLabel().set_text("<span style='color: red;'>@(Error)</span>").v_compact(), align=qx.Align.CenterF)
                            .add( qx.QTextEdit().h_compact(W).v_compact(H).set_font(qx.FontDB.FixedWidth).set_read_only(True).set_plain_text(str(error)) )
                    )

                yield ax.cancel(error)

            yield ax.sleep(0.1)

