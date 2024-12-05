from __future__ import annotations

from common.ImageFormat import QxImageFormat
from common.MediaSource import (MxMediaSource, QxMediaSourceCentral,
                                QxMediaSourceHead)
from core import ax, mx, qt, qx

from .MxFacesetMaker import MxFacesetMaker


class QxFacesetMaker(qx.QVBox):
    def __init__(self, fm : MxFacesetMaker, media_source_head_holder : qx.QBox=None):
        super().__init__()
        self._fm = fm
        self._media_source_head_holder = media_source_head_holder

        self._q_central_panel_vbox = qx.QVBox()
        self.add(self._q_central_panel_vbox)

        self._mx_settings.reflect(lambda settings, enter, bag=mx.Disposable().dispose_with(self):
                                    self.__ref_settings(settings, enter, bag))

    def __ref_settings(self, settings : qx.QSettings, enter : bool, bag : mx.Disposable):
        if enter:
            self.__settings = settings

            if (wnd := self.get_first_parent_by_class(qx.QMainWIndow)) is not None:
                self._f_post_info = lambda info: wnd.q_info_bar.add_text(info)
            else:
                self._f_post_info = lambda info: ...

            self._fm.mx_error.listen(lambda text: self._f_post_info(f"<span style='color: red'>@(Error)</span>: {text}")).dispose_with(bag)
            self._fm.mx_info.listen(lambda text: self._f_post_info(text)).dispose_with(bag)

            self._media_source_head_holder.add( QxMediaSourceHead(self._fm.mx_media_source).dispose_with(bag) )

            self._fm.mx_media_source.mx_source_type.reflect(lambda source_type, enter, bag=mx.Disposable().dispose_with(bag):
                                                            self._ref_ms_source_type(source_type, enter, bag=bag)).dispose_with(bag)

        else:
            bag.dispose_items()



    def _ref_ms_source_type(self, source_type : MxMediaSource.SourceType, enter : bool, bag : mx.Disposable):
        if enter:
            self._fm.mx_media_source.mx_media_path.mx_opened.reflect(lambda opened, enter, bag=mx.Disposable().dispose_with(bag):
                                                                     self._ref_media_path(opened, enter, bag)).dispose_with(bag)
        else:
            bag.dispose_items()


    def _ref_media_path(self, opened : bool, enter : bool, bag : mx.Disposable):
        if enter:
            if opened:
                fm = self._fm

                source_type = fm.mx_media_source.mx_source_type.get()

                self._item_view = qx.QCachedGridItemView(self._on_get_item_pixmap)

                wipe_dir_dlg = self._wipe_dir_dlg = qx.QInplaceDialog()
                wipe_dir_dlg.q_label_title.set_text('@(Are_you_sure)')
                wipe_dir_dlg.q_btn_ok.mx_clicked.listen(lambda: self._fm.delete_export_directories())

                q_label_active_jobs = qx.QLabel()
                q_label_jobs_done_per_sec = qx.QLabel()

                fm.mx_jobs_count.reflect(lambda count: q_label_active_jobs.set_text(f'{count}') ).dispose_with(bag)
                fm.mx_jobs_done_per_sec.reflect(lambda count: q_label_jobs_done_per_sec.set_text(f'{count:.1f}') ).dispose_with(bag)

                self._q_central_panel_vbox.add(
                    qx.QVBox().dispose_with(bag)
                        .add(qx.QHBox()
                                .add(qx.QVBox().h_compact()
                                    .add(qx.QHBox()
                                        .add(qx.QVScrollArea().set_widget(
                                            qx.QVBox().v_compact()
                                                .add(qx.QCollapsibleVBox().open().set_text('@(Face_detector)').inline(lambda collapsible: collapsible.content_vbox
                                                    .add(qx.QGrid().set_spacing(1)
                                                        .row(0)
                                                            .add(qx.QGrid().set_spacing(1)
                                                                    .row(0)
                                                                        .add(qx.QLabel().set_text('@(Type)'), align=qx.Align.RightF)
                                                                        .add(qx.QComboBoxMxStateChoice(self._fm.mx_face_detector_type).set_font(qx.FontDB.FixedWidth))
                                                                    .next_row()
                                                                        .add(qx.QLabel().set_text('@(Device)'), align=qx.Align.RightF)
                                                                        .add(qx.QComboBoxMxStateChoice(self._fm.mx_face_detector_device).set_font(qx.FontDB.FixedWidth).h_compact(200))
                                                                    .grid(), col_span=2)
                                                        .next_row()
                                                            .add(qx.QLabel().set_text('@(Resolution)'), align=qx.Align.RightF)
                                                            .add(qx.QComboBoxMxStateChoice(self._fm.mx_detector_resolution), align=qx.Align.LeftF)
                                                        .next_row()
                                                            .add(qx.QLabel().set_text('@(Augment_pyramid)'), align=qx.Align.RightF)
                                                            .add(qx.QCheckBoxMxFlag(self._fm.mx_augment_pyramid), align=qx.Align.LeftF)
                                                        .next_row()
                                                            .add(qx.QLabel().set_text('@(Minimum_confidence)'), align=qx.Align.RightF)
                                                            .add(qx.QDoubleSpinBoxMxNumber(self._fm.mx_detector_minimum_confidence), align=qx.Align.LeftF)
                                                        .next_row()
                                                            .add(qx.QLabel().set_text('@(Overlap_threshold)'), align=qx.Align.RightF)
                                                            .add(qx.QDoubleSpinBoxMxNumber(self._fm.mx_detector_overlap_threshold), align=qx.Align.LeftF)
                                                        .next_row()
                                                            .add(qx.QLabel().set_text('@(Min_face_size)'), align=qx.Align.RightF)
                                                            .add(qx.QDoubleSpinBoxMxNumber(self._fm.mx_min_face_size), align=qx.Align.LeftF)

                                                        .grid())))

                                                .add(qx.QCollapsibleVBox().open().set_text('@(Face_marker)').inline(lambda collapsible: collapsible.content_vbox
                                                    .add(qx.QGrid().set_spacing(1)
                                                        .row(0)
                                                            .add(qx.QLabel().set_text('@(Type)'), align=qx.Align.RightF)
                                                            .add(qx.QComboBoxMxStateChoice(self._fm.mx_face_marker_type).set_font(qx.FontDB.FixedWidth))
                                                        .next_row()
                                                            .add(qx.QLabel().set_text('@(Device)'), align=qx.Align.RightF)
                                                            .add(qx.QComboBoxMxStateChoice(self._fm.mx_face_marker_device).set_font(qx.FontDB.FixedWidth).h_compact(200))
                                                        .next_row()
                                                            .add(qx.QLabel().set_text('@(Pass_count)'), align=qx.Align.RightF)
                                                            .add(qx.QDoubleSpinBoxMxNumber(self._fm.mx_pass_count), align=qx.Align.LeftF)
                                                        .grid())))

                                                .add(qx.QCollapsibleVBox().open().set_text('@(Face_identifier)').inline(lambda collapsible: collapsible.content_vbox
                                                    .add(qx.QGrid().set_spacing(1)
                                                        .row(0)
                                                            .add(qx.QLabel().set_text('@(Type)'), align=qx.Align.RightF)
                                                            .add(qx.QComboBoxMxStateChoice(self._fm.mx_face_identifier_type).set_font(qx.FontDB.FixedWidth))
                                                        .next_row()
                                                            .add(qx.QLabel().set_text('@(Device)'), align=qx.Align.RightF)
                                                            .add(qx.QComboBoxMxStateChoice(self._fm.mx_face_identifier_device).set_font(qx.FontDB.FixedWidth).h_compact(200))
                                                        .grid())))

                                                .add(qx.QCollapsibleVBox().open().set_text('@(Face_aligner)').inline(lambda collapsible: collapsible.content_vbox
                                                    .add(qx.QGrid().set_spacing(1)
                                                        .row(0)
                                                            .add(qx.QLabel().set_text('@(Face_coverage)'), align=qx.Align.RightF)
                                                            .add(qx.QDoubleSpinBoxMxNumber(self._fm.mx_face_coverage), align=qx.Align.LeftF)
                                                        .next_row()
                                                            .add(qx.QLabel().set_text('@(Face_Y_offset)'), align=qx.Align.RightF)
                                                            .add(qx.QDoubleSpinBoxMxNumber(self._fm.mx_face_y_offset), align=qx.Align.LeftF)
                                                        .next_row()
                                                            .add(qx.QLabel().set_text('@(Face_Y_axis_offset)'), align=qx.Align.RightF)
                                                            .add(qx.QDoubleSpinBoxMxNumber(self._fm.mx_face_y_axis_offset), align=qx.Align.LeftF)
                                                        .next_row()
                                                            .add(qx.QLabel().set_text('@(Image_size)'), align=qx.Align.RightF)
                                                            .add(qx.QHBox()
                                                                    .add(qx.QDoubleSpinBoxMxNumber(self._fm.mx_min_image_size))
                                                                    .add(qx.QLabel().set_text('-'))
                                                                    .add(qx.QDoubleSpinBoxMxNumber(self._fm.mx_max_image_size))

                                                                    , align=qx.Align.LeftF)
                                                        .next_row()
                                                            .add(qx.QLabel().set_text('@(Border_type)'), align=qx.Align.RightF, col_span=1)
                                                            .add(qx.QComboBoxMxStateChoice(self._fm.mx_border_type), align=qx.Align.LeftF)
                                                        .grid())))


                                                .add(qx.QCollapsibleVBox().open().set_text('@(Face_list)').inline(lambda collapsible: collapsible.content_vbox
                                                    .add(qx.QGrid().set_spacing(1)
                                                        .row(0)
                                                            .add(qx.QLabel().set_text('@(Sort_by)'), align=qx.Align.RightF)
                                                            .add(qx.QComboBoxMxStateChoice(self._fm.mx_sort_by_type), align=qx.Align.LeftF)
                                                        .next_row()
                                                            .add(qx.QLabel().set_text('@(Max_faces)'), align=qx.Align.RightF)
                                                            .add(qx.QDoubleSpinBoxMxNumber(self._fm.mx_max_faces), align=qx.Align.LeftF)
                                                        .next_row()
                                                            .add(None)
                                                            .add( qx.QHBox()
                                                                    .add(qx.QCheckBoxMxFlag(self._fm.mx_max_faces_discard))
                                                                    .add(qx.QLabel().set_text('@(Discard_if_more)'))

                                                                 , align=qx.Align.LeftF)
                                                        .grid())))

                                                .add(qx.QCollapsibleVBox().open().set_text('@(Preview)').inline(lambda collapsible: collapsible.content_vbox
                                                    .add(qx.QGrid().set_spacing(1)
                                                        .row(0)
                                                            .add(qx.QLabel().set_text('@(Image_size)'), align=qx.Align.RightF)
                                                            .add(qx.QDoubleSpinBoxMxNumber(self._fm.mx_preview_image_size), align=qx.Align.LeftF)
                                                        .next_row()
                                                            .add(qx.QLabel().set_text('@(Draw_annotations)'), align=qx.Align.RightF)
                                                            .add(qx.QCheckBoxMxFlag(self._fm.mx_preview_draw_annotations), align=qx.Align.LeftF)
                                                        .grid())))

                                                .add(qx.QCollapsibleVBox().open().set_text('@(Jobs)').inline(lambda collapsible: collapsible.content_vbox
                                                    .add(qx.QGrid().set_spacing(1)
                                                        .row(0)
                                                            .add(qx.QLabel().set_text('@(Active)'), align=qx.Align.CenterF)
                                                            .add(qx.QLabel().set_text('@(Maximum)'), align=qx.Align.CenterF)
                                                            .add(qx.QLabel().set_text('@(per_second)'), align=qx.Align.CenterF)

                                                        .next_row()
                                                            .add(q_label_active_jobs.set_font(qx.FontDB.Digital), align=qx.Align.CenterF)
                                                            .add(qx.QDoubleSpinBoxMxNumber(self._fm.mx_jobs_max), align=qx.Align.CenterF)
                                                            .add(q_label_jobs_done_per_sec.set_font(qx.FontDB.Digital), align=qx.Align.CenterF)


                                                        .grid(), align=qx.Align.CenterF )))

                                                .add(qx.QCollapsibleVBox().open().set_text('@(Export)').inline(lambda collapsible: collapsible.content_vbox
                                                    .add(qx.QVBox()
                                                            .add(qx.QMxPathState(self._fm.mx_export_path).h_expand())
                                                            .add(qx.QHBox()
                                                                    .add(qx.QLabel().set_text('@(File_format)'), align=qx.Align.RightF)
                                                                    .add(QxImageFormat(self._fm.mx_export_file_format)) )

                                                            .add(qx.QCheckBoxMxFlag(self._fm.mx_export_dfl_mask).set_text('@(Export_DFL_mask)'), align=qx.Align.CenterF)

                                                            .add(qx.QPushButton().set_text('@(Delete_export_directories)')
                                                                        .inline(lambda btn: btn.mx_clicked.listen(lambda: wipe_dir_dlg.popup())) )
                                                            .add(wipe_dir_dlg)
                                                            .add(qx.QMxProgress(self._fm.mx_ded_progress, hide_inactive=True))

                                                            .add( qx.QOnOffPushButtonMxFlag(self._fm.mx_export_enabled)
                                                                    .inline(lambda btn: btn.on_button.set_text('@(Disable_export)')  )
                                                                    .inline(lambda btn: btn.off_button.set_text('@(Enable_export)')  ))


                                                    ))))))
                                        )

                                .add_spacer(8)
                                .add(qx.QSplitter().set_orientation(qx.Orientation.Vertical)
                                    .add(QxMediaSourceCentral(fm.mx_media_source))
                                    .add( qx.QVBox()
                                            .add(self._item_view.h_expand()) ))

                                    ))


                fm.mx_preview_image_size.reflect(lambda preview_image_size: self._item_view.apply_model( self._item_view.model.set_item_size(preview_image_size, preview_image_size) )).dispose_with(bag)
                fm.mx_preview_draw_annotations.listen(lambda _: self._item_view.update_items()).dispose_with(bag)
                fm.mx_preview_frame.reflect(lambda frame: self._ref_preview_frame(frame)).dispose_with(bag)
            else:
                self._q_central_panel_vbox.add(
                    qx.QVBox().dispose_with(bag)
                        .add(qx.QLabel().set_text('- @(open_the_source) -'), align=qx.Align.CenterF))
        else:
            bag.dispose_items()



    def _ref_preview_frame(self, p_frame : MxFacesetMaker.FParsedFrame|None ):
        if p_frame is not None and \
           p_frame.faces is not None:
            self._item_view.apply_model(self._item_view.model.set_item_count(len(p_frame.faces)))
        else:
            self._item_view.apply_model(self._item_view.model.set_item_count(0))

        self._item_view.update_items()

    @ax.task
    def _on_get_item_pixmap(self, item_id : int, size : qt.QSize ) -> qt.QPixmap:
        rect = qt.QRect(0,0, size.width(), size.height())

        pixmap = qt.QPixmap(size)
        pixmap.fill( qt.QColor(0,0,0,0) )

        qp = qt.QPainter(pixmap)
        qp.drawText(rect, qt.Qt.AlignmentFlag.AlignCenter, f'{item_id}')
        if (p_frame := self._fm.mx_preview_frame.get()) is not None:
            p_face = p_frame.faces[item_id]

            if (aligned_face_image := p_face.aligned_face_image) is not None:

                qp.drawPixmap (rect, qt.QPixmap_from_FImage(aligned_face_image))

        qp.end()
        return pixmap