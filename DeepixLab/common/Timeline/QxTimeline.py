import time

from core import qt, qx

from .MxTimeline import MxTimeline


class QxTimeline(qx.QVBox):
    def __init__(self, tl : MxTimeline):
        super().__init__()

        self._tl = tl

        play_btn = qx.QOnOffPushButtonMxFlag(tl.mx_playing)
        play_btn.on_button.set_icon_size(qx.Size.L).set_icon(qx.IconDB.pause_circle_outline, qt.QColor(255,0,0))
        play_btn.off_button.set_icon_size(qx.Size.L).set_icon(qx.IconDB.play_circle_outline, qt.QColor(255,0,0))

        q_eta_hbox = (qx.QHBox()
                        .add(q_label_eta := qx.QLabel())
                        .add(qx.QIconWidget().set_icon(qx.IconDB.time_outline, qx.StyleColor.Text)) )
        tl.mx_eta.reflect(lambda eta: ( q_eta_hbox.set_visible(eta != 0),
                                        q_label_eta.set_font(qx.FontDB.Digital).set_text( time.strftime('%H:%M:%S', time.gmtime(eta)) ),
                                      ) )

        (self

            .add(qx.QDoubleSliderMxNumber(tl.mx_frame_idx))

            .add(qx.QHBox()
                    .add(qx.QPushButton().set_icon_size(qx.Size.L).set_icon(qx.IconDB.play_skip_back_circle_outline, qt.QColor(255,0,0))
                               .inline(lambda btn: btn.mx_clicked.listen(lambda: tl.mx_frame_idx.set(tl.mx_play_range_begin_idx.get()) )))
                    .add(qx.QPushButton().set_icon_size(qx.Size.L).set_icon(qx.IconDB.play_back_circle_outline, qt.QColor(255,0,0))
                               .inline(lambda btn: btn.mx_clicked.listen(lambda: tl.mx_frame_idx.set(tl.mx_frame_idx.get()-1) )))
                    .add(play_btn)
                    .add(qx.QPushButton().set_icon_size(qx.Size.L).set_icon(qx.IconDB.play_forward_circle_outline, qt.QColor(255,0,0))
                               .inline(lambda btn: btn.mx_clicked.listen(lambda: tl.mx_frame_idx.set(tl.mx_frame_idx.get()+1) )))
                    .add(qx.QPushButton().set_icon_size(qx.Size.L).set_icon(qx.IconDB.play_skip_forward_circle_outline, qt.QColor(255,0,0))
                               .inline(lambda btn: btn.mx_clicked.listen(lambda: tl.mx_frame_idx.set(tl.mx_play_range_end_idx.get()) )))

                    .add(qx.QHBox()
                            .add(qx.QLabel().set_text('@(Begin)'))
                            .add(qx.QDoubleSpinBoxMxNumber(tl.mx_play_range_begin_idx))

                            .add(qx.QPushButton().h_compact().set_icon_size(qx.Size.M).set_icon(qx.IconDB.chevron_back_outline).set_tooltip('Set from current')
                                                    .inline(lambda btn: btn.mx_clicked.listen(lambda: tl.mx_play_range_begin_idx.set(tl.mx_frame_idx.get()) )))
                            .add(qx.QDoubleSpinBoxMxNumber(tl.mx_frame_idx))
                            .add(qx.QPushButton().h_compact().set_icon_size(qx.Size.M).set_icon(qx.IconDB.chevron_forward_outline).set_tooltip('Set from current')
                                                    .inline(lambda btn: btn.mx_clicked.listen(lambda: tl.mx_play_range_end_idx.set(tl.mx_frame_idx.get()) )))
                            .add(qx.QDoubleSpinBoxMxNumber(tl.mx_play_range_end_idx))
                            .add(qx.QLabel().set_text('@(End)'))
                            .add_spacer(4)
                            .add(qx.QLabel().set_text('@(Step)'))
                            .add(qx.QDoubleSpinBoxMxNumber(tl.mx_frame_step))

                            .add_spacer(4)
                            .add(q_eta_hbox)

                            # .add(qx.QLabel().set_text('ETA'))
                            # .add(q_label_eta)

                                , align=qx.Align.CenterV )
                    , align=qx.Align.LeftF)

        )



