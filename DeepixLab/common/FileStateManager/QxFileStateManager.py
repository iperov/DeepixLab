from core import mx, qx

from .MxFileStateManager import MxFileStateManager


class QxFileStateManager(qx.QVBox):

    def __init__(self, mgr : MxFileStateManager):
        super().__init__()
        self._mgr = mgr

        self._central_vbox = qx.QVBox()
        (self   .add(qx.QMxPathState(mgr.mx_path).v_compact())
                .add(self._central_vbox) )


        mgr.mx_error.listen(lambda text: wnd.q_info_bar.add_text(f"<span style='color: red'>@(Error)</span>: {text}") if (wnd := self.get_first_parent_by_class(qx.QMainWIndow)) is not None else ...).dispose_with(self)

        mgr.mx_initialized.reflect(lambda initialized, enter, bag=mx.Disposable().dispose_with(self):
                             self._ref_initialized(initialized, enter, bag)).dispose_with(self)


    def _ref_initialized(self, initialized : bool, enter : bool, bag : mx.Disposable):
        mgr = self._mgr
        if enter:

            if initialized:
                self._central_vbox.add(
                    qx.QVBox().dispose_with(bag)

                        .add( qx.QGrid().set_spacing(4)
                            .col(0)
                                .add(qx.QVBox()
                                    .add( save_btn := qx.QPushButton().set_text('@(Save)').inline(lambda btn: btn.mx_clicked.listen(lambda: mgr.save())))
                                    .inline(lambda _: mgr.mx_save_progress.mx_active.reflect(lambda active, enter: (save_btn.set_visible(not active) if enter else ...)).dispose_with(bag))
                                    .add(qx.QMxProgress(mgr.mx_save_progress, hide_inactive=True)))

                                .add(qx.QGrid().set_spacing(4).row(0)
                                        .add( qx.QLabel().set_text('@(Every)'), align=qx.Align.RightF )
                                        .add( qx.QDoubleSpinBoxMxNumber(mgr.mx_autosave) )
                                        .add( qx.QLabel().set_text('@(minutes)'), align=qx.Align.LeftF )
                                    .grid(), align=qx.Align.TopF)

                            .next_col()
                                .add(qx.QVBox()
                                    .add(backup_btn := qx.QPushButton().set_text('@(Backup)').inline(lambda btn: btn.mx_clicked.listen(lambda: mgr.save(backup=True))))
                                    .inline(lambda _: mgr.mx_backup_progress.mx_active.reflect(lambda active, enter: (backup_btn.set_visible(not active) if enter else ...)).dispose_with(bag))
                                    .add(qx.QMxProgress(mgr.mx_backup_progress, hide_inactive=True)))

                                .add(qx.QGrid().set_spacing(4).row(0)
                                        .add( qx.QLabel().set_text('@(Every)'), align=qx.Align.RightF )
                                        .add( qx.QDoubleSpinBoxMxNumber(mgr.mx_autobackup) )
                                        .add( qx.QLabel().set_text('@(minutes)'), align=qx.Align.LeftF )
                                    .next_row()
                                        .add( qx.QLabel().set_text('@(Maximum)'), align=qx.Align.RightF )
                                        .add( qx.QDoubleSpinBoxMxNumber(mgr.mx_backup_count) )
                                        .add( qx.QLabel().set_text('@(backups)'), align=qx.Align.LeftF )
                                    .grid(), align=qx.Align.TopF)

                            .grid(), align=qx.Align.CenterH )

                        )
            else:
                self._central_vbox.add(qx.QMxProgress(self._mgr.mx_loading_progress, hide_inactive=True).dispose_with(bag), align=qx.Align.CenterE)

        else:
            bag.dispose_items()

