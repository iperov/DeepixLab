from common.ImageFormat import QxImageFormat
from core import mx, qx

from .MxExport import MxExport


class QxExport(qx.QVBox):
    def __init__(self, export : MxExport):
        super().__init__()
        self._export = export
        self._export_task = None

        export.mx_error.listen(lambda text: wnd.q_info_bar.add_text(f"<span style='color: red'>@(Error)</span>: {text}") if (wnd := self.get_first_parent_by_class(qx.QMainWIndow)) is not None else ...).dispose_with(self)

        (self

            .add(qx.QGrid()
                .row(0)
                    .add(qx.QLabel().set_text('@(Input_directory)'), align=qx.Align.RightF)
                    .add(qx.QMxPathState(export.mx_input_path))
                .next_row()
                    .add(qx.QLabel().set_text('@(Output_directory)'), align=qx.Align.RightF)
                    .add(qx.QMxPathState(export.mx_output_path))
                .next_row()
                    .add(qx.QLabel().set_text('@(Patch_mode)'), align=qx.Align.RightF)
                    .add( qx.QHBox()
                            .add(qx.QCheckBoxMxFlag(export.mx_patch_mode))
                            .add(patch_mode_holder := qx.QHBox() ), align=qx.Align.LeftF)

                .next_row()
                    .add(qx.QLabel().set_text('@(Levels_range)'), align=qx.Align.RightF)
                    .add( qx.QHBox()
                            .add(qx.QDoubleSpinBoxMxNumber(export.mx_levels_min))
                            .add(qx.QLabel().set_text('-'))
                            .add(qx.QDoubleSpinBoxMxNumber(export.mx_levels_max))

                        , align=qx.Align.LeftF)

                .next_row()
                    .add(qx.QLabel().set_text('@(File_format)'), align=qx.Align.RightF)
                    .add( QxImageFormat(export.mx_image_format), col_span=2, align=qx.Align.LeftF)

                .next_row()
                    .add(qx.QLabel().set_text('@(Delete_output_directory)'), align=qx.Align.RightF)
                    .add(qx.QCheckBoxMxFlag(export.mx_delete_output_directory), align=qx.Align.LeftF)

                .grid())

            .add_spacer(4)

            .add( (export_btn := qx.QPushButton()).set_text('@(Start)')
                            .inline(lambda btn: btn.mx_clicked.listen(lambda: export.start()).dispose_with(self)))

            .add( qx.QHBox()
                    .add( qx.QMxProgress(export.mx_progress, hide_inactive=True).set_show_it_s(True).set_font(qx.FontDB.FixedWidth).h_expand())
                    .add( (cancel_btn := qx.QPushButton()).h_compact().set_text('@(Cancel)')
                                    .inline(lambda btn: btn.mx_clicked.listen(lambda: export.stop()).dispose_with(self)))) )

        export.mx_progress.mx_active.reflect(lambda active, enter:
            (export_btn.hide(),
             cancel_btn.show()) if active and enter else
            (export_btn.show(),
             cancel_btn.hide())
        ).dispose_with(self)

        export.mx_patch_mode.reflect(lambda patch_mode, bag=mx.Disposable().dispose_with(self):
                                     self._ref_patch_mode(patch_mode, patch_mode_holder, bag))


    def _ref_patch_mode(self, patch_mode, holder : qx.QHBox, bag : mx.Disposable):
        bag.dispose_items()
        if patch_mode:
            holder.add( qx.QHBox().dispose_with(bag)
                            .add(qx.QLabel().set_text('@(Sample_count)'))
                            .add(qx.QDoubleSpinBoxMxNumber(self._export.mx_sample_count))
                            .add_spacer(4)
                            .add(qx.QCheckBoxMxFlag(self._export.mx_fix_borders).set_text('@(Fix_borders)'))
                            )
