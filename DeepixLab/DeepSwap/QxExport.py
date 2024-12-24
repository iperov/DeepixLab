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
                    .add(qx.QLabel().set_text('@(Output_directory) SWAP'), align=qx.Align.RightF)
                    .add(qx.QMxPathState(export.mx_output_swap_path))
                .next_row()
                    .add(qx.QLabel().set_text('@(Output_directory) SWAP_GUIDE'), align=qx.Align.RightF)
                    .add(qx.QMxPathState(export.mx_output_swap_guide_path))
                .next_row()
                    .add(qx.QLabel().set_text('@(Output_directory) REC_GUIDE'), align=qx.Align.RightF)
                    .add(qx.QMxPathState(export.mx_output_rec_guide_path))
                    
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
