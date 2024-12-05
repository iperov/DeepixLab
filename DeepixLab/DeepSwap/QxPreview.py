from common.SSI import QSSI, SSI
from core import mx, qt, qx

from .MxPreview import MxPreview


class QxPreview(qx.QVBox):
    def __init__(self, preview : MxPreview):
        super().__init__()
        self._preview = preview

        preview.mx_error.listen(lambda text: wnd.q_info_bar.add_text(f"<span style='color: red'>@(Error)</span>: {text}") if (wnd := self.get_first_parent_by_class(qx.QMainWIndow)) is not None else ...).dispose_with(self)

        self._source_type_vbox = qx.QVBox()

        self._q_sheet_vbox = qx.QHBox()
        (self
            .add(self._q_sheet_vbox)
            .add(qx.QHBox().v_compact()
                .add(qx.QLabel().set_text('@(Source)').h_compact())
                .add(qx.QComboBoxMxStateChoice( preview.mx_source_type,
                                                stringifier=lambda val: {MxPreview.SourceType.DataGenerator : '@(Data_generator)',
                                                                         MxPreview.SourceType.Directory : '@(Directory)',
                                                                        }[val],
                                                ).h_compact())

                .add(self._source_type_vbox)

                , align=qx.Align.CenterH)
        )

        preview.mx_ssi_sheet.reflect(lambda ssi_sheet, bag=mx.Disposable().dispose_with(self):
                                     self._ref_sheet(ssi_sheet, bag)).dispose_with(self)
        preview.mx_source_type.reflect(lambda source_type, enter, bag=mx.Disposable().dispose_with(self):
                                       self._ref_source_type(source_type, enter, bag)).dispose_with(self)

    def _ref_source_type(self, source_type : MxPreview.SourceType, enter : bool, bag : mx.Disposable):
        if enter:
            if source_type == MxPreview.SourceType.DataGenerator:
                generate_shortcut = qx.QShortcut(qt.QKeyCombination(qt.Qt.Key.Key_P)).set_parent(self._source_type_vbox).dispose_with(bag)
                generate_shortcut.mx_press.listen(lambda: self._preview.generate_one())

                self._source_type_vbox.add( qx.QPushButton().dispose_with(bag)
                                                .set_text(f'@(Generate) {qx.hfmt.colored_shortcut_keycomb(generate_shortcut)}').inline(lambda btn: (btn.mx_clicked.listen(lambda: generate_shortcut.press()), btn.mx_released.listen(lambda: generate_shortcut.release())) ) )
            elif source_type == MxPreview.SourceType.Directory:
                self._source_type_vbox.add(
                    qx.QHBox().dispose_with(bag)
                        .add(qx.QMxPathState(self._preview.mx_directory_path))
                        .add(sub_holder := qx.QVBox())
                    )

                self._preview.mx_directory_path.mx_opened.reflect(lambda opened, enter, bag=mx.Disposable().dispose_with(bag):
                                                                  self._ref_directory_path_state(opened, enter, sub_holder, bag)).dispose_with(bag)
        else:
            bag.dispose_items()

    def _ref_directory_path_state(self, opened : bool, enter : bool, holder : qx.QVBox, bag : mx.Disposable):
        if enter:
            if opened:
                holder.add(
                    qx.QHBox().dispose_with(bag).set_spacing(4)
                        .add( qx.QHBox()
                                .add(qx.QLabel().set_text('@(Image_index)'))
                                .add(qx.QDoubleSpinBoxMxNumber(self._preview.mx_directory_image_idx)))
                    )

        else:
            bag.dispose_items()

    def _ref_sheet(self, ssi_sheet : SSI.Sheet, bag : mx.Disposable ):
        bag.dispose_items()
        self._q_sheet_vbox.add(QSSI.Sheet(ssi_sheet).dispose_with(bag))
