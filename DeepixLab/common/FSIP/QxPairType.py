import re

from core import mx, qx
from core.lib.dataset.FSIP import IFSIP_v

from .MxPairType import MxPairType


class QxPairType(qx.QVBox):

    def __init__(self, mx_pair_type : MxPairType):
        """
        view-model for MxPairType.

        Select, reveal in explorer,
        optional: add, delete
        """
        super().__init__()
        self._mx_pair_type = mx_pair_type

        self._q_main_hbox = (
            qx.QHBox()
                .add( qx.QComboBoxMxStateChoice(self._mx_pair_type))
                .add( qx.QRevealInExplorerButton(lambda:self._mx_pair_type._fsip_info.get_pair_dir_path(pair_type)
                                                        if (pair_type := self._mx_pair_type.get()) not in [None, MxPairType.NO_PAIR]
                                                        else None )))

        self.add(self._q_main_hbox)


class QxPairTypeAddDel(QxPairType):

    def __init__(self, mx_pair_type : MxPairType, fsip : IFSIP_v):
        """
        + add, delete
        """
        super().__init__(mx_pair_type)
        self._fsip = fsip

        self._mx_error = mx.Event1[str]().dispose_with(self)

        def set_enabled_buttons(b : bool):
            btn_new_pair_type.set_enabled(b)
            btn_delete_pair_type.set_enabled(b)

        btn_new_pair_type = qx.QPushButton().h_compact().set_tooltip('@(Add)').set_icon(qx.IconDB.add_outline)
        btn_new_pair_type.inline(lambda btn: btn.mx_clicked.listen(lambda: (set_enabled_buttons(False), self._new_pair_type_dlg.popup()) ))

        dlg = self._new_pair_type_dlg = qx.QInplaceInputDialog()
        dlg.q_label_title.set_text('@(Name)')
        dlg.q_lineedit.set_filter(lambda s: re.sub(r'\W', '', s))
        dlg.q_btn_ok.mx_clicked.listen(lambda: (set_enabled_buttons(True), self._on_btn_new_pair_type()))
        dlg.q_btn_cancel.mx_clicked.listen(lambda: set_enabled_buttons(True))

        self._q_main_hbox.add(btn_new_pair_type)
        self.add(dlg)

        btn_delete_pair_type = qx.QPushButton().h_compact().set_tooltip('@(Delete)').set_icon(qx.IconDB.trash_outline)
        btn_delete_pair_type.inline(lambda btn: btn.mx_clicked.listen(lambda: (set_enabled_buttons(False), self._delete_pair_type_dlg.popup()) ))

        dlg = self._delete_pair_type_dlg = qx.QInplaceDialog()
        dlg.q_label_title.set_text("<span style='color: red'>@(Warning):</span> @(Pair_type_files_will_be_deleted)")
        dlg.q_btn_ok.mx_clicked.listen(lambda: (set_enabled_buttons(True), self._on_btn_delete_pair_type()))
        dlg.q_btn_cancel.mx_clicked.listen(lambda: (set_enabled_buttons(True), ))

        self._q_main_hbox.add(btn_delete_pair_type)


        self.add(dlg)

    @property
    def mx_error(self) -> mx.IEvent1_rv[str]:
        return self._mx_error

    def _on_btn_new_pair_type(self):
        try:
            pair_type = self._new_pair_type_dlg.q_lineedit.get_text()
            self._fsip.add_pair_type(pair_type)
            self._mx_pair_type.reevaluate()
            self._mx_pair_type.set(pair_type)

        except Exception as e:
            self._mx_error.emit(str(e))

    def _on_btn_delete_pair_type(self):
        try:
            if (pair_type := self._mx_pair_type.get()) not in [None, MxPairType.NO_PAIR]:
                self._fsip.delete_pair_type(pair_type)

                self._mx_pair_type.reevaluate()
                self._mx_pair_type.set(self._mx_pair_type.mx_avail.get()[0])
        except Exception as e:
            self._mx_error.emit(str(e))
