from core import mx, qx

from .MxFSIPRef import MxFSIPRef
from .QxPairType import QxPairType


class QxFSIPRef(qx.QHBox):
    def __init__(self, item : MxFSIPRef):
        super().__init__()
        self._item = item
        self._q_main_hbox = qx.QHBox()
        (self
            .add(qx.QMxPathState(item.mx_fsip_path))
            .add(self._q_main_hbox))

        item.mx_fsip_path.mx_opened.reflect(lambda opened, enter, bag=mx.Disposable().dispose_with(self):
                                           self._ref_path(opened, enter, bag) ).dispose_with(self)


    def _ref_path(self, opened : bool, enter : bool, bag : mx.Disposable):
        if enter:
            q_main_hbox = self._q_main_hbox

            if opened:
                q_main_hbox.add(sub_holder := qx.QHBox().dispose_with(bag))

                pair_type_mode = self._item.pair_type_mode
                if pair_type_mode == MxFSIPRef.PairTypeMode.SINGLE_CHOICE:
                    sub_holder.add(QxPairType(self._item.mx_pair_type))

                if self._item.allow_rate:
                    sub_holder.add(qx.QLabel().set_text('@(Rate)'))
                    sub_holder.add(qx.QDoubleSpinBoxMxNumber(self._item.mx_rate))

                if self._item.allow_att_id:
                    sub_holder.add(qx.QLabel().set_text('@(AttID)'))
                    sub_holder.add(qx.QDoubleSpinBoxMxNumber(self._item.mx_att_id))
        else:
            bag.dispose_items()