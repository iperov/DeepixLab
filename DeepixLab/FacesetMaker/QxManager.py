from __future__ import annotations

from common.MultiStateManager import QxMultiStateManager
from core import mx, qx

from .MxManager import MxManager
from .QxFacesetMaker import QxFacesetMaker


class QxManager(qx.QVBox):
    def __init__(self, mgr : MxManager):
        super().__init__()
        self._mgr = mgr
        self._media_source_head_holder = qx.QVBox()

        self._q_central_vbox = qx.QVBox()

        (self   .add( qx.QHBox().v_compact()
                        .add(QxMultiStateManager(mgr.mx_multistate_state_mgr))
                        .add(self._media_source_head_holder)
                        )
                .add_spacer(4)
                .add(self._q_central_vbox))

        mgr.mx_multistate_state_mgr.mx_state.reflect(lambda state_key, enter, bag=mx.Disposable().dispose_with(self):
                                                     self._ref_state(state_key, enter, bag)).dispose_with(self)

    def _ref_state(self, state_key, enter : bool, bag : mx.Disposable):
        if enter:
            q_fm = QxFacesetMaker(self._mgr.mx_faceset_maker,
                                  media_source_head_holder=self._media_source_head_holder).dispose_with(bag)
            self._q_central_vbox.add(q_fm)
        else:
            bag.dispose_items()
