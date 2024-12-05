from pathlib import Path

from common.MultiStateManager import MxMultiStateManager
from core import ax, mx
from core.lib.collections import FDict

from .MxFacesetMaker import MxFacesetMaker


class MxManager(mx.Disposable):
    def __init__(self, state_path : Path):
        super().__init__()

        self._fg = ax.FutureGroup().dispose_with(self)
        self._main_thread = ax.get_current_thread()

        bag = mx.Disposable().dispose_with(self)

        self._mx_faceset_maker = None

        self._default_state_name = '@(Default)'
        self._mx_multistate_state_mgr = MxMultiStateManager(state_path,
                                                            default_state_name=self._default_state_name,
                                                            on_close=lambda state_name, bag=bag: self._on_close(state_name, bag),
                                                            on_load=lambda state, state_name, bag=bag: self._on_load(state, state_name, bag),
                                                            get_state=lambda state_name: self._get_state(state_name),
                                                            ).dispose_with(self)

        self._mx_multistate_state_mgr.mx_state.set(self._default_state_name)


    @property
    def mx_multistate_state_mgr(self) -> MxMultiStateManager:
        return self._mx_multistate_state_mgr

    @property
    def mx_faceset_maker(self) -> MxFacesetMaker:
        """avail if mx_multistate_state_mgr.mx_state is set"""
        return self._mx_faceset_maker


    def _on_close(self, state_name : str, bag : mx.Disposable):
        bag.dispose_items()

    def _on_load(self, state : FDict, state_name : str, bag : mx.Disposable):
        self._mx_faceset_maker = MxFacesetMaker(state=state.get('faceset_maker', None)).dispose_with(bag)

    @ax.task
    def _get_state(self, state_name : str) -> FDict:
        yield ax.switch_to(self._main_thread)
        yield ax.attach_to(self._fg)

        return FDict({'faceset_maker' : self._mx_faceset_maker.get_state(),
                        })