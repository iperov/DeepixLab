from enum import Enum, auto
from pathlib import Path

from core import mx
from core.lib.collections import FDict, HFDict

from .MxPairType import MxPairType


class MxFSIPRef(mx.Disposable):

    class PairTypeMode(Enum):
        NONE = auto()
        SINGLE_CHOICE = auto()

    def __init__(self,  pair_type_mode : PairTypeMode = PairTypeMode.NONE,
                        allow_rate = False,
                        allow_att_id = False,
                        state : FDict = None):
        super().__init__()
        self._pair_type_mode = pair_type_mode
        self._allow_rate = allow_rate
        self._allow_att_id = allow_att_id

        self._state = state = HFDict(state)

        fsip_path_state = HFDict(state.get('fsip_path_state', None))

        bag = mx.Disposable().dispose_with(self)
        self._mx_fsip_path = mx.Path(   config=mx.Path.Config(dir=True, desc="Dataset directory"),
                                        on_close=lambda bag=bag: (bag.dispose_items(), fsip_path_state.clear()),
                                        on_open=lambda path, bag=bag:
                                                self._on_open(path, fsip_path_state, bag)).dispose_with(self)

        state_upd = lambda: state.update({  'fsip_path' : self._mx_fsip_path.get(),
                                            'fsip_path_state' : fsip_path_state})

        self._update_state_ev = mx.Event0().dispose_with(self)
        self._update_state_ev.listen(state_upd).dispose_with(self)
        mx.CallOnDispose(state_upd).dispose_with(self)

        if (path := state.get('fsip_path', None)) is not None:
            self._mx_fsip_path.open(path)


    @property
    def pair_type_mode(self) -> PairTypeMode: return self._pair_type_mode
    @property
    def allow_rate(self) -> bool: return self._allow_rate
    @property
    def allow_att_id(self) -> bool: return self._allow_att_id

    @property
    def mx_fsip_path(self) -> mx.IPath_v: return self._mx_fsip_path

    @property
    def mx_pair_type(self) -> MxPairType|None:
        """avail if mx_fsip_path is opened
        and
        ```
        pair_type_mode == NONE          -> None
        pair_type_mode == SINGLE_CHOICE -> mx.IStateChoice_v[MxFSIPRef.PairType|str]

        ```"""
        return self._mx_pair_type

    @property
    def mx_rate(self) -> mx.INumber_v|None:
        """avail if _allow_rate and mx_fsip_path is opened"""
        return self._mx_rate

    @property
    def mx_att_id(self) -> mx.INumber_v|None:
        """avail if _allow_att_id and mx_fsip_path is opened"""
        return self._mx_att_id

    def get_state(self) -> FDict:
        self._update_state_ev.emit(reverse=True)
        return FDict(self._state)


    def _on_open(self, path : Path, state : HFDict, bag : mx.Disposable):
        if self._pair_type_mode == self.PairTypeMode.NONE:
            state_upd = lambda: ...

        elif self._pair_type_mode == self.PairTypeMode.SINGLE_CHOICE:
            self._mx_pair_type = MxPairType(path, state.get('pair_type', None) ).dispose_with(bag)
            state_upd = lambda: state.update({'pair_type' : self._mx_pair_type.get_state() })

        if self._allow_rate:
            self._mx_rate = mx.Number(state.get('rate', 100), config=mx.Number.Config(1, 100)).dispose_with(bag)
            state_upd = lambda state_upd=state_upd: (state_upd(), state.update({'rate' : self._mx_rate.get()} ))

        if self._allow_att_id:
            self._mx_att_id = mx.Number(state.get('att_id', 0), config=mx.Number.Config(0, 32)).dispose_with(bag)
            state_upd = lambda state_upd=state_upd: (state_upd(), state.update({'att_id' : self._mx_att_id.get()} ))

        self._update_state_ev.listen(state_upd).dispose_with(bag)
        mx.CallOnDispose(state_upd).dispose_with(bag)

        return True