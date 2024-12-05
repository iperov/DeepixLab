from common.List import MxList
from core.lib.collections import FDict

from .MxFSIPRef import MxFSIPRef


class MxFSIPRefList(MxList[MxFSIPRef]):
    PairTypeMode = MxFSIPRef.PairTypeMode

    def __init__(self,  pair_type_mode : PairTypeMode = PairTypeMode.NONE,
                        allow_rate = False,
                        allow_att_id = False,
                        state : FDict = None):
        super().__init__()
        self._pair_type_mode = pair_type_mode
        self._allow_rate = allow_rate
        self._allow_att_id = allow_att_id
        state = FDict(state)

        self.mx_removed.listen(lambda value: value.dispose())

        for value_state in state.get('values', []):
            self.append_new(state=value_state)

    def __dispose__(self):
        for value in self.values:
            value.dispose()
        super().__dispose__()

    def get_state(self) -> FDict: return FDict({'values' : [value.get_state() for value in self.values] })

    def append_new(self, state : FDict = None) -> MxFSIPRef:
        self.append(MxFSIPRef(  pair_type_mode=self._pair_type_mode,
                                allow_rate=self._allow_rate,
                                allow_att_id=self._allow_att_id,
                                state=state ))
