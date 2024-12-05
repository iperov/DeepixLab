from pathlib import Path

from core import mx
from core.lib.collections import FDict
from core.lib.dataset.FSIP import FSIPInfo


class _NO_PAIR:
    def __str__(self) -> str: return '- @(no_pair) -'

class MxPairType( mx.StateChoice[_NO_PAIR|str] ):
    """
    pair_type StateChoice controller
    """
    # Public constant
    NO_PAIR = _NO_PAIR()

    def __init__(self, fsip_root : Path, state : FDict = None):
        fsip_info = FSIPInfo(fsip_root)
        super().__init__( availuator=lambda: (self.NO_PAIR,)+fsip_info.load_pair_types()  )
        self._fsip_info = fsip_info

        state = FDict(state)
        self.set(state.get('pair_type', None), MxPairType.NO_PAIR)

    def get_state(self) -> FDict:
        return FDict({'pair_type':  pair_type
                                    if ( pair_type := self.get() ) not in [None, MxPairType.NO_PAIR]
                                    else None   })

    @property
    def fsip_info(self) -> FSIPInfo:
        return self._fsip_info


