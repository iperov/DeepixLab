from .. import mx
from ..lib.collections import HFDict


class QSettings:
    @property
    def state(self) -> HFDict: 
        """
        Mutable HFDict that loaded/saved by QApplication
        
        You can either mutate the state dynamically when your data changes,
        or do that on ev_update
        """
        
    @property
    def ev_update(self) -> mx.IEvent0_rv:
        """
        event emitted if QApplication requires update state now. Typically before saving the state to the disk.
        
        You should commit your changes to .state on event.
        """
        