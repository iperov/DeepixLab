from typing import Any, Callable, Sequence

from .. import mx
from .QBox import QHBox, QVBox
from .QCheckBox import QCheckBox


class QCheckBoxMxMultiChoice(QVBox):
    def __init__(self,  mc : mx.IMultiChoice_v,
                        stringifier : Callable[ [Any], str ] = None,
                        **kwargs):
        super().__init__(**kwargs)
        self._mc = mc

        if stringifier is None:
            stringifier = lambda val: '' if val is None else str(val)
        self._stringifier = stringifier

        self._avail = None
        
        self._holder = QHBox()
        self.add(self._holder)
        
        self._bag = mx.Disposable().dispose_with(self)

        mc.listen(lambda _: self.update_items()).dispose_with(self)
        mc.mx_avail.reflect(lambda new_avail: 
                            self._ref_avail(new_avail)).dispose_with(self)

        # Currently no reevaluation button

    def _ref_avail(self, new_avail : Sequence):
        if self._avail != new_avail:
            self._avail = new_avail
            self.update_items()

    def update_items(self):
        bag = self._bag
        bag.dispose_items()
        
        avail  = self._mc.mx_avail.get()
        values = self._mc.get()
        
        self._holder.add( holder := QHBox().dispose_with(bag))
        for v in avail:
            q_checkbox = QCheckBox().set_text(self._stringifier(v)).set_checked(v in values)
            q_checkbox.mx_toggled.listen(lambda checked, v=v: (self._mc.update_added(v) if checked else self._mc.update_removed(v)) )

            holder.add(q_checkbox)

