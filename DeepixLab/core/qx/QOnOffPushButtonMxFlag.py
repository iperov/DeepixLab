from __future__ import annotations

from .. import mx
from .QBox import QVBox
from .QPushButton import QPushButton


class QOnOffPushButtonMxFlag(QVBox):
    def __init__(self, flag : mx.IFlag_rv):
        """
        
        """
        super().__init__()
        self._flag = flag

        on_button = self._on_button = QPushButton()
        on_button.mx_clicked.listen(lambda: flag.set(False)) 
        
        
        off_button = self._off_button = QPushButton()
        off_button.mx_clicked.listen(lambda: flag.set(True)) 
       
        self.add(on_button).add(off_button)

        flag.reflect(lambda flag:   ( on_button.set_visible(flag),
                                      off_button.set_visible(not flag),
                                    )).dispose_with(self)

    @property
    def on_button(self) -> QPushButton: return self._on_button
    @property
    def off_button(self) -> QPushButton: return self._off_button
