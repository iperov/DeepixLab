from typing import Any, Callable

from .. import mx
from .QComboBox import QComboBox


class QComboBoxMxStateChoice(QComboBox):
    def __init__(self,  sc : mx.IStateChoice_v,
                        stringifier : Callable[ [Any], str ] = None,
                        allow_unselected = False, 
                        **kwargs):
        """
        
            allow_unselected        None(undefined state) of StateChoice
                                    can be selected by user
        """
        super().__init__(**kwargs)
        self._sc = sc
        if stringifier is None:
            stringifier = lambda val: '' if val is None else str(val)
        self._stringifier = stringifier
        self._allow_unselected = allow_unselected

        self._idx_funcs = []
        
        self._conn = self.mx_current_index.listen(lambda idx: self._idx_funcs[idx]())
        
        sc.listen(lambda value, enter: self.update_items() if enter else ...).dispose_with(self)
        sc.mx_avail.reflect(lambda _: self.update_items()).dispose_with(self)
        

    def update_items(self):
        avail = self._sc.mx_avail.get()
        value = self._sc.get()
        

        with self._conn.disabled_scope():
            self.clear()
            try:
                idx = avail.index(value)
            except:
                idx = None
    
            idx_funcs = self._idx_funcs = []
            i = 0
            
            if self._allow_unselected or value is None:
                self.add_item('@(Menu_unselected)')
                idx_funcs.append(lambda: self._sc.set(None))
                i += 1
            
            for x in avail:
                self.add_item(self._stringifier(x))
                idx_funcs.append(lambda x=x: self._sc.set(x))
                i += 1

            if idx is not None:
                self.set_current_index(idx + (1 if self._allow_unselected else 0) )
            elif value is None:
                self.set_current_index(0)
                

    def show_popup(self) -> None:
        # Reevaluate avail items on open combobox 
        self._sc.reevaluate()        
        super().show_popup()
