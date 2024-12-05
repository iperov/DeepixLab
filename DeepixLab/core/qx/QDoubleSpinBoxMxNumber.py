from .. import mx
from .QDoubleSpinBox import QDoubleSpinBox


class QDoubleSpinBoxMxNumber(QDoubleSpinBox):
    def __init__(self, number : mx.INumber_v):
        super().__init__()
        self._number = number

        config = number.config
        self.set_decimals(config.decimals)
        self.set_single_step(config.step)
        self.set_minimum(config.min)
        self.set_maximum(config.max)
        self.set_read_only(config.read_only)
        
        if config.zero_is_auto:
            self.set_special_value_text('@(Auto)')
    
        self._conn = self.mx_value.listen(lambda value: self._on_spinbox_value(value))

        number.reflect(self._on_number).dispose_with(self)

    def _on_number(self, value):
        with self._conn.disabled_scope():
            self.set_value(value)

    def _on_spinbox_value(self, value):
        if self._number.config.decimals == 0:
            value = int(value)
        else:
            value = float(value)

        self._number.set(value)
