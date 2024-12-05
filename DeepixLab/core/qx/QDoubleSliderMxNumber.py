from .. import mx
from .QDoubleSlider import QDoubleSlider


class QDoubleSliderMxNumber(QDoubleSlider):
    def __init__(self, number : mx.INumber_v):
        super().__init__()
        self._number = number

        config = number.config
        
        self.set_decimals(config.decimals)
        self.set_step(config.step)
        self.set_minimum(config.min)
        self.set_maximum(config.max)

        self._conn = self.mx_value.listen(lambda value: self._on_slider_value(value))
        number.reflect(self._on_number).dispose_with(self)


    def _on_number(self, value : float ):
        with self._conn.disabled_scope():
            self.set_value(value)

    def _on_slider_value(self, value : float):        
        self._number.set(value)
