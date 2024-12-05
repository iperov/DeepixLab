from .. import mx
from .QSlider import QSlider


class QDoubleSlider(QSlider):
    def __init__(self, **kwargs):
        super().__init__()
        
        self._decimals = 0        
        self._step = 1
        self._minimum = 0
        self._maximum = 1
        
        self.__mx_value = mx.GetSetProperty[float](lambda: QDoubleSlider.get_value(self),
                                                   lambda val: QDoubleSlider.set_value(self, val),
                                                   super().mx_value.event).dispose_with(self)
        
        
    @property
    def mx_value(self) -> mx.IProperty_v[float]: return self.__mx_value
    
    def get_value(self) -> float: return (super().get_value() * self._step) + self._minimum
    def set_value(self, value : float):  return super().set_value( int( (value-self._minimum) / self._step ) )

    def set_decimals(self, decimals : int): self._set_config(decimals=decimals)
    def set_step(self, step : float): self._set_config(step=step)
    def set_minimum(self, minimum : float): self._set_config(minimum=minimum)
    def set_maximum(self, maximum : float): self._set_config(maximum=maximum)
        
    def _set_config(self,   minimum = None,
                            maximum = None,
                            step = None,
                            decimals = None):
        v = self.__mx_value.get()
        
        minimum  = self._minimum = minimum if minimum is not None else self._minimum
        maximum  = self._maximum = maximum if maximum is not None else self._maximum
        step = self._step = step if step is not None else self._step
        decimals = self._decimals = decimals if decimals is not None else self._decimals
        
        s_min = 0
        s_max = int( (maximum-minimum) / step )
        
        super().set_minimum( s_min )
        super().set_maximum( s_max )
        
        self.__mx_value.set(v)