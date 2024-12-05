import math
from typing import Tuple

from .. import mx
from .QHRangeSlider import QHRangeSlider


class QHRangeDoubleSlider(QHRangeSlider):
    def __init__(self):
        super().__init__()

        self.__decimals = 1
        self.__decimals_pow = 10

        self.__mx_values = mx.Property[ Tuple[float,float] ]( (0,1.0),
                                                                defer=lambda n,o, prop, super=super(): super.mx_values.set( ( int(n[0]*self.__decimals_pow), int(n[1]*self.__decimals_pow) ))
                                                                ).dispose_with(self)

        super().mx_values.listen(lambda v: self.__mx_values._set((v[0]/self.__decimals_pow, v[1]/self.__decimals_pow)))

    @property
    def mx_values(self) -> mx.IProperty_v[ Tuple[float,float] ]:
        return self.__mx_values

    def get_decimals(self) -> int: return self.__decimals

    def get_range(self) -> Tuple[float, float]:
        minimum, maximum = super().get_range()
        return minimum / self.__decimals_pow, maximum / self.__decimals_pow

    def set_decimals(self, decimals : int):
        minimum, maximum = self.get_range()

        self.__decimals = decimals
        self.__decimals_pow = math.pow(10, decimals)

        self.set_range(minimum, maximum)
        return self

    def set_range(self, minimum : float, maximum : float):
        super().set_range(  int( minimum * self.__decimals_pow),
                            int( maximum * self.__decimals_pow) )
        return self
