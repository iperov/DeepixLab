from __future__ import annotations

import dataclasses as dc
from typing import Callable

from .Property import IProperty_rv, IProperty_v, Property


@dc.dataclass
class Config:
    min : int|float
    max : int|float
    step : int|float = 1
    read_only : bool = False
    zero_is_auto : bool = False # hint for ViewController
    
    decimals : int = dc.field(init=False) # .step digits after point
    
    def __post_init__(self):
        self.decimals = len(sx.split('.')[1]) if '.' in (sx := str(self.step)) else 0
    
    def filter(self, new_value : int|float, value : int|float) -> int|float:
        if self.read_only:
            return value

        step = self.step
        if step != 1:
            new_value = round(new_value / step) * step

        new_value = min(max(self.min, new_value), self.max)

        return new_value

class INumber_rv(IProperty_rv[int|float]):
    """view interface of Number"""
    Config = Config
    
    @property
    def config(self) -> Config: ...
  
  
class INumber_v(IProperty_v[int|float]):
    """view interface of Number"""
    Config = Config
    
    @property
    def config(self) -> Config: ...


class Number(Property[int|float], INumber_v):
    """
    Number is Property of NumberType, filterable by config.
    """
    def __init__(self,  number : int|float,
                        config : Config|None = None,

                        filter : Callable[[int|float, int|float], int|float] = None,
                        defer : Callable[ [int|float, int|float, Number], None ] = None,
                 ):

        self.__config = config = config if config is not None else Number.Config()

        if filter is not None:
            filter_func=lambda n, o: filter(config.filter(n, o), o)
        else:
            filter_func=lambda n, o: config.filter(n, o)
        
        super().__init__(number, filter=filter_func, defer=defer)

    @property
    def config(self) -> Config:  return self.__config
    

