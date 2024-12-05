from typing import Self

from ..collections import FDict


class FAnno:
    """base class for face annotations"""

    def clone(self) -> Self:
        f = self.__class__.__new__(self.__class__)
        return f

    def get_state(self) -> FDict:
        raise NotImplementedError()

    def __str__(self): return f'{self.__class__.__name__}'
    def __repr__(self): return self.__str__()