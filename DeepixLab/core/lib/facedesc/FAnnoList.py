from __future__ import annotations

from typing import Self

from ..collections import FDict, FList
from ..math import FAffMat2
from .FAnno import FAnno
from .FAnno2D import FAnno2D
from .FAnnoID import FAnnoID
from .FAnnoLmrk2D import FAnnoLmrk2D
from .FAnnoLmrk2D68 import FAnnoLmrk2D68
from .FAnnoLmrk2D106 import FAnnoLmrk2D106
from .FAnnoLmrk2DYSA import FAnnoLmrk2DYSA
from .FAnnoLmrk2DYSARange import FAnnoLmrk2DYSARange
from .FAnnoPose import FAnnoPose

# ^ imports used in globals() DON'T REMOVE !!!

class FAnnoList(FList[FAnno]):
    """
    list of annotations.

    New version of the same class must be placed first.
    """

    @staticmethod
    def from_state(state : FDict|None) -> FAnnoList|None:
        state = FDict(state)

        if (values_state := state.get('values', None)) is not None:
            d = globals()
            values = []
            for cls_name, value_state in values_state:
                if (cls := d.get(cls_name, None)) is not None:
                    values.append(cls.from_state(value_state))
            return FAnnoList(values)
        return None

    def get_state(self) -> FDict:
        return FDict({'values' : [ (x.__class__.__name__, x.get_state()) for x in self ],})

    def get_pose(self) -> FAnnoPose|None:
        if (pose := self.get_first_by_class(FAnnoPose)) is None:
            if (anno := self.get_first_by_class_prio([FAnnoLmrk2D])) is not None:
                pose = anno.get_pose()
        return pose

    def transform(self, mat : FAffMat2) -> Self:
        """transform annotations"""
        self = self.clone()
        self._values = tuple(   anno.transform(mat) if isinstance(anno, FAnno2D) else
                                anno for anno in self)
        return self