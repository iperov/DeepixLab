from __future__ import annotations

from typing import Self

from ..collections import FDict
from .FAnno import FAnno


class FAnnoPose(FAnno):
    """describes the pose of the face"""

    @staticmethod
    def from_state(state : FDict|None) -> FAnnoPose|None:
        state = FDict(state)
        if (pitch := state.get('pitch', None)) is not None and \
           (yaw := state.get('yaw', None)) is not None and \
           (roll := state.get('roll', None)) is not None:
            return FAnnoPose(pitch, yaw, roll)
        return None

    def __init__(self, pitch : float, yaw : float, roll : float):
        """in radians"""
        super().__init__()
        self._pitch = pitch
        self._yaw = yaw
        self._roll = roll

    def clone(self) -> Self:
        f = super().clone()
        f._pitch = self._pitch
        f._yaw = self._yaw
        f._roll = self._roll
        return f

    def get_state(self) -> FDict: return FDict({'pitch' : self._pitch,
                                                'yaw' : self._yaw,
                                                'roll' : self._roll,
                                                })

    @property
    def pitch(self) -> float:
        """in rad"""
        return self._pitch
    @property
    def yaw(self) -> float:
        """in rad"""
        return self._yaw
    @property
    def roll(self) -> float:
        """in rad"""
        return self._roll
