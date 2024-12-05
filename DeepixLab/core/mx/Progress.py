import time
from typing import Self

from .Disposable import Disposable
from .FProperty import FProperty, IFProperty_rv
from .State import IState_rv, State


class FModel:
    def __init__(self):
        self._caption : str|None = None
        self._progress = 0
        self._progress_max = 0

        self._first_step_time = None
        self._last_step_time = None
        self._step_count = 0

    def clone(self) -> Self:
        f = self.__class__.__new__(self.__class__)
        f._caption = self._caption
        f._progress = self._progress
        f._progress_max = self._progress_max
        f._first_step_time = self._first_step_time
        f._last_step_time = self._last_step_time
        f._step_count = self._step_count

        return f

    def _sps_reset(self) -> Self:
        self = self.clone()
        self._first_step_time = None
        self._last_step_time = None
        self._step_count = 0
        return self

    def _sps_step(self, amount=1) -> Self:
        self = self.clone()
        t = time.perf_counter()
        if self._first_step_time is None:
            self._first_step_time = t
        self._last_step_time = t
        self._step_count += amount
        return self

    @property
    def caption(self) -> str|None: return self._caption
    @property
    def is_infinity(self) -> bool: return self._progress_max is None
    @property
    def progress(self) -> int: 
        """avail if not is_infinity"""
        return self._progress
    @property
    def progress_max(self) -> int: 
        """avail if not is_infinity"""
        return self._progress_max
    @property
    def it_s(self) -> float: 
        if self._first_step_time is not None:
            time_diff = self._last_step_time - self._first_step_time
            if time_diff != 0:
                return self._step_count / time_diff
        return 0
    
    def inc(self) -> Self: return self.set(self._progress+1)
    
    def set_caption(self, caption : str|None) -> Self:
        self = self.clone()
        self._caption = caption
        return self
    
    def set_inf(self) -> Self:
        self = self.clone()
        self._progress = None
        self._progress_max = None
        return self

    def set(self, progress : int, progress_max : int = None) -> Self:
        self = (old_self := self).clone()
        self._progress = progress
        if progress_max is not None:
            self._progress_max = progress_max
        if progress == 0:
            self = self._sps_reset()
        else:
            old_progress = old_self._progress
            if old_progress is None:
                old_progress = 0
            diff = max(0, progress - old_progress)
            self = self._sps_step(diff)

        return self

class IProgress_rv:
    FModel = FModel
    """view interface of Progress"""
    @property
    def mx_active(self) -> IState_rv[bool]:
        """indicates progress state"""
    @property
    def mx_model(self) -> IFProperty_rv[FModel]:
        """avail if mx_active == True"""


class Progress(Disposable, IProgress_rv):
    """
    Int value/infinite progress control with caption.
    """

    def __init__(self):
        """"""
        super().__init__()

        self._mx_state = State[bool]().set(False).dispose_with(self)
        self._mx_model = FProperty[FModel](FModel()).dispose_with(self)

    @property
    def mx_active(self) -> IState_rv[bool]: return self._mx_state
    @property
    def mx_model(self) -> IFProperty_rv[FModel]: return self._mx_model

    def start(self) -> Self:
        """switch to active state, set None caption, set inf progress"""
        if not self._mx_state.get():
            self._mx_state.set(True)
            self.set_caption(None).set_inf()
        return self

    def finish(self) -> Self:
        """switch to inactive state"""
        if self._mx_state.get():
            self._mx_state.set(False)
        return self

    def set_caption(self, caption : str|None) -> Self:
        """set or remove caption"""
        self._mx_model.set(self._mx_model.get().set_caption(caption))
        return self

    def set_inf(self) -> Self:
        self._mx_model.set(self._mx_model.get().set_inf())
        return self

    def set(self, progress : int, progress_max : int = None) -> Self:
        self._mx_model.set(self._mx_model.get().set(progress, progress_max))
        return self

    def inc(self) -> Self:
        self._mx_model.set(self._mx_model.get().inc())
        return self
