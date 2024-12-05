from __future__ import annotations

from datetime import datetime
from typing import Self, Tuple

from core.lib.time import FSPSCounter


class FTimeline:
    """Timeline model"""

    def __init__(self, frame_count : int):
        self._frame_count  : int    = frame_count
        self._fps          : int    = 0
        self._frame_step   : int    = 1
        self._playing      : bool   = False
        self._autorewind   : bool   = False
        self._frame_idx    : int    = 0
        self._play_range   : Tuple[int, int] = (0, frame_count-1)
        self._frame_ts : float =  None

        self._sps_counter = FSPSCounter()
        self._eta : float = 0

    def clone(self) -> Self:
        f = self.__class__.__new__(self.__class__)
        f._frame_count = self._frame_count
        f._fps = self._fps
        f._frame_step = self._frame_step

        f._playing = self._playing
        f._autorewind = self._autorewind
        f._frame_idx = self._frame_idx
        f._play_range = self._play_range
        f._frame_ts = self._frame_ts

        f._sps_counter = self._sps_counter
        f._eta = self._eta
        return f

    @property
    def frame_count(self) -> int: return self._frame_count
    @property
    def fps(self) -> float: return self._fps
    @property
    def frame_step(self) -> int: return self._frame_step
    @property
    def is_playing(self) -> bool: return self._playing
    @property
    def is_autorewind(self) -> bool: return self._autorewind
    @property
    def frame_idx(self) -> int: return self._frame_idx
    @property
    def play_range(self) -> Tuple[int, int]: return self._play_range
    @property
    def eta(self) -> float: return self._eta

    def set_fps(self, fps : float) -> Self:
        if self._fps != fps:
            self = self.clone()
            self._fps = fps
        return self

    def set_frame_step(self, frame_step : float) -> Self:
        if self._frame_step != frame_step:
            self = self.clone()
            self._frame_step = max(1, frame_step)
        return self

    def set_playing(self, playing : bool) -> Self:
        if self._playing != playing:
            self = self.clone()
            self._playing = playing
            if playing:
                if self._frame_idx < self._play_range[0] or self._frame_idx >= self._play_range[1]:
                    self._frame_idx = self._play_range[0]

                self._sps_counter = self._sps_counter.reset()
            self._eta = 0

        return self

    def set_autorewind(self, autorewind : bool) -> Self:
        if self._autorewind != autorewind:
            self = self.clone()
            self._autorewind = autorewind
        return self

    def set_frame_idx(self, frame_idx : int) -> Self:
        if self._playing:
            begin_idx, end_idx = self._play_range
        else:
            begin_idx, end_idx = (0, self._frame_count-1)

        if self._autorewind:
            frame_idx = begin_idx + frame_idx % (begin_idx-end_idx)
        else:
            if self._playing and (frame_idx < begin_idx or frame_idx > end_idx):
                self = self.set_playing(False)

            frame_idx = max(begin_idx, min(end_idx, frame_idx) )

        self = self.clone()
        self._frame_idx = frame_idx

        return self

    def set_play_range(self, begin_idx : int = 0, end_idx : int = -1) -> Self:
        self = self.clone()

        if begin_idx < 0: begin_idx += self._frame_count
        if end_idx < 0: end_idx += self._frame_count

        begin_idx = max(0, min(self._frame_count-1, begin_idx))
        end_frame_idx   = max(0, min(self._frame_count-1, end_idx))

        if begin_idx > end_frame_idx:
            begin_idx = end_frame_idx

        if end_frame_idx < begin_idx:
            end_frame_idx = begin_idx

        self._play_range = (begin_idx, end_frame_idx)

        return self

    def step(self) -> Self:
        if self._playing:
            frame_idx = self._frame_idx
            frame_ts  = self._frame_ts

            if self._fps != 0:
                # Fps set. frame_idx by time
                diff_frames = int( (datetime.now().timestamp() - frame_ts) / (1.0/self._fps) )
                if diff_frames != 0:
                    frame_idx += diff_frames
                    frame_ts += diff_frames * (1.0/self._fps)
            else:
                # Fps not set. frame_idx is just step idx
                frame_idx += self._frame_step
                frame_ts = datetime.now().timestamp()

            self = self.set_frame_idx(frame_idx).clone()
            self._frame_ts = frame_ts

            self._sps_counter = self._sps_counter.step()

            sps = self._sps_counter.sps

            _, end_idx = self._play_range

            s_to_end = (end_idx-self._frame_idx) / self._frame_step

            self._eta = (s_to_end / sps) if sps != 0 else 0

        return self