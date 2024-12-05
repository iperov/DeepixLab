from collections import deque
from typing import Callable

from core import ax, mx
from core.lib.collections import FDict

from .FTimeline import FTimeline


class MxTimeline(mx.Disposable):
    def __init__(self,  frame_count : int,
                        on_frame_idx : Callable [[int], ax.Future] = None,
                        state : FDict|None = None):
        """

            on_frame_idx(int) -> ax.Future
                called when frame_idx is emitted.
                When Future is finished, next frame_idx is ready to be emitted as soon as possible.

        """
        super().__init__()
        self._on_frame_idx = on_frame_idx
        state = FDict(state)

        self._main_thread = ax.get_current_thread()
        self._fg = ax.FutureGroup().dispose_with(self)
        self._frame_idx_queue = deque()

        self._mx_frame_count = mx.Number(frame_count-1, mx.Number.Config(min=frame_count-1, max=frame_count-1, read_only=True)).dispose_with(self)
        #self._mx_fps = mx.Number(0, mx.Number.Config(min=0, max=60),
        #                         defer=lambda fps, *_: self.apply_model(self._m.set_fps(fps)) ).dispose_with(self)

        self._mx_frame_step = mx.Number(1, mx.Number.Config(min=1, max=120),
                                        defer=lambda frame_step, *_: self.apply_model(self._m.set_frame_step(frame_step)) ).dispose_with(self)

        self._mx_play_range_begin_idx = mx.Number(0, mx.Number.Config(min=0, max=frame_count-1),
                                                  defer=lambda begin_idx,*_: self.apply_model(self._m.set_play_range(begin_idx, self._mx_play_range_end_idx.get())) ).dispose_with(self)

        self._mx_play_range_end_idx   = mx.Number(frame_count-1, mx.Number.Config(min=0, max=frame_count-1),
                                                  defer=lambda end_idx,*_:   self.apply_model(self._m.set_play_range(self._mx_play_range_begin_idx.get(), end_idx)) ).dispose_with(self)

        self._mx_frame_idx = mx.Number(0, mx.Number.Config(min=0, max=frame_count-1),
                                       defer=lambda frame_idx,*_: self.apply_model(self._m.set_frame_idx(frame_idx), seek=True) ).dispose_with(self)

        self._mx_playing = mx.Flag(False,
                                   defer=lambda playing, *_: self.apply_model(self._m.set_playing(playing))).dispose_with(self)

        self._mx_eta = mx.Property[float](0.0).dispose_with(self)


        self._m = m = FTimeline(frame_count)
        m = (m  .set_autorewind(state.get('autorewind', m.is_autorewind))
                .set_play_range(*state.get('play_range', m.play_range))
                #.set_fps(state.get('fps', m.fps))
                .set_frame_step(state.get('frame_step', m.frame_step))
                .set_frame_idx(state.get('frame_idx', m.frame_idx))
                .set_playing(state.get('playing', m.is_playing))    )

        self.apply_model(m, seek=True)

        self._bg_task()

    def get_state(self) -> FDict:
        m = self._m
        return FDict({  'autorewind' : m.is_autorewind,
                        'play_range' : m.play_range,
                        'fps' : m.fps,
                        'frame_step' : m.frame_step,
                        'frame_idx' : m.frame_idx,
                        'playing' : m.is_playing,   })

    @property
    def mx_frame_count(self) -> mx.INumber_rv: return self._mx_frame_count
    #@property
    #def mx_fps(self) -> mx.INumber_v: return self._mx_fps
    @property
    def mx_frame_step(self) -> mx.INumber_v: return self._mx_frame_step
    @property
    def mx_play_range_begin_idx(self) -> mx.INumber_v: return self._mx_play_range_begin_idx
    @property
    def mx_play_range_end_idx(self) -> mx.INumber_v: return self._mx_play_range_end_idx
    @property
    def mx_frame_idx(self) -> mx.INumber_v: return self._mx_frame_idx
    @property
    def mx_playing(self) -> mx.IFlag_v: return self._mx_playing
    @property
    def mx_eta(self) -> mx.IProperty_rv[float]: return self._mx_eta

    @property
    def model(self) -> FTimeline: return self._m

    def apply_model(self, new_m : FTimeline, seek = False):
        m, self._m = self._m, new_m

        changed_frame_idx  = new_m.frame_idx != m.frame_idx
        changed_playing    = new_m.is_playing != m.is_playing

        # if new_m.fps != m.fps:
        #     self._mx_fps._set(new_m.fps)

        if new_m.eta != m.eta:
            self._mx_eta.set(new_m.eta)

        if new_m.frame_step != m.frame_step:
            self._mx_frame_step._set(new_m.frame_step)

        if changed_frame_idx or seek:
            self._mx_frame_idx._set(new_m.frame_idx)

        if changed_playing:
            self._mx_playing._set(new_m.is_playing)

        if new_m.play_range != m.play_range:
            begin_idx, end_idx = new_m.play_range
            self._mx_play_range_begin_idx._set(begin_idx)
            self._mx_play_range_end_idx._set(end_idx)

        if (changed_playing and new_m.is_playing) or \
            changed_frame_idx or seek:
            if seek:
                self._frame_idx_queue = deque()

            self._frame_idx_queue.append(new_m.frame_idx)

    @ax.task
    def _bg_task(self):
        yield ax.attach_to(self._fg)

        yield ax.sleep(0)

        self._on_frame_idx_fut = self._on_frame_idx(self._m.frame_idx)

        while True:
            if self._on_frame_idx_fut.finished and len(self._frame_idx_queue) == 0:
                self.apply_model(self._m.step())

            while self._on_frame_idx_fut.finished and len(self._frame_idx_queue) != 0:
                self._on_frame_idx_fut = self._on_frame_idx(self._frame_idx_queue.popleft())

            yield ax.sleep(0)





