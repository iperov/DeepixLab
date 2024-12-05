import time
from collections import deque
from datetime import datetime
from typing import Self

import numpy as np


class Stopwatch:
    def __init__(self):
        self.reset()

    def reset(self):
        self._time = time.perf_counter()

    def elapsed(self):
        return time.perf_counter()-self._time


class timeit:
    def __init__(self, msg : str = None):
        self._msg = msg if msg is not None else ''

    def __enter__(self):
        self.t = time.perf_counter()
    def __exit__(self, a,b,c):
        print(f'Time of {self._msg}: {time.perf_counter()-self.t}')

class measure:
    def __init__(self):
        self._t = time.perf_counter()

    def elapsed(self):
        return time.perf_counter()-self._t

class AverageMeasurer:
    def __init__(self, samples=120):
        self._samples = samples
        self._ts = None

        self._measurements = [0]*samples
        self._n_sample = 0

    def start(self):
        self._ts = datetime.now().timestamp()

    def discard(self):
        self._ts = None

    def stop(self) -> float:
        """
        stop measure and return current average time sec
        """
        measurements = self._measurements
        if self._ts is None:
            raise Exception('AverageMeasurer was not started')

        t = datetime.now().timestamp() - self._ts
        measurements[self._n_sample] = t
        self._n_sample = (self._n_sample + 1) % self._samples
        self._ts = None
        return max(0.001, np.mean(measurements))

class FPSCounter:
    def __init__(self, samples=120):
        self._steps = deque(maxlen=samples)

    def step(self) -> float:
        """
        make a step for FPSCounter and returns current FPS
        """
        steps = self._steps
        steps.append(datetime.now().timestamp()  )
        if len(steps) >= 2:
            div = steps[-1] - steps[0]
            if div != 0:
                return (len(steps)-1) / div
        return 0.0




class FSPSCounter:
    """F-model step per second counter"""
    def __init__(self):
        self._first_step_time = None
        self._last_step_time = None
        self._step_count = 0

    def clone(self) -> Self:
        f = self.__class__.__new__(self.__class__)
        f._first_step_time = self._first_step_time
        f._last_step_time = self._last_step_time
        f._step_count = self._step_count
        return f

    def reset(self) -> Self:
        return FSPSCounter()

    @property
    def sps(self) -> float:
        """step per sec"""
        time_diff = self._last_step_time - self._first_step_time
        if time_diff == 0:
            return 0
        return self._step_count / time_diff

    def step(self, amount=1) -> Self:
        """Count a step"""
        self = self.clone()

        time = datetime.now().timestamp()
        if self._first_step_time is None:
            self._first_step_time = time

        self._last_step_time = time
        self._step_count += amount

        return self

class SPSCounter:
    """Step per second counter"""
    def __init__(self):
        self.reset()

    def reset(self):
        self._first_step_time = None
        self._last_step_time = None
        self._step_count = 0

    def step(self, amount=1) -> float:
        """Count a step. Returns current steps per second"""
        time = datetime.now().timestamp()
        if self._first_step_time is None:
            self._first_step_time = time

        self._last_step_time = time
        self._step_count += amount

        time_diff = self._last_step_time - self._first_step_time
        if time_diff == 0:
            return 0

        return self._step_count / time_diff

