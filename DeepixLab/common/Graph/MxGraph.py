from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

from core import ax, mx
from core.lib.collections import FDict


class MxGraph(mx.Disposable):
    @dataclass(frozen=True)
    class Data:
        length : int          # real N
        buffer : np.ndarray   # shape (N+, C)
        names : Sequence[str] # name for every C

    def __init__(self,  state : FDict = None):
        super().__init__()
        state = FDict(state)

        self._main_thread = ax.get_current_thread()
        self._graph_thread = ax.Thread().dispose_with(self)
        self._fg = ax.FutureGroup().dispose_with(self)

        ###
        length = state.get('length', 0)
        names = state.get('names', ())
        if (buffer := state.get('buffer', None)) is not None:
            buffer = np.array(buffer, np.float32) # to writable
        else:
            buffer = np.zeros((length, len(names)), np.float32)


        self._graph_data = MxGraph.Data(length=length, buffer=buffer, names=names,)
        ### ^ operated only in _graph_thread

        self._mx_data = mx.Property[MxGraph.Data](self._graph_data).dispose_with(self)

    def get_state(self) -> FDict:
        data = self._graph_data
        return FDict({  'length' : data.length,
                        'buffer' : data.buffer,
                        'names' : data.names  })

    @property
    def mx_data(self) -> mx.IProperty_rv[Data]: return self._mx_data

    @ax.task
    def add(self, values : Dict[str, float]):
        """
        Task to add values at the end of graph.

        if graph of name does not exist it will be created with zero values.

        if name exists and value not specified in argument, it will not be set.
        """
        yield ax.attach_to(self._fg)
        yield ax.switch_to(self._graph_thread)

        data = self._graph_data

        length = data.length
        buffer = data.buffer
        names  = data.names

        N, C = buffer.shape

        if length >= N:
            # Requested length more than avail capacity
            # Expand capacity
            new_buffer = np.empty( (length+256*1024, C), np.float32 )
            new_buffer[:N, :] = buffer
            new_buffer[N:, :] = 0.0

            buffer = new_buffer
            N, C = buffer.shape

        # Expand names and buffer if new names arrived
        C_diff = 0
        for name in values.keys():
            if name not in names:
                names = names + (name,)
                C_diff += 1

        if C_diff != 0:
            buffer = np.concatenate([buffer,
                                    np.zeros( (N, C_diff), np.float32) ], 1)
            N, C = buffer.shape


        buffer[length] = [ values.get(name, 0) for name in names ]
        length += 1

        data = self._graph_data = MxGraph.Data(length=length, buffer=buffer, names=names)

        yield ax.switch_to(self._main_thread)

        self._mx_data.set(data)


    @ax.task
    def trim(self, f_start : float, f_end : float):
        """
        Task to trim the graph

            f_start [0..1] inclusive

            f_end   [0..1] inclusive
        """
        yield ax.attach_to(self._fg)
        yield ax.switch_to(self._graph_thread)

        data = self._graph_data

        length = data.length
        buffer = data.buffer
        names  = data.names

        f_end   = max(0.0, min(f_end, 1.0))
        f_start = max(0.0, min(f_start, f_end, 1.0))

        start_length = int(f_start * length)
        end_length   = int(f_end * length)


        length = end_length - start_length
        buffer = buffer[start_length:end_length, :]

        valid_n = [ n for n in range(buffer.shape[1]) if np.any(buffer[:, n] != 0) ]
        buffer = buffer[:, valid_n]
        names = tuple( x for n, x in enumerate(names) if n in valid_n )

        data = self._graph_data = MxGraph.Data(length=length, buffer=buffer, names=names)

        yield ax.switch_to(self._main_thread)

        self._mx_data.set(data)


