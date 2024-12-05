from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Self, Sequence, Tuple

import numpy as np

from core import qx
from core.lib import cc
from core.lib.image import FImage
from core.lib.math import FVec3f


class FGraph(qx.FBaseWidget):
    @dataclass(frozen=True)
    class Data:
        array : np.ndarray     # shape (N, C)
        names : Sequence[str]  # name for every N
        colors : Sequence[ FVec3f ]
        N_start : int = 0 # real N start from array

        def clip_u(self, start : float, end : float) -> Self:
            """clip by uniform start -> end"""
            array = self.array
            N = array.shape[0]
            N_start, N_end = int(N*start), int(N*end)
            return self.clip(N_start, N_end)

        def clip(self, N_start : int, N_end : int) -> Self:
            """clip by [N_start..N_end) """
            array = self.array[N_start:N_end, :]
            return FGraph.Data(array, self.names, self.colors, N_start=self.N_start+N_start)


    def __init__(self):
        super().__init__()
        self._data_view_range : Tuple[float,float] = (0,1)
        self._data = FGraph.Data(np.zeros( (0,0), np.float32 ), (), ())
        self._cli_image = FImage.zeros( self.cli_size.y, self.cli_size.x, 3)
        self._selected_names = ()

    def clone(self) -> Self:
        f = super().clone()
        f._data_view_range = self._data_view_range
        f._data = self._data
        f._cli_image = self._cli_image
        f._selected_names = self._selected_names
        return f

    def set_cli_size(self, *args) -> Self:
        self = super().set_cli_size(*args)
        self = self.update_cli_image()
        return self


    @cached_property
    def cli_selection_ab(self) -> Tuple[int, int]|None:
        """selection in cli X coords limited by [0, cli_size.x]"""
        if (mouse_cli_pt := self.mouse_cli_pt) is not None:
            sel_start = sel_end = mouse_cli_pt.x
            if (mouse_l_down_cli_pt := self.mouse_l_down_cli_pt) is not None:
                sel_end = mouse_l_down_cli_pt.x
            if sel_end < sel_start:
                sel_start, sel_end = sel_end, sel_start
            W = self.cli_size.x
            return ( int(max(0, min(sel_start, W))),
                     int(max(0, min(sel_end, W))) )
        return None

    @property
    def data_view_range(self) -> Tuple[float,float]: return self._data_view_range
    def set_data_view_range(self, data_view_range : Tuple[float,float]) -> Self:
        self = self.clone()
        a, b = data_view_range
        b = max(0, min(b, 1))
        a = max(0, min(a, b, 1))
        self._data_view_range = (a,b)
        self = self.update_cli_image()
        return self

    @property
    def names(self) -> Sequence[str]:
        return self._data.names

    @property
    def selected_names(self) -> Sequence[str]: return self._selected_names
    def set_selected_names(self, selected_names : Sequence[str]) -> Self:
        self = self.clone()
        self._selected_names = selected_names
        self = self.update_cli_image()
        return self

    @cached_property
    def data(self) -> Data:
        """data limited by selected names"""
        data = self._data
        selected_names = self._selected_names

        idxs = [i for i, name in enumerate(data.names) if name in selected_names ]

        data = FGraph.Data( array=data.array[:, idxs],
                            names=[data.names[i] for i in idxs],
                            colors=[data.colors[i] for i in idxs], )

        return data

    def set_data(self, data) -> Self:
        self = self.clone()
        self._data = data
        self = self.update_cli_image()
        return self

    @cached_property
    def view_data(self) -> Data:
        """.data limited by data_view_range"""
        return self.data.clip_u(self._data_view_range[0], self._data_view_range[1])

    @cached_property
    def selected_view_data(self) -> Data|None:
        """.view_data limited by cli_selection_ab"""
        if (cli_selection_ab := self.cli_selection_ab) is not None:
            A, B = cli_selection_ab
            W = self.cli_size.x
            view_data = self.view_data
            N = view_data.array.shape[0]
            if N <= W:
                return view_data.clip(A, B+1)
            else:
                return view_data.clip_u(A/W, (B+1)/W)
        return None

    @property
    def cli_image(self) -> FImage:
        return self._cli_image

    def update_cli_image(self) -> Self:
        W, H = self.cli_size

        if W != 0 and H != 0:
            img = _draw_bg(W,H)

            view_data = self.view_data
            array = view_data.array
            colors = view_data.colors
            N, C = array.shape
            if C > 0:
                img_graph, g_min, g_max = _preprocess_data(W, array)
                _overlay_graph(img, img_graph, g_min, g_max, colors=np.float32(colors) )

            self = self.clone()
            self._cli_image = FImage.from_numpy(img)
        return self

lib_path = cc.get_lib_path(Path(__file__).parent, Path(__file__).stem)

@cc.lib_import(lib_path)
def c_draw_bg(img : cc.c_float32_p, H : cc.c_int32, W : cc.c_int32) -> cc.c_void: ...
def _draw_bg(W, H) -> np.ndarray:
    img = np.empty( (H,W,4), np.float32)
    c_draw_bg(img.ctypes.data_as(cc.c_float32_p), H, W)
    return img

@cc.lib_import(lib_path)
def c_preprocess_data(graph : cc.c_float32_p, data : cc.c_float32_p, N : cc.c_int32, W : cc.c_int32, C : cc.c_int32, out_g_min : cc.c_float32_p, out_g_max : cc.c_float32_p) -> cc.c_void: ...
def _preprocess_data(W, graph : np.ndarray):
    graph = np.ascontiguousarray(graph)
    N, C = graph.shape
    image_data = np.zeros( (W,C,3), np.float32 )
    g_min = cc.c_float32()
    g_max = cc.c_float32()
    c_preprocess_data(graph.ctypes.data_as(cc.c_float32_p), image_data.ctypes.data_as(cc.c_float32_p), N, W, C, g_min, g_max )

    return image_data, g_min.value, g_max.value

@cc.lib_import(lib_path)
def c_overlay_graph(img : cc.c_float32_p, graph : cc.c_float32_p, C : cc.c_int32, H : cc.c_int32, W : cc.c_int32, g_min : cc.c_float32, g_max : cc.c_float32, colors : cc.c_float32_p) -> cc.c_void: ...
def _overlay_graph(img : np.ndarray, img_graph : np.ndarray, g_min, g_max, colors : np.ndarray):
    img = np.ascontiguousarray(img)
    img_graph = np.ascontiguousarray(img_graph)

    H,W,_ = img.shape
    _, C, _ = img_graph.shape

    c_overlay_graph(img.ctypes.data_as(cc.c_float32_p), img_graph.ctypes.data_as(cc.c_float32_p), C, H, W, g_min, g_max, colors.ctypes.data_as(cc.c_float32_p))


def setup_compile():
    return cc.Compiler(Path(__file__).parent).shared_library(lib_path).minimal_c().optimize().include_glsl().include('FGraph.cpp').compile()
