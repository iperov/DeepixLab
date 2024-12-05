from __future__ import annotations

from typing import Self, overload

from ..lib.math import FVec2f, FVec2i


class FBaseWidget:
    """base code for widget models"""

    def __init__(self):
        self._cli_size = FVec2i(320,240)
        self._mouse_cli_pt = None
        self._mouse_l_down_cli_pt = None
        self._mouse_m_down_cli_pt = None
        self._mouse_r_down_cli_pt = None

    def clone(self) -> Self:
        f = self.__class__.__new__(self.__class__)
        f._cli_size = self._cli_size
        f._mouse_cli_pt = self._mouse_cli_pt
        f._mouse_l_down_cli_pt = self._mouse_l_down_cli_pt
        f._mouse_m_down_cli_pt = self._mouse_m_down_cli_pt
        f._mouse_r_down_cli_pt = self._mouse_r_down_cli_pt
        return f    

    @property
    def cli_size(self) -> FVec2i: return self._cli_size
    @overload
    def set_cli_size(self, width : int|float, height : int|float) -> Self: ...
    @overload
    def set_cli_size(self, cli_size : FVec2i) -> Self: ...
    def set_cli_size(self, *args) -> Self:
        if len(args) == 1:
            cli_size, = args
        elif len(args) == 2:
            cli_size = FVec2i(args[0], args[1])

        if self._cli_size != cli_size:
            self = self.clone()
            self._cli_size = cli_size
        return self

    @property
    def mouse_cli_pt(self) -> FVec2f|None:
        """mouse coord in client space"""
        return self._mouse_cli_pt
    @property
    def mouse_l_down_cli_pt(self) -> FVec2f|None:
        """mouse lbtn down coord in client space"""
        return self._mouse_l_down_cli_pt
    @property
    def mouse_m_down_cli_pt(self) -> FVec2f|None:
        """mouse mbtn down coord in client space"""
        return self._mouse_m_down_cli_pt
    @property
    def mouse_r_down_cli_pt(self) -> FVec2f|None:
        """mouse rbtn down coord in client space"""
        return self._mouse_r_down_cli_pt

    def mouse_leave(self) -> Self:
        if self._mouse_cli_pt is not None:
            self = self.clone()
            self._mouse_cli_pt = None
            self._mouse_l_down_cli_pt = None
            self._mouse_m_down_cli_pt = None
            self._mouse_r_down_cli_pt = None
        return self

    def mouse_move(self, cli_pt : FVec2f, ctrl_pressed = False, shift_pressed = False) -> Self:
        self = self.clone()
        self._mouse_cli_pt = cli_pt
        return self

    def mouse_lbtn_down(self, cli_pt : FVec2f, ctrl_pressed = False, shift_pressed = False) -> Self:
        if self._mouse_l_down_cli_pt is None:
            self = self.mouse_move(cli_pt, ctrl_pressed=ctrl_pressed, shift_pressed=shift_pressed).clone()
            self._mouse_l_down_cli_pt = cli_pt
        return self

    def mouse_lbtn_up(self, cli_pt : FVec2f, ctrl_pressed = False, shift_pressed = False) -> Self:
        if self._mouse_l_down_cli_pt is not None:
            self = self.mouse_move(cli_pt, ctrl_pressed=ctrl_pressed, shift_pressed=shift_pressed).clone()
            self._mouse_l_down_cli_pt = None
        return self

    def mouse_mbtn_down(self, cli_pt : FVec2f, ctrl_pressed = False, shift_pressed = False) -> Self:
        if self._mouse_m_down_cli_pt is None:
            self = self.mouse_move(cli_pt, ctrl_pressed=ctrl_pressed, shift_pressed=shift_pressed).clone()
            self._mouse_m_down_cli_pt = cli_pt
        return self

    def mouse_mbtn_up(self, cli_pt : FVec2f, ctrl_pressed = False, shift_pressed = False) -> Self:
        if self._mouse_m_down_cli_pt is not None:
            self = self.mouse_move(cli_pt, ctrl_pressed=ctrl_pressed, shift_pressed=shift_pressed).clone()
            self._mouse_m_down_cli_pt = None
        return self

    def mouse_rbtn_down(self, cli_pt : FVec2f, ctrl_pressed = False, shift_pressed = False) -> Self:
        if self._mouse_r_down_cli_pt is None:
            self = self.mouse_move(cli_pt, ctrl_pressed=ctrl_pressed, shift_pressed=shift_pressed).clone()
            self._mouse_r_down_cli_pt = cli_pt
        return self

    def mouse_rbtn_up(self, cli_pt : FVec2f, ctrl_pressed = False, shift_pressed = False) -> Self:
        if self._mouse_r_down_cli_pt is not None:
            self = self.mouse_move(cli_pt, ctrl_pressed=ctrl_pressed, shift_pressed=shift_pressed).clone()
            self._mouse_r_down_cli_pt = None
        return self

    def mouse_wheel(self, cli_pt : FVec2f, delta : float, ctrl_pressed = False, shift_pressed = False) -> Self:
        #self = self.clone()
        #self._mouse_cli_pt = cli_pt

        self = self.mouse_move(cli_pt, ctrl_pressed=ctrl_pressed, shift_pressed=shift_pressed).clone()
        return self

    def update(self, time_delta : float) -> Self:
        return self