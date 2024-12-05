from __future__ import annotations

from functools import cached_property
from typing import Self

from core.lib.math import FVec2f
from core.qx.FBaseWidget import FBaseWidget

from ..lib.math import FAffMat2, FVec2f
from .FBaseWidget import FBaseWidget


class FBaseWorldViewWidget(FBaseWidget): 
    """base code for world projection widget models"""
    
    def __init__(self):
        super().__init__()
        self._mouse_l_down_world_pt = None
        self._mouse_m_down_world_pt = None
        self._mouse_r_down_world_pt = None
        
    def clone(self) -> Self:
        f = super().clone()
        f._mouse_l_down_world_pt = self._mouse_l_down_world_pt
        f._mouse_m_down_world_pt = self._mouse_m_down_world_pt
        f._mouse_r_down_world_pt = self._mouse_r_down_world_pt
        return f
        
    @cached_property
    def mouse_world_pt(self) -> FVec2f|None: 
        """mouse position in world space"""
        if self.mouse_cli_pt is not None:
            return self.cli2w_mat.map([self._mouse_cli_pt])[0]
        return None
    
    @property
    def mouse_l_down_world_pt(self) -> FVec2f|None: 
        """mouse down position in world space when click happened"""
        return self._mouse_l_down_world_pt
    
    @property
    def mouse_m_down_world_pt(self) -> FVec2f|None: 
        """mouse down position in world space when click happened"""
        return self._mouse_m_down_world_pt
        
    @property
    def mouse_r_down_world_pt(self) -> FVec2f|None: 
        """mouse down position in world space when click happened"""
        return self._mouse_r_down_world_pt
    
    @property
    def w2cli_mat(self) -> FAffMat2:
        """
        mat to transform world to cli
        """
        # Typically your own or for example from FWorldProj
        raise NotImplementedError()

    @property    
    def cli2w_mat(self) -> FAffMat2: return self.w2cli_mat.inverted
        
    def mouse_lbtn_down(self, cli_pt: FVec2f, ctrl_pressed=False, shift_pressed=False) -> Self:
        new_self = super().mouse_lbtn_down(cli_pt, ctrl_pressed, shift_pressed)
        if not (new_self is self):
            new_self = new_self.clone()
            new_self._mouse_l_down_world_pt = self.cli2w_mat.map(new_self.mouse_l_down_cli_pt)
        return new_self
        
    def mouse_lbtn_up(self, cli_pt: FVec2f, ctrl_pressed=False, shift_pressed=False) -> Self:
        new_self = super().mouse_lbtn_up(cli_pt, ctrl_pressed=ctrl_pressed, shift_pressed=shift_pressed)
        if not (new_self is self):
            new_self = new_self.clone()
            new_self._mouse_l_down_world_pt = None
        return new_self
    
    def mouse_mbtn_down(self, cli_pt: FVec2f, ctrl_pressed=False, shift_pressed=False) -> Self:
        new_self = super().mouse_mbtn_down(cli_pt, ctrl_pressed, shift_pressed)
        if not (new_self is self):
            new_self = new_self.clone()
            new_self._mouse_m_down_world_pt = self.cli2w_mat.map(new_self.mouse_m_down_cli_pt)
        return new_self
        
    def mouse_mbtn_up(self, cli_pt: FVec2f, ctrl_pressed=False, shift_pressed=False) -> Self:
        new_self = super().mouse_mbtn_up(cli_pt, ctrl_pressed=ctrl_pressed, shift_pressed=shift_pressed)
        if not (new_self is self):
            new_self = new_self.clone()
            new_self._mouse_m_down_world_pt = None
        return new_self
    
    def mouse_rbtn_down(self, cli_pt: FVec2f, ctrl_pressed=False, shift_pressed=False) -> Self:
        new_self = super().mouse_rbtn_down(cli_pt, ctrl_pressed, shift_pressed)
        if not (new_self is self):
            new_self = new_self.clone()
            new_self._mouse_r_down_world_pt = self.w2cli_mat.inverted.map(new_self.mouse_r_down_cli_pt)
        return new_self
        
    def mouse_rbtn_up(self, cli_pt: FVec2f, ctrl_pressed=False, shift_pressed=False) -> Self:
        new_self = super().mouse_rbtn_up(cli_pt, ctrl_pressed=ctrl_pressed, shift_pressed=shift_pressed)
        if not (new_self is self):
            new_self = new_self.clone()
            new_self._mouse_r_down_world_pt = None
        return new_self