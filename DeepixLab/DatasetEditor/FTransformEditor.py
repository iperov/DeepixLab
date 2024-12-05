from __future__ import annotations

import math
from enum import Enum, auto
from functools import cached_property
from typing import Self

from core import qx
from core.lib.functools import cached_method
from core.lib.math import (FAffMat2, FBBoxf, FCirclef, FRectf, FVec2f, FVec2i,
                           FWorldProj2D)


class FTransformEditor(qx.FBaseWorldViewWidget):
    """transform editor model"""

    class ViewType(Enum):
        FREE = auto()
        CENTER_FIT = auto()

    class FDrag:
        def __init__(self, mouse_cli_pt : FVec2f, mouse_world_pt : FVec2f):
            self._mouse_cli_pt : FVec2f   = mouse_cli_pt
            self._mouse_world_pt : FVec2f = mouse_world_pt

        @property
        def mouse_cli_pt(self) -> FVec2f: return self._mouse_cli_pt
        @property
        def mouse_world_pt(self) -> FVec2f: return self._mouse_world_pt

        def clone(self):
            f = self.__class__.__new__(self.__class__)
            f._mouse_cli_pt = self._mouse_cli_pt
            f._mouse_world_pt = self._mouse_world_pt
            return f

    class FDragViewProj(FDrag):
        def __init__(self,  mouse_cli_pt : FVec2f, mouse_world_pt : FVec2f,
                            view_proj : FWorldProj2D,
                    ):
            super().__init__(mouse_cli_pt=mouse_cli_pt, mouse_world_pt=mouse_world_pt)
            self._view_proj = view_proj

        @property
        def view_proj(self) -> FWorldProj2D: return self._view_proj

        def clone(self):
            f = super().clone()
            f._view_proj = self._view_proj
            return f

    class FDragImagePos(FDrag):
        def __init__(self,  mouse_cli_pt : FVec2f, mouse_world_pt : FVec2f,
                            m2w_img_mat : FAffMat2  ):
            super().__init__(mouse_cli_pt=mouse_cli_pt, mouse_world_pt=mouse_world_pt)
            self._m2w_img_mat = m2w_img_mat
        @property
        def m2w_img_mat(self) -> FAffMat2: return self._m2w_img_mat
        def clone(self):
            f = super().clone()
            f._m2w_img_mat = self._m2w_img_mat
            return f

    class FDragImageScaleRotation(FDrag):
        def __init__(self,  mouse_cli_pt : FVec2f, mouse_world_pt : FVec2f,
                            m2w_img_mat : FAffMat2  ):
            super().__init__(mouse_cli_pt=mouse_cli_pt, mouse_world_pt=mouse_world_pt)
            self._m2w_img_mat = m2w_img_mat
        @property
        def m2w_img_mat(self) -> FAffMat2: return self._m2w_img_mat
        def clone(self):
            f = super().clone()
            f._m2w_img_mat = self._m2w_img_mat
            return f


    def __init__(self, W : int, H : int):
        super().__init__()
        self._view_type = self.ViewType.CENTER_FIT
        self._view_proj = FWorldProj2D().set_vp_size(self.cli_size)
        self._img_size = FVec2i(W, H)
        self._m2w_img_mat = FAffMat2()

        self._drag : FTransformEditor.FDrag|None = None

    def clone(self) -> FTransformEditor:
        f = super().clone()
        f._view_type = self._view_type
        f._view_proj = self._view_proj
        f._img_size  = self._img_size
        f._m2w_img_mat  = self._m2w_img_mat
        f._drag  = self._drag
        return f

    # FBaseWidget
    def set_cli_size(old_self, *args) -> Self:
        self = super().set_cli_size(*args)
        if not (self is old_self):
            self = self.clone()
            self._view_proj = self.view_proj.set_vp_size(self.cli_size)
        return self

    # FBaseWorldViewWidget
    @property
    def w2cli_mat(self) -> FAffMat2: return self.view_proj.w2vp_mat

    def mouse_move(self, cli_pt : FVec2f, ctrl_pressed=False, shift_pressed=False) -> Self:
        self = super().mouse_move(cli_pt, ctrl_pressed=ctrl_pressed, shift_pressed=shift_pressed)

        if (mouse_cli_pt := self.mouse_cli_pt) is not None:
            if isinstance(drag := self._drag, self.FDragViewProj):
                diff = drag.mouse_cli_pt - mouse_cli_pt
                view_proj = drag.view_proj
                self._view_proj = view_proj.set_w_view_pos(view_proj.vp2w_mat.map(diff+view_proj.w2vp_mat.map(view_proj.w_view_pos)))

            elif isinstance(drag, self.FDragImagePos):
                if (mouse_world_pt := self.mouse_world_pt) is not None:
                    self = self.clone()
                    self._m2w_img_mat = drag.m2w_img_mat * FAffMat2().translate(mouse_world_pt - drag.mouse_world_pt)

            elif isinstance(drag , self.FDragImageScaleRotation):
                img_pc = FVec2f(self._img_size)/2
                w_img_pc = drag.m2w_img_mat.map(img_pc)

                cli_img_cp = self.w2cli_mat.map(w_img_pc)

                vp_diff = mouse_cli_pt - cli_img_cp
                vp_drag_diff = drag.mouse_cli_pt - cli_img_cp

                rot_diff = vp_diff.atan2() - vp_drag_diff.atan2()
                scale_diff = vp_diff.length / vp_drag_diff.length

                self = self.clone()
                self._m2w_img_mat = drag.m2w_img_mat * FAffMat2().translate(-w_img_pc).rotate(rot_diff).scale(scale_diff).translate(w_img_pc)

        return self


    def mouse_lbtn_down(self, cli_pt: FVec2f, ctrl_pressed=False, shift_pressed=False) -> Self:
        self = super().mouse_lbtn_down(cli_pt, ctrl_pressed=ctrl_pressed, shift_pressed=shift_pressed)

        if (mouse_cli_pt := self.mouse_cli_pt) is not None and \
           (mouse_world_pt := self.mouse_world_pt) is not None:

            if self._drag is None:
                if self.is_hovering_ctrl_circle_inside():
                    self = self .set_view_type(self.ViewType.FREE) \
                                .set_drag(self.FDragImagePos(mouse_cli_pt=mouse_cli_pt, mouse_world_pt=mouse_world_pt, m2w_img_mat=self._m2w_img_mat))

                # Clicking on ellipse edge
                if self.is_hovering_ctrl_circle_edge():
                    self = self .set_view_type(self.ViewType.FREE) \
                                .set_drag(self.FDragImageScaleRotation(mouse_cli_pt=mouse_cli_pt, mouse_world_pt=mouse_world_pt, m2w_img_mat=self._m2w_img_mat))
        return self

    def mouse_lbtn_up(self, cli_pt: FVec2f, ctrl_pressed=False, shift_pressed=False) -> Self:
        self = super().mouse_lbtn_up(cli_pt, ctrl_pressed=ctrl_pressed, shift_pressed=shift_pressed)
        if self._drag is not None:
            self = self.set_drag(None)
        return self

    def mouse_mbtn_down(self, cli_pt: FVec2f, ctrl_pressed=False, shift_pressed=False) -> Self:
        self = super().mouse_mbtn_down(cli_pt, ctrl_pressed=ctrl_pressed, shift_pressed=shift_pressed)
        if self._drag is None:
            self = self .set_view_type(self.ViewType.FREE) \
                        .set_drag(self.FDragViewProj(mouse_cli_pt=self.mouse_cli_pt, mouse_world_pt=self.mouse_world_pt, view_proj=self.view_proj))
        return self

    def mouse_mbtn_up(self, cli_pt: FVec2f, ctrl_pressed=False, shift_pressed=False) -> Self:
        self = super().mouse_mbtn_up(cli_pt, ctrl_pressed=ctrl_pressed, shift_pressed=shift_pressed)
        if isinstance(self._drag, self.FDragViewProj):
            self = self.set_drag(None)
        return self

    def mouse_wheel(self, cli_pt : FVec2f, delta : float, ctrl_pressed = False, shift_pressed = False) -> Self:
        self = super().mouse_wheel(cli_pt, delta, ctrl_pressed=ctrl_pressed, shift_pressed=shift_pressed)

        if self._drag is None and \
           (mouse_cli_pt := self.mouse_cli_pt) is not None:
            self = self.set_view_type(self.ViewType.FREE).clone()
            self._view_proj = self.view_proj.scale_at(mouse_cli_pt, (1/1.15) if delta > 0 else 1.15)

        return self

    # FTransformEditor

    @property
    def view_type(self) -> ViewType: return self._view_type
    def set_view_type(self, view_type : ViewType) -> Self:
        if self._view_type != view_type:
            self = self.clone()
            self._view_proj = self.view_proj
            self._view_type = view_type
        return self



    @property
    def view_proj(self) -> FWorldProj2D:
        view_proj = self._view_proj
        if self._view_type == self.ViewType.CENTER_FIT:
            view_proj = view_proj.center_fit(FBBoxf(FVec2f(0,0), FVec2f(self._img_size)), coverage=1.5)
        return view_proj
    def set_view_proj(self, view_proj : ViewType) -> Self:
        if self._view_proj != view_proj:
            self = self.set_view_type(self.ViewType.FREE).clone()
            self._view_proj = view_proj
        return self

    @property
    def drag(self) -> FDrag|None: return self._drag
    def set_drag(self, drag : FDrag|None):
        if not (self._drag is drag):
            self = self.clone()
            self._drag = drag
        return self

    @property
    def img_size(self) -> FVec2i:
        return self._img_size

    @property
    def result_uni_mat(self) -> FAffMat2:
        """uniform mat to transform w_stencil_rect to w_img_rect"""
        return FAffMat2.estimate(self.w_stencil_rect, self.w_img_rect).scale_space(1/FVec2f(self.img_size))

    @cached_property
    def w_img_rect(self) -> FRectf:
        """image FRectf in world space"""
        return FRectf(self._img_size).transform(self._m2w_img_mat)
    @cached_property
    def cli_img_rect(self) -> FRectf:
        """image FRectf in cli space"""
        return self.w_img_rect.transform(self.w2cli_mat)

    @cached_property
    def w_stencil_rect(self) -> FRectf:
        """stencil rect in world space"""
        return FRectf(self._img_size)
    @cached_property
    def cli_stencil_rect(self) -> FRectf:
        """stencil rect in cli space"""
        return self.w_stencil_rect.transform(self.w2cli_mat)
    @cached_property
    def cli_ctrl_circle(self) -> FCirclef:
        """ctrl FCirclef in cli space"""
        w_img_rect = self.w_img_rect
        return FCirclef(w_img_rect.pc, math.sqrt(2)*max(w_img_rect.width, w_img_rect.height)/2 ).transform(self.w2cli_mat)
    @cached_property
    def cli_ctrl_center_pt(self) -> FVec2f:
        """ctrl center FVec2f point in vp space"""
        return self.w2cli_mat.map(self.w_img_rect.pc)

    @cached_method
    def is_mouse_on_ctrl_circle_edge(self) -> bool:
        if (mouse_cli_pt := self.mouse_cli_pt) is not None:
            return self.cli_ctrl_circle.is_point_on_edge(mouse_cli_pt, 0, 20)
        return False
    @cached_method
    def is_mouse_on_ctrl_circle_inside(self) -> bool:
        if (mouse_cli_pt := self.mouse_cli_pt) is not None:
            return self.cli_ctrl_circle.is_point_inside(mouse_cli_pt)
        return False

    def is_hovering_ctrl_circle_edge(self) -> bool: return self.is_mouse_on_ctrl_circle_edge()
    def is_hovering_ctrl_circle_inside(self) -> bool: return not self.is_hovering_ctrl_circle_edge() and self.is_mouse_on_ctrl_circle_inside()



    def recover_from_undo_redo(self, current : FTransformEditor) -> Self:
        self = self.set_cli_size(current.cli_size)
        self = self.set_view_type(current.view_type).clone()
        self._view_proj = current.view_proj
        if (mouse_cli_pt := current.mouse_cli_pt) is not None:
            self = self.mouse_move(mouse_cli_pt)
        return self



    def is_changed_for_undo(self, other : FTransformEditor) -> bool:
        if isinstance(drag := self.drag, (self.FDragImagePos, self.FDragImageScaleRotation)):
            return type(drag) != type(other.drag)
        else:
            return self._m2w_img_mat != other._m2w_img_mat



