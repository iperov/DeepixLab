from __future__ import annotations

import weakref
from enum import Enum, auto
from functools import cached_property
from typing import Self, Tuple

import cv2
import numpy as np

from core import qx
from core.lib.functools import cached_method
from core.lib.image import FImage
from core.lib.math import (FAffMat2, FBBoxf, FPoly2f, FVec2f, FVec2i,
                           FWorldProj2D)


class FMaskEditor(qx.FBaseWorldViewWidget):

    class ViewType(Enum):
        FREE = auto()
        CENTER_FIT = auto()

    class PolyApplyType(Enum):
        INCLUDE = auto()
        EXCLUDE = auto()

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

    class FDragStatePolyMovePt(FDrag):
        def __init__(self,  mouse_cli_pt : FVec2f, mouse_world_pt : FVec2f,
                            state_poly_pt_id : int, state_poly_pt : FVec2f,
                            perp_constraint : bool,
                        ):
            super().__init__(mouse_cli_pt=mouse_cli_pt, mouse_world_pt=mouse_world_pt)
            self._state_poly_pt_id : int = state_poly_pt_id
            self._state_poly_pt : FVec2f = state_poly_pt
            self._perp_constraint = perp_constraint

        @property
        def state_poly_pt_id(self) -> int: return self._state_poly_pt_id
        @property
        def state_poly_pt(self) -> FVec2f: return self._state_poly_pt
        @property
        def perp_constraint(self) -> bool: return self._perp_constraint

        def clone(self):
            f = super().clone()
            f._state_poly_pt_id = self._state_poly_pt_id
            f._state_poly_pt = self._state_poly_pt
            return f


    class FState:
        def __init__(self):
            super().__init__()
            self._host_ref : weakref.ReferenceType[FMaskEditor]  = None

        def clone(self) -> Self:
            f = self.__class__.__new__(self.__class__)
            f._host_ref = None
            return f

        def set_host(self, host) -> Self:
            self = self.clone()
            self._host_ref = weakref.ref(host)
            return self

    class FStatePoly(FState):
        def __init__(self, poly : FPoly2f):
            super().__init__()
            self._poly = poly

        def clone(self):
            f = super().clone()
            f._poly = self._poly
            return f

        @property
        def poly(self) -> FPoly2f: return self._poly
        def set_poly(self, poly : FPoly2f) -> Self:
            self = self.clone()
            self._poly = poly
            return self

        @cached_property
        def cli_poly(self) -> FPoly2f:
            return self._poly.transform(self._host_ref().w2cli_mat)

        @cached_method
        def is_mouse_at_poly(self) -> bool:
            host = self._host_ref()
            if (mouse_cli_pt := host.mouse_cli_pt) is not None and \
               (dist := self.cli_poly.dist(mouse_cli_pt)) is not None:
                return dist <= host.pt_select_radius
            return False

        @cached_property
        def poly_pt_id_at_mouse(self) -> int|None:
            host = self._host_ref()

            if (mouse_cli_pt := host.mouse_cli_pt) is not None:
                x = [ (i, l) for i, poly_cli_pt in enumerate(self.cli_poly.points)
                        if (l := (mouse_cli_pt-poly_cli_pt).length) <= host.pt_select_radius ]
                x = sorted(x, key=lambda x: x[1])
                return None if len(x) == 0 else x[0][0]
            return None

        @cached_property
        def poly_edge_id_pt_at_mouse(self) -> Tuple[int, FVec2f, FVec2f] | None:
            host = self._host_ref()

            if (mouse_world_pt := host.mouse_world_pt) is not None and \
               (mouse_cli_pt := host.mouse_cli_pt) is not None:

                if (edge_id_pt := self.poly.nearest_edge_id_pt(mouse_world_pt)) is not None:
                    edge_id, pt = edge_id_pt

                    cli_pt = self._host_ref().w2cli_mat.map(pt)

                    if (mouse_cli_pt-cli_pt).length <= host.pt_select_radius:
                        return edge_id, pt, cli_pt

            return None


    class FStateEditPoly(FStatePoly):

        class EditMode(Enum):
            PT_MOVE = auto()
            PT_MOVE_PERP = auto()
            PT_ADD_DEL = auto()

        def __init__(self, poly : FPoly2f):
            super().__init__(poly)
            self._edit_mode = self.EditMode.PT_MOVE

        def clone(self):
            f = super().clone()
            f._edit_mode = self._edit_mode
            return f

        @property
        def edit_mode(self) -> EditMode: return self._edit_mode
        def set_edit_mode(self, edit_mode : EditMode) -> Self:
            self = self.clone()
            self._edit_mode = edit_mode
            return self


    class FStateDrawPoly(FStatePoly): ...

    def __init__(self, W : int, H : int):
        """Functional core of Mask Editor"""
        super().__init__()
        self._image_size = FVec2i(W, H)
        self._mask_image : FImage = FImage.zeros(W, H, 1)

        self._view_type = self.ViewType.CENTER_FIT
        self._view_proj = FWorldProj2D().set_vp_size(self.cli_size)

        self._drag : FMaskEditor.FDrag|None = None
        self._state : FMaskEditor.FState|None = None

        self._center_on_cursor_cnt : int =  0

    def clone(self) -> Self:
        f = super().clone()
        f._image_size = self._image_size
        f._mask_image = self._mask_image

        f._view_type = self._view_type
        f._view_proj = self._view_proj
        f._drag = self._drag
        f._state = self._state.set_host(f) if self._state is not None else None

        f._center_on_cursor_cnt = self._center_on_cursor_cnt
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

            elif isinstance(state := self._state, self.FStateEditPoly) and \
                 isinstance(drag, self.FDragStatePolyMovePt):
                    pt_id = drag.state_poly_pt_id
                    diff = self.mouse_world_pt - drag.mouse_world_pt

                    if drag.perp_constraint:
                        # Restrict moving by perpendicular of nearby points
                        state_poly = state.poly
                        pts = state_poly.points
                        if (pts_len := len(pts)) >= 3:
                            pl = pts[(pt_id-1) % pts_len]
                            pr = pts[(pt_id+1) % pts_len]
                            q = (pr-pl).normalize().cross
                            diff = q*diff.dot(q)

                    self = self.set_state(state.set_poly(state.poly.replace(pt_id, drag.state_poly_pt + diff)))

        return self

    def mouse_lbtn_down(self, cli_pt: FVec2f, ctrl_pressed=False, shift_pressed=False) -> Self:
        self = super().mouse_lbtn_down(cli_pt, ctrl_pressed, shift_pressed)

        if (mouse_world_pt := self.mouse_world_pt) is not None:
            if (state := self._state) is None:
                # Pressing in NO OPERATION mode
                # Click on empty space -> create new poly with single point
                self = self.set_state(self.FStateDrawPoly(FPoly2f().add(mouse_world_pt)))

            elif isinstance(state, self.FStateDrawPoly):
                # Pressing in DRAW_POLY mode

                if state.poly_pt_id_at_mouse == 0 and state.poly.points_count >= 3:
                    # Click on first point -> switch to edit state
                    self = self.set_state(self.FStateEditPoly(state.poly))
                else:
                    # Click on empty space -> add point to current poly
                    self = self.set_state(state.set_poly(state.poly.add(mouse_world_pt)))

            elif isinstance(state, self.FStateEditPoly):
                # Pressing in EDIT_POLY mode

                if (poly_pt_id_at_mouse := state.poly_pt_id_at_mouse) is not None:
                    # Click on point of state_poly

                    if state.edit_mode == self.FStateEditPoly.EditMode.PT_ADD_DEL:
                        # delete point
                        state_poly = state.poly.remove(poly_pt_id_at_mouse)

                        if state_poly.points_count >= 3:
                            self = self.set_state(state.set_poly(state_poly))
                        else:
                            # not enough points after delete -> exit state
                            self = self.set_state(None)

                    elif state.edit_mode in [self.FStateEditPoly.EditMode.PT_MOVE, self.FStateEditPoly.EditMode.PT_MOVE_PERP]:
                        if self._drag is None:
                            self = self.set_drag(self.FDragStatePolyMovePt(
                                                        mouse_cli_pt=self.mouse_cli_pt,
                                                        mouse_world_pt=self.mouse_world_pt,
                                                        state_poly_pt_id=poly_pt_id_at_mouse,
                                                        state_poly_pt=state.poly.points[poly_pt_id_at_mouse],
                                                        perp_constraint= state.edit_mode == self.FStateEditPoly.EditMode.PT_MOVE_PERP, ))
                else:
                    # Click on non point of state_poly, edge
                    if state.edit_mode == self.FStateEditPoly.EditMode.PT_ADD_DEL:
                        if (edge_id_cli_pt := state.poly_edge_id_pt_at_mouse) is not None:
                            edge_id, pt, _ = edge_id_cli_pt
                            self = self.set_state( state.set_poly(state.poly.insert(edge_id+1, pt)) )

        return self

    def mouse_lbtn_up(self, cli_pt: FVec2f, ctrl_pressed=False, shift_pressed=False) -> Self:
        self = super().mouse_lbtn_up(cli_pt, ctrl_pressed=ctrl_pressed, shift_pressed=shift_pressed)

        if isinstance(self._state, self.FStateEditPoly) and \
           isinstance(self._drag, self.FDragStatePolyMovePt):
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


    # FMaskEditor
    @property
    def image_size(self) -> FVec2i: return self._image_size
    def set_image_size(self, image_size : FVec2i) -> Self:
        self = self.clone()
        self._image_size = image_size
        self._mask_image = self._mask_image.resize(image_size.x, image_size.y)
        return self

    @property
    def mask_image(self) -> FImage: return self._mask_image
    def set_mask_image(self, mask_image : FImage) -> Self:
        self = self.clone()
        self._mask_image = mask_image.ch1().f32().resize(self._image_size.x, self._image_size.y)
        return self

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
            view_proj = view_proj.center_fit(FBBoxf(FVec2f(0,0), self._image_size))
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
    def state(self) -> FState|FStateEditPoly|FStateDrawPoly|None: return self._state
    def set_state(self, state : FState|None) -> Self:
        if not (self._state is state):
            old_state = self._state
            changed_type = type(old_state) != type(state)

            if changed_type:
                # Exit from
                if isinstance(old_state, self.FStateEditPoly):
                    if isinstance(self._drag, self.FDragStatePolyMovePt):
                        self = self.set_drag(None)

            self = self.clone()
            self._state = state.set_host(self) if state is not None else None

            # if changed_type:
            #     # Enter to
        return self

    @property
    def pt_select_radius(self): return 8

    def is_activated_center_on_cursor(self, old : FMaskEditor) -> bool: return self._center_on_cursor_cnt > old._center_on_cursor_cnt

    def center_on_cursor(self) -> Self:
        if (mouse_cli_pt := self.mouse_cli_pt) is not None:
            self = self.set_view_type(self.ViewType.FREE).clone()
            self._view_proj = self._view_proj.set_vp_view_pos(mouse_cli_pt)
            self._center_on_cursor_cnt += 1
        return self

    def _get_mask_from_poly(self) -> FImage|None:
        if isinstance(state := self._state, (self.FStateDrawPoly, self.FStateEditPoly) ):
            state_poly = state.poly
            if state_poly.points_count >= 3:
                mask = np.zeros( (self._image_size.y, self._image_size.x, 1), np.float32)
                cv2.fillPoly(mask, [ state.poly.as_np().astype(np.int32) ], [1.0])
                return FImage.from_numpy(mask)
        return None

    def apply_state_poly(self, apply_type : PolyApplyType) -> Self:
        if isinstance(state := self._state, self.FStateDrawPoly):
            # Currently in FStateDrawPoly mode
            if (mouse_world_pt := self.mouse_world_pt) is not None:
                if state.poly_pt_id_at_mouse != 0:
                    # Mouse not pointing first point id on empty space -> add point to current poly
                    self = self.set_state(state.set_poly(state.poly.add(mouse_world_pt)))

        if isinstance(state := self._state, self.FStateDrawPoly):
            # Currently in FStateDrawPoly mode
            if state.poly.points_count >= 3:
                # Switch to edit state
                self = self.set_state(self.FStateEditPoly(state.poly))
            else:
                self = self.set_state(None)

        if isinstance(state := self._state, self.FStateEditPoly):
            # Currently in FStateEditPoly mode
            if state.poly.points_count != 0:
                poly_mask = self._get_mask_from_poly()

                if apply_type == self.PolyApplyType.INCLUDE:
                    mask_image = (self._mask_image + poly_mask).clip()
                else:
                    mask_image = self._mask_image * poly_mask.invert()

                self = self.clone()
                self._mask_image = mask_image

            self = self.set_state(None)

        return self

    def delete_state_poly(self) -> Self:
        if isinstance(self._state, (self.FStateDrawPoly, self.FStateEditPoly)):
            self = self.set_state(None)
        return self

    def half_edge(self) -> Self:
        if isinstance(state := self._state, self.FStateEditPoly):
            if state.poly_pt_id_at_mouse is None and \
               (edge_id_pt := state.poly_edge_id_pt_at_mouse) is not None:
                edge_id, _, _ = edge_id_pt
                edge = state.poly.edges[edge_id]
                self = self.set_state( state.set_poly(state.poly.insert(edge_id+1, edge.get_line_pt(0.5) )) )
        return self

    def smooth_corner(self) -> Self:
        if isinstance(state := self._state, self.FStateEditPoly):
            if (pt_id := state.poly_pt_id_at_mouse) is not None:
                pts = state.poly.points
                if len(pts) >= 3:

                    state_poly = state.poly.remove(pt_id)

                    p0 = pts[(pt_id - 1) % len(pts)]
                    p1 = pts[pt_id]
                    p2 = pts[(pt_id + 1) % len(pts)]

                    n_pts = 16
                    for i in range(1, n_pts):
                        t = i / n_pts
                        mid_pt = p0*(t**2) + p1*2*t*(1-t) + p2*(1-t)**2
                        state_poly = state_poly.insert(pt_id, mid_pt)

                    self = self.set_state( state.set_poly(state_poly))
        return self

    def recover_from_undo_redo(self, current : FMaskEditor) -> Self:
        self = self.set_cli_size(current.cli_size)
        self = self.set_view_type(current.view_type).clone()
        self._view_proj = current.view_proj
        if (mouse_cli_pt := current.mouse_cli_pt) is not None:
            self = self.mouse_move(mouse_cli_pt)
        return self

    def is_changed_for_undo(self, old : FMaskEditor) -> bool:
        drag = self.drag
        drag_type = type(drag)
        old_drag = old.drag
        old_drag_type = type(old_drag)

        r = type(self.state) != type(old.state)
        r = r or not (self.mask_image is old.mask_image)

        if isinstance(drag, self.FDragStatePolyMovePt):
            r = r or drag_type != old_drag_type
        else:
            if not isinstance(old_drag, self.FDragStatePolyMovePt):
                r = r or (  isinstance(state := self.state, self.FStatePoly) and \
                            isinstance(old_state := old.state, self.FStatePoly) and \
                            not (state.poly is old_state.poly) )
        return r