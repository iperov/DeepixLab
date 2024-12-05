from __future__ import annotations

import math
from functools import cached_property
from typing import Iterable, Self, Tuple

from ..lib.collections import FIndices
from ..lib.functools import cached_method
from ..lib.math import FAffMat2, FBBoxf, FVec2f
from .FBaseItemView import FBaseItemView


class FGridItemView(FBaseItemView):
    def __init__(self):
        super().__init__()

        self._selected_items = FIndices()
        self._selection_anchor = None
        self._marked_items = FIndices()
        self._selecting_mode = 0

        self._scroll_value = 0.0

    def clone(self) -> Self:
        f = super().clone()

        f._selected_items = self._selected_items
        f._selection_anchor = self._selection_anchor
        f._marked_items = self._marked_items
        f._selecting_mode = self._selecting_mode

        f._scroll_value = self._scroll_value
        return f

    # FBaseWidget
    def set_cli_size(self, *args) -> Self:
        v_col_count = self.v_col_count
        v_item_start = self.v_item_start
        v_item_count = self.v_item_count

        self = super().set_cli_size(*args)

        if self.v_col_count != v_col_count:
            # selected_items = self.selected_items
            # if len(selected_items) != 0:
            #     self = self.center_item(selected_items.max)
            # else:
            self = self.center_item(v_item_start + v_item_count//2)

        return self

    def mouse_lbtn_down(self, cli_pt : FVec2f, ctrl_pressed = False, shift_pressed = False) -> Self:
        if not (ctrl_pressed or shift_pressed):
           self = self.unselect_all()

        self = super().mouse_lbtn_down(cli_pt, ctrl_pressed, shift_pressed).clone()

        if ctrl_pressed:
            if (mouse_item_id := self.mouse_item_id) is not None:
                self._selecting_mode = 1 if mouse_item_id in self.selected_items else 2

            self._selecting_mode = 1
        elif shift_pressed:
            #self._selecting_mode = 2

            if (mouse_grid_item_id := self.mouse_grid_item_id) is not None:

                if (selection_anchor := self._selection_anchor) is not None:
                    a_min, a_max = selection_anchor

                    if mouse_grid_item_id <= a_min:
                        self = self.select(FIndices.from_range(mouse_grid_item_id, a_min+1), update_selection_anchor=False )
                    elif mouse_grid_item_id >= a_max:
                        self = self.select(FIndices.from_range(a_max, mouse_grid_item_id+1), update_selection_anchor=False )
        else:
            self._selecting_mode = 2
            self = self.unselect_all()

        return self

    def mouse_lbtn_up(self, cli_pt : FVec2f) -> Self:
        if self._selecting_mode != 0:
            if (aabb := self.selection_aabb) is not None:
                row_aa, row_bb, col_aa, col_bb = aabb

                selected = []
                unselected = []

                v_col_count = self.v_col_count

                for row in range( max(0, row_aa), min(row_bb+1, self.row_count)):
                    for col in range( max(0, col_aa), min(col_bb+1, v_col_count)):
                        item_id = row*v_col_count+col
                        if self._selecting_mode == 1 and item_id in self._selected_items:
                            unselected.append(item_id)
                        else:
                            selected.append(item_id)

                self = self.select( (self.selected_items | selected).difference(unselected) )

        self = super().mouse_lbtn_up(cli_pt).clone()
        self._selecting_mode = 0
        return self

    def mouse_wheel(self, cli_pt : FVec2f, delta : float, ctrl_pressed = False, shift_pressed = False) -> Self:
        self = super().mouse_wheel(cli_pt, delta, ctrl_pressed, shift_pressed).clone()

        diff = 6 if (ctrl_pressed or shift_pressed) else 1
        diff = diff if delta < 0 else -diff
        self = self.scroll_to_value( self.scroll_value + diff )

        return self

    def update(self, time_delta : float) -> Self:
        self = super().update(time_delta)

        if self.mouse_l_down_cli_pt is not None:
            cli_size = self.cli_size
            mouse_cli_pt_y = self.mouse_cli_pt.y
            if mouse_cli_pt_y < 0:
                view_row_diff = mouse_cli_pt_y / self.row_height

            elif mouse_cli_pt_y >= cli_size.y:
                view_row_diff = (mouse_cli_pt_y-cli_size.y) / self.row_height
            else:
                view_row_diff = 0.0

            if view_row_diff != 0.0:
                sign = -1 if view_row_diff < 0 else 1
                self = self.scroll_to_row( self.scroll_value + sign*( 6+abs(view_row_diff)**4)*time_delta )

        return self

    # FBaseWorldViewWidget
    @cached_property
    def w2cli_mat(self) -> FAffMat2:
        return FAffMat2().translate(0, -self.scroll_value*self.row_height)

    # FBaseItemView
    def set_item_count(self, item_count : int) -> Self:
        new_self = super().set_item_count(item_count)
        if not (new_self is self):
            new_self = new_self.clone()
            new_self._selected_items = new_self._selected_items.discard_ge(new_self.item_count)
            new_self._marked_items = new_self._marked_items.discard_ge(new_self.item_count)
        return new_self

    def set_item_size(self, item_width : int, item_height : int) -> Self:
        v_col_count = self.v_col_count
        v_item_start = self.v_item_start
        v_item_count = self.v_item_count

        self = super().set_item_size(item_width, item_height)

        if self.v_col_count != v_col_count:
            selected_items = self.selected_items

            if len(selected_items) != 0:
                self = self.center_item(selected_items.max)
            else:
                self = self.center_item(v_item_start + v_item_count//2)
        return self

    def get_item_id_from_world_pt(self, pt : FVec2f) -> int|None:
        row = int(pt.y / self.row_height)
        col = int(pt.x / self.col_width)
        if row < self.row_count and col < self.v_col_count:
            item_id = row*self.v_col_count + col
            if item_id >= 0 and item_id < self.item_count:
                return item_id
        return None

    def get_item_cli_box(self, item_id : int) -> FBBoxf:
        px = (item_id  % self.v_col_count)*self.col_width  + self.item_spacing//2
        py = (item_id // self.v_col_count)*self.row_height + self.item_spacing//2
        
        return FBBoxf(FVec2f(px, py), self.item_size).transform(self.w2cli_mat)

    # FGridItemView

    @property
    def selecting_mode(self) -> int:
        """0 - no, 1 - inverting selecting, 2 - adding selecting"""
        return self._selecting_mode

    # Calculable props
    @cached_property
    def col_width(self) -> int: return self.item_size.x + self.item_spacing
    @cached_property
    def row_height(self) -> int: return self.item_size.y + self.item_spacing

    @cached_property
    def row_count(self) -> int: return math.ceil(self.item_count / self.v_col_count)
    @cached_property
    def scroll_value(self) -> float: return max(0, min(self._scroll_value, self.scroll_value_max))
    @cached_property
    def scroll_value_max(self) -> float: return max(0, self.row_count-math.floor(self.v_row_count))

    @cached_property
    def v_col_count(self) -> int: return max(1, int(self.cli_size.x // self.col_width))
    @cached_property
    def v_row_start(self) -> float: return self.scroll_value
    @cached_property
    def v_row_count(self) -> float: return self.cli_size.y / self.row_height
    @cached_property
    def v_row_end(self) -> float: return self.v_row_start+self.v_row_count

    @cached_property
    def v_item_start(self) -> int: return int(max(0, min(math.floor(self.scroll_value)*self.v_col_count, self.item_count-1)))
    @cached_property
    def v_item_count(self) -> int: return int(max(0, min(math.ceil(self.v_row_end)*self.v_col_count, self.item_count) - self.v_item_start))
    @cached_property
    def v_items(self) -> FIndices: return FIndices.from_range(self.v_item_start, self.v_item_start+self.v_item_count)

    @cached_property
    def mouse_grid_item_id(self) -> int|None:
        if (mouse_world_pt := self.mouse_world_pt) is not None:
            grid_row = int(mouse_world_pt.y / self.row_height)
            grid_col = int(mouse_world_pt.x / self.col_width)
            return grid_row*self.v_col_count+grid_col
        return None

    @cached_property
    def mouse_item_id(self) -> int|None:
        if (item_id := self.mouse_grid_item_id) is not None:
            if item_id >= 0 and item_id < self.item_count:
                return item_id
        return None

    @cached_property
    def selection_aabb(self) -> Tuple[int,int,int,int]|None:
        if (self._selecting_mode != 0) and \
           (mouse_l_down_world_pt := self.mouse_l_down_world_pt) is not None and \
           (mouse_world_pt := self.mouse_world_pt) is not None:

            grid_row_1 = int(mouse_l_down_world_pt.y / self.row_height)
            grid_col_1 = int(mouse_l_down_world_pt.x / self.col_width)

            grid_row_2 = int(mouse_world_pt.y / self.row_height)
            grid_col_2 = int(mouse_world_pt.x / self.col_width)

            row_aa = max(0, min(grid_row_1, grid_row_2))
            row_bb = max(0, grid_row_1, grid_row_2)

            col_aa = max(0, min(grid_col_1, grid_col_2))
            col_bb = max(0, grid_col_1, grid_col_2)
            return row_aa, row_bb, col_aa, col_bb
        return None

    @cached_property
    def selection_cli_box(self) -> FBBoxf:
        """if selecting_mode != 0"""
        return FBBoxf([self.w2cli_mat.map([self.mouse_l_down_world_pt])[0], self.mouse_cli_pt])

    @property
    def selected_items(self) -> FIndices: return self._selected_items
    @property
    def selection_anchor(self) -> Tuple[int,int]|None:
        """pair min, max indices from which shift-selection will select a whole range"""
        return self._selection_anchor
    @property
    def marked_items(self) -> FIndices: return self._marked_items
    def is_selected(self, item_id : int) -> bool: return item_id in self._selected_items
    def is_marked(self, item_id : int) -> bool: return item_id in self._marked_items
    @cached_method
    def is_selected_all(self) -> bool: return self._selected_items.count == self.item_count
    @cached_method
    def is_marked_all(self) -> bool: return self._marked_items.count == self.item_count

    def select_all(self) -> Self:
        if not self.is_selected_all():
            self = self.select(FIndices.from_range(self.item_count))
        return self

    def unselect_all(self) -> Self:
        if self._selected_items.count != 0:
            self = self.select(None)
        return self

    def mark(self, indices : FIndices|Iterable|int|None) -> Self:
        """replace marked items with `incices`"""
        self = self.clone()
        self._marked_items = FIndices(indices).discard_ge(self.item_count)
        return self

    def mark_all(self) -> Self:
        if not self.is_marked_all():
            self = self.mark(FIndices.from_range(self.item_count))
        return self

    def unmark_all(self) -> Self:
        if len(self._marked_items) != 0:
            self = self.mark(None)
        return self

    def select(self, indices : FIndices|Iterable|int|None, update_selection_anchor=True) -> Self:
        """replace selected items with `indices`"""
        self = self.clone()

        new_selected_items = FIndices(indices).discard_ge(self.item_count)
        if update_selection_anchor:
            anchor_ind = new_selected_items.difference(self._selected_items)
            if len(anchor_ind) != 0:
                self._selection_anchor = (anchor_ind.min, anchor_ind.max)
            else:
                self._selection_anchor = None
        self._selected_items = new_selected_items
        return self


    def select_prev(self, row_up=False, page_up=False, first=False) -> Self:
        if self.item_count == 0:
            return self

        if first:
            indice = 0
        else:
            indices = self._selected_items
            if len(indices) == 0:
                indices = self.v_items
            indice = indices.min if len(indices) != 0 else 0

            if row_up:
                indice -= self.v_col_count
            elif page_up:
                indice -= ( math.ceil(self.v_row_count / 2)) * self.v_col_count
            else:
                indice -= 1
        indice = max(0, indice)

        return self.select(indice).ensure_visible_item(indice)

    def select_next(self, row_down=False, page_down=False, last=False) -> Self:
        if self.item_count == 0:
            return self

        if last:
            indice = self.item_count-1
        else:
            indices = self._selected_items
            if len(indices) == 0:
                indices = self.v_items

            indice = indices.max if len(indices) != 0 else 0

            if row_down:
                indice += self.v_col_count
            elif page_down:
                indice += ( math.ceil(self.v_row_count / 2)) * self.v_col_count
            else:
                indice += 1
        indice = min(indice, self.item_count-1)

        return self.select(indice).ensure_visible_item(indice)


    def scroll_to_value(self, value : float) -> Self: return self.scroll_to_row(value)
    def scroll_to_row(self, row : float) -> Self:
        if self._scroll_value != row:
            self = self.clone()
            self._scroll_value = row
        return self

    def center_item(self, item_id : int) -> Self:
        if item_id >= 0 and item_id < self.item_count:
            self = self.center_row(item_id // self.v_col_count)
        return self

    def center_row(self, row : int) -> Self:
        return self.scroll_to_row(row - self.v_row_count / 2 + 0.5)

    def ensure_visible_item(self, item_id : int) -> Self:
        if item_id >= 0 and item_id < self.item_count:
            self = self.ensure_visible_row(item_id // self.v_col_count)
        return self

    def ensure_visible_row(self, row : int) -> Self:
        if row < self.v_row_start:
            self = self.scroll_to_row(row)
        elif row >= math.floor(self.v_row_end):
            self = self.scroll_to_row( row-self.v_row_count+1 )
        return self





