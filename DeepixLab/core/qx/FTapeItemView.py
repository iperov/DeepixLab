from __future__ import annotations

import math
from functools import cached_property

from ..lib.math import FAffMat2, FBBoxf, FVec2f
from .FBaseItemView import FBaseItemView


class FTapeItemView(FBaseItemView):
    
    def __init__(self):
        super().__init__()
        self._current_item_id = -1

    def clone(self) -> FTapeItemView:
        f = super().clone()
        f._current_item_id=self._current_item_id
        return f
    
    
    # FBaseWorldViewWidget
    @property
    def w2cli_mat(self) -> FAffMat2:
        """mat to transform world to cli"""
        return FAffMat2()
    
    @property
    def current_item_id(self) -> int: return self._current_item_id
    def set_current_item_id(self, current_item_id : int) -> FTapeItemView:
        current_item_id = min(max(0, current_item_id), self.item_count-1)
        if self._current_item_id != current_item_id:
            self = self.clone()
            self._current_item_id = current_item_id
        return self
        
    # Calculable props
    @cached_property
    def col_width(self) -> int: return self.item_size.x + self.item_spacing
    @cached_property
    def row_height(self) -> int: return self.item_size.y + self.item_spacing
    
    @cached_property
    def v_col_count(self) -> int: 
        x = max(1, int(self.cli_size.x // self.col_width))
        if x % 2 == 0:
            x -= 1
        return x   
    @cached_property
    def v_row_count(self) -> int: 
        x = max(1, int(math.ceil(self.cli_size.y / self.row_height)))
        if x % 2 == 0:
            x -= 1
        return x
    
    @cached_property
    def v_grid_item_start(self) -> int: return self._current_item_id - (self.v_row_count*self.v_col_count) // 2
    @cached_property
    def v_grid_item_end(self) -> int: return self._current_item_id + (self.v_row_count*self.v_col_count) // 2
    
    @cached_property
    def v_item_start(self) -> int: return max(0, min(self.v_grid_item_start, self.item_count-1))
    @cached_property
    def v_item_end(self) -> int: return max(0, min(self.v_grid_item_end, self.item_count-1))
    @cached_property
    def v_item_count(self) -> int: return self.v_item_end - self.v_item_start + 1
    
    
    @cached_property
    def v_grid_start_x(self) -> float: return self.cli_size.x/2 - (self.v_col_count//2 + 0.5)*self.col_width
    @cached_property
    def v_grid_start_y(self) -> float: return self.cli_size.y/2 - (self.v_row_count//2 + 0.5)*self.row_height 
    
    @cached_property
    def scroll_value(self) -> float: return self._current_item_id
    @property
    def scroll_value_max(self) -> float: return self.item_count
    
    def get_item_cli_box(self, item_id : int) -> FBBoxf:
        diff_idx = int(item_id-self.v_grid_item_start)
        x = self.v_grid_start_x + (diff_idx % self.v_col_count) * self.col_width  + self.item_spacing//2
        y = self.v_grid_start_y + (diff_idx //self.v_col_count) * self.row_height + self.item_spacing//2
        return FBBoxf(FVec2f(x, y), self.item_size)
    
    def scroll_to(self, value : float) -> FTapeItemView:
        return self.set_current_item_id(value)
    

