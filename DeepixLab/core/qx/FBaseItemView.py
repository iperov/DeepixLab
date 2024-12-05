from __future__ import annotations

from typing import overload

from ..lib.math import FBBoxf, FVec2f
from .FBaseWorldViewWidget import FBaseWorldViewWidget


class FBaseItemView(FBaseWorldViewWidget):
    """base code for item view models"""

    def __init__(self):
        super().__init__()
        self._item_count = 0
        self._item_size = FVec2f(64,64)
        self._item_spacing = 4
        
    def clone(self) -> FBaseItemView:
        f = super().clone()
        f._item_count = self._item_count
        f._item_size = self._item_size
        f._item_spacing = self._item_spacing
        return f

    def get_item_id_from_cli_pt(self, pt : FVec2f) -> int|None: return self.get_item_id_from_world_pt(self.mat.inverted.map([pt])[0])
    def get_item_id_from_world_pt(self, pt : FVec2f) -> int|None: raise NotImplementedError()
    def get_item_cli_box(self, item_id : int) -> FBBoxf: raise NotImplementedError()
    
    @property
    def item_count(self) -> int: return self._item_count
    def set_item_count(self, item_count : int) -> FBaseItemView:
        if self._item_count != item_count:
            self = self.clone()
            self._item_count = item_count
        return self

    @property
    def item_size(self) -> FVec2f: return self._item_size
    @overload
    def set_item_size(self, width : float, height : float) -> FBaseItemView: ...
    @overload
    def set_item_size(self, item_size : FVec2f) -> FBaseItemView: ...
    def set_item_size(self, *args) -> FBaseItemView:
        if len(args) == 1:
            item_size, = args
        elif len(args) == 2:
            width, height = args
            item_size = FVec2f(width, height)

        if self._item_size != item_size:
            self = self.clone()
            self._item_size = item_size
        return self


    @property
    def item_spacing(self) -> int: return self._item_spacing
    def set_item_spacing(self, item_spacing : int) -> FBaseItemView:
        if self._item_spacing != item_spacing:
            self = self.clone()
            self._item_spacing = item_spacing
        return self
