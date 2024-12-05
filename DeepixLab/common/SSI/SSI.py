from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from core import mx
from core.lib.collections import FDict
from core.lib.image import FImage


class SSI:
    """Sheet-Section-Item"""

    @dataclass(frozen=True)
    class Section: ...

    @dataclass(frozen=True)
    class Item: ...

    @dataclass(frozen=True)
    class Image(Item):
        image   : FImage|None = None
        caption : str|None     = None

        @staticmethod
        def from_state(state : FDict|None) -> SSI.Image|None:
            state = FDict(state)
            try:
                return SSI.Image( image   = FImage.from_state(state.get('image', None)),
                                  caption = state.get('caption', None), )
            except Exception as e:
                return None

        def get_state(self) -> FDict:
            return FDict({  'caption' : self.caption,
                            'image'   : self.image.get_state() if self.image is not None else None})

    @dataclass(frozen=True)
    class Grid(Section):
        """ [ (row,col) ] = item """

        items : Dict[ Tuple[int,int], SSI.Item ] = field(default_factory=dict)

        @staticmethod
        def from_state(state : FDict|None) -> SSI.Grid:
            state = FDict(state)
            try:
                items = {}
                for type_name, key, item_state in state.get('items', []):
                    if (type_cls := getattr(SSI, type_name, None)) is not None:
                        if issubclass(type_cls, SSI.Item):

                            if (item := type_cls.from_state(item_state) ) is not None:
                                items[key] = item

                return SSI.Grid(items=items)
            except Exception as e:
                return SSI.Grid()

        def get_state(self) -> FDict:
            return FDict({'items' : [ (type(item).__name__, key, item.get_state()) for key, item in self.items.items() ] })

    @dataclass(frozen=True)
    class Sheet:
        sections : Dict[str, SSI.Section] = field(default_factory=dict)

        @staticmethod
        def from_state(state : FDict|None) -> SSI.Sheet:
            state = FDict(state)
            try:
                sections = {}
                for type_name, key, section_state in state.get('sections', []):

                    if (type_cls := getattr(SSI, type_name, None)) is not None:
                        if issubclass(type_cls, SSI.Section):

                            if (section := type_cls.from_state(section_state) ) is not None:
                                sections[key] = section

                return SSI.Sheet(sections=sections)
            except Exception as e:
                return SSI.Sheet()

        def get_state(self) -> FDict:
            return FDict({'sections' : [ (type(section).__name__, key, section.get_state()) for key, section in self.sections.items() ] })

    class SheetProp(mx.Property[Sheet]):
        ...

