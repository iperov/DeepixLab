from __future__ import annotations

from pathlib import Path
from typing import Self
from urllib import parse as url_parse

from ..collections import FDict
from ..math import FVec2i


class FFrame:
    """Immutable class. Describes info about the frame"""

    @staticmethod
    def from_state(state : FDict|None) -> FFrame|None:
        state = FDict(state)

        if (image_size_state := state.get('image_size', None)) is not None and \
           (image_size := FVec2i.from_state(image_size_state)) is not None:

            return (FFrame(image_size)
                        .set_media_url(state.get('media_url', None))
                        .set_frame_idx(state.get('frame_idx', None)) )
        return None

    def __init__(self, image_size : FVec2i):
        self._image_size = image_size
        self._media_url : str|None = None
        self._frame_idx : int|None = None

    def clone(self) -> Self:
        f = self.__class__.__new__(self.__class__)
        f._image_size = self._image_size
        f._media_url = self._media_url
        f._frame_idx = self._frame_idx
        return f

    def get_state(self) -> FDict:
        return FDict({  'image_size' : self._image_size.get_state(),
                        'media_url' : self._media_url,
                        'frame_idx' : self._frame_idx,  })

    @property
    def image_size(self) -> FVec2i: return self._image_size
    @property
    def media_url(self) -> str|None: return self._media_url
    @property
    def media_path(self) -> Path|None:
        """get local file Path from media_url if possible, otherwise None"""
        x = url_parse.urlparse(self._media_url)
        if x.scheme == 'file':
            return Path(url_parse.unquote(x.path.lstrip('/')))
        return None
    @property
    def frame_idx(self) -> int|None:
        """index of frame of video or camera"""
        return self._frame_idx

    def set_media_url(self, media_url : str|None) -> Self:  f = self.clone(); f._media_url = media_url; return f
    def set_frame_idx(self, frame_idx : int|None) -> Self:  f = self.clone(); f._frame_idx = frame_idx; return f


