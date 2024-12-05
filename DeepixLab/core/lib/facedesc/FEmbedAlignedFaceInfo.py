from __future__ import annotations

import pickle
import uuid
from pathlib import Path
from typing import Self

from .. import sfeas
from ..collections import FDict
from .FAlignedFace import FAlignedFace
from .FFace import FFace
from .FFrame import FFrame


class FEmbedAlignedFaceInfo:
    """Immutable class. Describes embeddable meta data of aligned face in an image."""

    @staticmethod
    def remove_from(path : Path):
        """raise on error"""
        sfeas.remove(path, FEmbedAlignedFaceInfo._UUID)

    @staticmethod
    def from_embed(path : Path) -> FEmbedAlignedFaceInfo|None:
        """raise on error"""
        try:
            if (data_bytes := sfeas.read(path, FEmbedAlignedFaceInfo._UUID)) is not None:
                return FEmbedAlignedFaceInfo.from_state( pickle.loads(data_bytes) )
        except Exception as e:
            raise Exception('Corrupted FEmbedAlignedFaceInfo')

        return None

    @staticmethod
    def from_state(state : FDict|None) -> FEmbedAlignedFaceInfo|None:
        """
        raise no errors
        """
        state = FDict(state)
        if (aligned_face := FAlignedFace.from_state(state.get('aligned_face', None)) ) is not None and \
           (source_face := FFace.from_state(state.get('source_face', None)) ) is not None and \
           (source_frame := FFrame.from_state(state.get('source_frame', None)) ) is not None:
            return FEmbedAlignedFaceInfo(aligned_face, source_face, source_frame)

        return None

    def __init__(self, aligned_face : FAlignedFace, source_face : FFace, source_frame : FFrame):
        self._aligned_face = aligned_face
        self._source_face = source_face
        self._source_frame = source_frame

    def clone(self) -> Self:
        f = self.__class__.__new__(self.__class__)
        f._aligned_face = self._aligned_face
        f._source_face = self._source_face
        f._source_frame = self._source_frame
        return f

    def get_state(self) -> FDict:
        return FDict({  'aligned_face' : self._aligned_face.get_state(),
                        'source_face' : self._source_face.get_state(),
                        'source_frame' : self._source_frame.get_state(),
                        })

    @property
    def aligned_face(self) -> FAlignedFace: return self._aligned_face
    @property
    def source_face(self) -> FFace: return self._source_face
    @property
    def source_frame(self) -> FFrame: return self._source_frame

    def embed_to(self, path : Path):
        """raises on error"""
        sfeas.append(path, FEmbedAlignedFaceInfo._UUID, pickle.dumps(self.get_state()) )

    def set_aligned_face(self, aligned_face : FAlignedFace) -> Self:    f = self.clone(); f._aligned_face = aligned_face; return f
    def set_source_face(self, source_face : FFace) -> Self:             f = self.clone(); f._source_face = source_face; return f
    def set_source_frame(self, source_frame : FFrame) -> Self:          f = self.clone(); f._source_frame = source_frame; return f

    _UUID = uuid.UUID('cce58050-0ba2-4b7e-bae9-31f63ce8c893')