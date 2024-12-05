from __future__ import annotations

from functools import cached_property
from typing import Self

from ..collections import FDict
from ..math import FAffMat2, FVec2f, FVec2i
from .FAnnoList import FAnnoList


class FAlignedFace:
    """Immutable class. Describes the aligned face"""

    @staticmethod
    def from_state(state : FDict|None) -> FAlignedFace|None:
        state = FDict(state)

        if (image_size := FVec2i.from_state(state.get('image_size', None))) is not None and \
           (mat := FAffMat2.from_state(state.get('mat', None))) is not None and \
           (coverage := state.get('coverage', None)) is not None:

            if (annotations := FAnnoList.from_state(state.get('annotations', None))) is None:
                annotations = FAnnoList()

            return FAlignedFace(image_size, mat, coverage).set_annotations(annotations)

        return None

    def __init__(self, image_size : FVec2i, mat : FAffMat2, coverage : float):
        self._image_size = image_size
        self._mat = mat
        self._coverage = coverage
        self._annotations = FAnnoList()

    def clone(self) -> Self:
        f = self.__class__.__new__(self.__class__)
        f._image_size = self._image_size
        f._mat = self._mat
        f._coverage = self._coverage
        f._annotations = self._annotations
        return f

    def get_state(self) -> FDict:
        return FDict({  'image_size'    : self._image_size.get_state(),
                        'mat'           : self._mat.get_state(),
                        'coverage'      : self._coverage,
                        'annotations'   : self._annotations.get_state(),    })

    @property
    def image_size(self) -> FVec2i:
        """size of image containing the aligned face"""
        return self._image_size
    @property
    def mat(self) -> FAffMat2:
        """FAffMat2 matrix for transform source space to aligned space"""
        return self._mat
    @cached_property
    def uni_mat(self) -> FAffMat2:
        """FAffMat2 matrix for transform source space to uniform aligned space"""
        return self._mat.scale(1/FVec2f(self._image_size))
    @property
    def coverage(self) -> float:
        """coverage value used to align the face"""
        return self._coverage
    @property
    def annotations(self) -> FAnnoList: return self._annotations

    def transform(self, mat : FAffMat2) -> Self:
        """transform and change mat and annotations"""
        return self.set_mat(self._mat*mat).set_annotations( self._annotations.transform(mat))

    def resize(self, image_size : FVec2i) -> Self:
        """set new image_size, and transform"""
        sv = FVec2f(image_size) / FVec2f(self._image_size)
        if sv.x != 1.0 or sv.y != 1.0:
            self = self.set_image_size(image_size).transform( FAffMat2().scale(sv) )
        return self

    def set_image_size(self, image_size : FVec2i) -> Self:      f = self.clone(); f._image_size = image_size; return f
    def set_mat(self, mat : FAffMat2) -> Self:                  f = self.clone(); f._mat = mat; return f
    def set_coverage(self, coverage : float) -> Self:           f = self.clone(); f._coverage = coverage; return f
    def set_annotations(self, annotations : FAnnoList) -> Self: f = self.clone(); f._annotations = annotations; return f




