from __future__ import annotations

from typing import Self, Sequence

from ..collections import FDict
from ..math import FAffMat2, FRectf, FVec2f, FVec2i
from .FAlignedFace import FAlignedFace
from .FAnnoList import FAnnoList
from .FAnnoLmrk2D import FAnnoLmrk2D
from .FAnnoLmrk2D68 import FAnnoLmrk2D68
from .FAnnoLmrk2D106 import FAnnoLmrk2D106
from .FAnnoLmrk2DYSA import FAnnoLmrk2DYSA
from .FAnnoLmrk2DYSARange import FAnnoLmrk2DYSARange


class FFace:
    """Immutable class. Describes the face in an image."""

    @staticmethod
    def from_state(state : FDict|None) -> FFace|None:
        state = FDict(state)

        if (rect := FRectf.from_state(state.get('rect', None))) is not None:

            if (annotations := FAnnoList.from_state(state.get('annotations', None))) is None:
                annotations = FAnnoList()

            return (FFace(rect)
                            .set_confidence(state.get('confidence', None))
                            .set_detector_id(state.get('detector_id', None))
                            .set_annotations(annotations) )
        return None

    @staticmethod
    def sorted_by_confidence(faces : Sequence[FFace]) -> Sequence[FFace]: return sorted(faces, key=lambda face: face.confidence if face.confidence is not None else 0, reverse=True)
    @staticmethod
    def sorted_by_largest(faces : Sequence[FFace]) -> Sequence[FFace]: return sorted(faces, key=lambda face: face.rect.area, reverse=True)
    @staticmethod
    def sorted_by_dist_from(faces : Sequence[FFace], pt : FVec2f) -> Sequence[FFace]: return sorted(faces, key=lambda face: (face.rect.pc - pt).length )
    @staticmethod
    def sorted_by_left_to_right(faces : Sequence[FFace]) -> Sequence[FFace]: return sorted(faces, key=lambda face: face.rect.pc.x )
    @staticmethod
    def sorted_by_right_to_left(faces : Sequence[FFace]) -> Sequence[FFace]: return sorted(faces, key=lambda face: face.rect.pc.x, reverse=True )
    @staticmethod
    def sorted_by_top_to_bottom(faces : Sequence[FFace]) -> Sequence[FFace]: return sorted(faces, key=lambda face: face.rect.pc.y )
    @staticmethod
    def sorted_by_bottom_to_top(faces : Sequence[FFace]) -> Sequence[FFace]: return sorted(faces, key=lambda face: face.rect.pc.y, reverse=True )

    def __init__(self, rect : FRectf):
        self._rect : FRectf = rect
        self._confidence : float|None = None
        self._detector_id : str|None = None
        self._annotations = FAnnoList()

    def clone(self) -> Self:
        f = self.__class__.__new__(self.__class__)
        f._rect = self._rect
        f._confidence = self._confidence
        f._detector_id = self._detector_id
        f._annotations = self._annotations
        return f

    def get_state(self) -> FDict:
        return FDict({  'rect'         : self._rect.get_state(),
                        'confidence'   : self._confidence,
                        'detector_id'  : self._detector_id,
                        'annotations'  : self._annotations.get_state(), })
    @property
    def rect(self) -> FRectf:
        """detection rectangle"""
        return self._rect
    @property
    def confidence(self) -> float|None:
        """[0..1] confidence of the face"""
        return self._confidence
    @property
    def detector_id(self) -> str|None:
        """id of detector used to detect the face"""
        return self._detector_id
    @property
    def annotations(self) -> FAnnoList:
        """various annotation of face"""
        return self._annotations

    def transform(self, mat : FAffMat2) -> Self:
        """transform and change rect and annotations"""
        return self.set_rect(self._rect.transform(mat)).set_annotations(self._annotations.transform(mat))

    def set_rect(self, rect : FRectf) -> Self:                      f = self.clone(); f._rect = rect; return f
    def set_confidence(self, confidence : float|None) -> Self:      f = self.clone(); f._confidence = confidence; return f
    def set_detector_id(self, detector_id : str|None) -> Self:      f = self.clone(); f._detector_id = detector_id; return f
    def set_annotations(self, annotations : FAnnoList) -> Self:     f = self.clone(); f._annotations = annotations; return f

    def align(self, coverage : float = 1.0, ux_offset : float = 0.0, uy_offset : float = 0.0, y_axis_offset : float = 0.0, min_image_size : int = 128, max_image_size : int = 1024 ) -> FAlignedFace|None:
        """
        try to do the best alignment of the face using provided parameters

        if no annotations found to do the alignment returns None
        """

        if (align_lmrks := self.annotations.get_first_by_class(FAnnoLmrk2DYSARange)) is not None:
            align_lmrks = align_lmrks.to_2DYSA(y_axis_offset)
        else:
            align_lmrks = self.annotations.get_first_by_class_prio([FAnnoLmrk2DYSA, FAnnoLmrk2D106, FAnnoLmrk2D68, FAnnoLmrk2D])

        if isinstance(align_lmrks, FAnnoLmrk2D):

            # Get align mat with specified params
            align_mat = align_lmrks.get_align_mat(coverage=coverage, ux_offset=ux_offset, uy_offset=uy_offset)

            # Transform u_rect by inverted lmrks_align_mat, thus we get global space FRectf
            g_rect = FRectf(1,1).transform( align_mat.inverted )

            # get image_size from g_rect, so it will be close to source size
            # therefore -> less dataset size, less details loss due to interpolations
            image_size = int(g_rect.inflate_to_square().size.x)
            # up to nearest 4
            image_size = image_size - image_size % -4
            # clip to user setting
            image_size = max(min_image_size, min(image_size, max_image_size))

            image_size = FVec2i(image_size, image_size)

            # Get mat to transform global space FRectf to uniform space with size image_size
            mat = FAffMat2.estimate(g_rect, FRectf(image_size))

            # Transfer annotations
            annotations = self.annotations.transform(mat)

            return FAlignedFace(image_size, mat, coverage).set_annotations(annotations)

        return None

