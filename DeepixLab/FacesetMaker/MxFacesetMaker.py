from __future__ import annotations

import multiprocessing
import shutil
from enum import StrEnum
from pathlib import Path
from typing import List, Self, Sequence

import numpy as np

from common.ImageFormat import MxImageFormat
from common.MediaSource import MxMediaSource
from core import ax, mx
from core.lib import facedesc as fd
from core.lib import onnxruntime as lib_ort
from core.lib.collections import FDict, HFDict, get_enum_id_by_name
from core.lib.DFLIMG import DFLJPG
from core.lib.image import FImage
from core.lib.math import FAffMat2, FRectf, FVec2f, FVec2i
from core.lib.time import SPSCounter
from modelhub.onnx.IDMMD import IDMMD
from modelhub.onnx.TDDFAV3 import TDDFAV3
from modelhub.onnx.YoloV7Face import YoloV7Face


class MxFacesetMaker(mx.Disposable):

    class DetectorResolution(StrEnum):
        Source = '@(Source)'
        _480p = '480p'
        _720p = '720p'
        _1080p = '1080p'

    class FaceDetectorType(StrEnum):
        YoloV7 = 'YoloV7'

    class FaceMarkerType(StrEnum):
        TDDFAV3 = '3DDFAv3'

    class FaceIdentifierType(StrEnum):
        Not_applied = '@(Not_applied)'
        IDMMD = 'IDMMD'


    class SortBy(StrEnum):
        Confidence = '@(Confidence)'
        Largest = '@(Largest)'
        DistFromCenter = '@(Dist_from_center)'
        LeftToRight = '@(Left_to_right)'
        RightToLeft = '@(Right_to_left)'
        TopToBottom = '@(Top_to_bottom)'
        BottomToTop = '@(Bottom_to_top)'


    def __init__(self, state : FDict|None = None):
        super().__init__()

        self._state = state = HFDict(state)

        self._main_thread = ax.get_current_thread()

        self._mx_error = mx.TextEmitter().dispose_with(self)
        self._mx_info = mx.TextEmitter().dispose_with(self)


        self._mx_media_source = MxMediaSource(self._on_media_source_frame, state=self._state.get('mx_media_source', None)).dispose_with(self)

        sub_state = HFDict(state.get('mx_media_source.mx_source_type', None))

        state_upd = lambda: state.update({'mx_media_source' : self._mx_media_source.get_state(),
                                          'mx_media_source.mx_source_type' : sub_state})
        self._update_state_ev = mx.Event0().dispose_with(self)
        self._update_state_ev.listen(state_upd).dispose_with(self)
        mx.CallOnDispose(state_upd).dispose_with(self)

        self._mx_media_source.mx_source_type.reflect(lambda source_type, enter, bag=mx.Disposable().dispose_with(self):
                                                     self._ref_ms_type(source_type, enter, bag, state=sub_state)).dispose_with(self)

    def get_state(self) -> FDict:
        self._update_state_ev.emit(reverse=True)
        return FDict(self._state)

    @property
    def mx_error(self) -> mx.ITextEmitter_v: return self._mx_error
    @property
    def mx_info(self) -> mx.ITextEmitter_v: return self._mx_info
    @property
    def mx_media_source(self) -> MxMediaSource: return self._mx_media_source

    ###############################################################
    ######## All below avail if mx_media_source.mx_media_path.mx_opened

    # Face detector
    @property
    def mx_face_detector_type(self) -> mx.IStateChoice_v[FaceDetectorType]: return self._mx_face_detector_type
    @property
    def mx_face_detector_device(self) -> mx.IStateChoice_v: return self._mx_face_detector_device
    @property
    def mx_detector_resolution(self) -> mx.IStateChoice_v: return self._mx_detector_resolution
    @property
    def mx_augment_pyramid(self) -> mx.IFlag_v: return self._mx_augment_pyramid
    @property
    def mx_detector_minimum_confidence(self) -> mx.INumber_v: return self._mx_detector_minimum_confidence
    @property
    def mx_detector_overlap_threshold(self) -> mx.INumber_v: return self._mx_detector_overlap_threshold
    @property
    def mx_min_face_size(self) -> mx.INumber_v: return self._mx_min_face_size

    # Face marker
    @property
    def mx_face_marker_type(self) -> mx.IStateChoice_v[FaceMarkerType]: return self._mx_face_marker_type
    @property
    def mx_face_marker_device(self) -> mx.IStateChoice_v: return self._mx_face_marker_device
    @property
    def mx_pass_count(self) -> mx.INumber_v: return self._mx_pass_count

    # Face identifier
    @property
    def mx_face_identifier_type(self) -> mx.IStateChoice_v[FaceMarkerType]: return self._mx_face_identifier_type
    @property
    def mx_face_identifier_device(self) -> mx.IStateChoice_v: return self._mx_face_identifier_device

    # Face aligner
    @property
    def mx_face_coverage(self) -> mx.INumber_v: return self._mx_face_coverage
    @property
    def mx_face_y_offset(self) -> mx.INumber_v: return self._mx_face_y_offset
    @property
    def mx_face_y_axis_offset(self) -> mx.INumber_v: return self._mx_face_y_axis_offset

    @property
    def mx_min_image_size(self) -> mx.INumber_v: return self._mx_min_image_size
    @property
    def mx_max_image_size(self) -> mx.INumber_v: return self._mx_max_image_size
    @property
    def mx_border_type(self) -> mx.IStateChoice_v[FImage.Border]: return self._mx_border_type

    # Face list
    @property
    def mx_sort_by_type(self) -> mx.IStateChoice_v: return self._mx_sort_by_type
    @property
    def mx_max_faces(self) -> mx.INumber_v: return self._mx_max_faces
    @property
    def mx_max_faces_discard(self) -> mx.IFlag_rv: return self._mx_max_faces_discard

    # System
    @property
    def mx_jobs_max(self) -> mx.INumber_v: return self._mx_jobs_max
    @property
    def mx_jobs_count(self) -> mx.Property[int]: return self._mx_jobs_count
    @property
    def mx_jobs_done_per_sec(self) -> mx.Property[float]: return self._mx_jobs_done_per_sec

    # Export
    @property
    def mx_export_path(self) -> mx.IPath_v: return self._mx_export_path
    @property
    def mx_export_file_format(self) -> MxImageFormat: return self._mx_export_file_format

    @property
    def mx_export_dfl_mask(self) -> mx.IFlag_v:
        """avail if source_type == ImageSequence"""
        return self._mx_export_dfl_mask

    @property
    def mx_ded_progress(self) -> mx.IProgress_rv: return self._mx_ded_progress

    @property
    def mx_export_enabled(self) -> mx.IFlag_v: return self._mx_export_enabled

    @property
    def mx_preview_frame(self) -> mx.IProperty_rv[FParsedFrame|None]: return self._mx_preview_frame
    @property
    def mx_preview_image_size(self) -> mx.INumber_v: return self._mx_preview_image_size
    @property
    def mx_preview_draw_annotations(self) -> mx.IFlag_v: return self._mx_preview_draw_annotations

    @ax.task
    def delete_export_directories(self):
        """"""
        if self._mx_media_source.mx_media_path.mx_opened.get():
            yield ax.attach_to(self._fg)
            yield ax.switch_to(self._main_thread)

            self._mx_ded_progress.start().set_inf()

            wipe_paths = []

            if self._mx_export_path.mx_opened.get() and \
              (path := self._mx_export_path.get()).exists():
                wipe_paths.append(path)

            yield ax.switch_to(self._thread_pool)

            err = None
            try:
                for path in wipe_paths:
                    shutil.rmtree(path)
            except Exception as e:
                err = e

            yield ax.switch_to(self._main_thread)

            self._mx_ded_progress.finish()
            if err is not None:
                self._mx_error.emit(str(e))

    ###############################################################

    def _ref_ms_type(self, source_type : MxMediaSource.SourceType, enter : bool,  bag : mx.Disposable, state : HFDict):
        if enter:
            ms = self._mx_media_source

            sub_state = HFDict(state.get('mx_media_path.mx_opened', None))
            state_upd = lambda: state.set('mx_media_path.mx_opened', sub_state)

            self._update_state_ev.listen(state_upd).dispose_with(bag)
            mx.CallOnDispose(state_upd).dispose_with(bag)

            ms.mx_media_path.mx_opened.reflect(lambda opened, enter, bag=mx.Disposable().dispose_with(bag):
                                               self._ref_ms_path(opened, enter, bag, state=sub_state)).dispose_with(bag)
        else:
            bag.dispose_items()


    def _ref_ms_path(self, opened : bool, enter : bool, bag : mx.Disposable, state : HFDict):
        if enter:
            if opened:
                self._fg = ax.FutureGroup().dispose_with(bag)

                # Single thread where models will be created in order to prevent concurrent GPU errors.
                self._model_creator_thread = ax.Thread('model_creator_thread').dispose_with(bag)
                self._face_detector_thread = ax.Thread('face_detector_thread').dispose_with(bag)
                self._face_marker_thread = ax.Thread('face_marker_thread').dispose_with(bag)
                self._face_identifier_thread = ax.Thread('face_identifier_thread').dispose_with(bag)
                self._preview_thread = ax.Thread('preview_thread').dispose_with(bag)
                self._thread_pool = ax.ThreadPool().dispose_with(bag)

                self._jobs : List[ax.Future[MxFacesetMaker.FParsedFrame]] = []
                self._jobs_sps = SPSCounter()

                ort_devices = lib_ort.get_avail_gpu_devices() + [lib_ort.get_cpu_device()]
                ort_best_device = lib_ort.get_best_device(ort_devices)

                if (face_detector_device_state := state.get('face_detector_device_state', None) ) is not None:
                    face_detector_device = lib_ort.DeviceRef.from_state(face_detector_device_state)
                else:
                    face_detector_device = ort_best_device

                if (face_marker_device_state := state.get('face_marker_device_state', None) ) is not None:
                    face_marker_device = lib_ort.DeviceRef.from_state(face_marker_device_state)
                else:
                    face_marker_device = ort_best_device

                if (face_identifier_device_state := state.get('face_identifier_device_state', None) ) is not None:
                    face_identifier_device = lib_ort.DeviceRef.from_state(face_identifier_device_state)
                else:
                    face_identifier_device = ort_best_device

                self._mx_face_detector_type = mx.StateChoice[self.FaceDetectorType](availuator=lambda: self.FaceDetectorType).dispose_with(bag)
                self._mx_face_detector_type.set(get_enum_id_by_name(self.FaceDetectorType, state.get('face_detector_type', None), self.FaceDetectorType.YoloV7))

                self._mx_face_detector_device = mx.StateChoice[lib_ort.DeviceRef](availuator=lambda: ort_devices).dispose_with(bag) \
                                                    .set(face_detector_device)

                self._mx_detector_resolution = mx.StateChoice[MxFacesetMaker.DetectorResolution](availuator=lambda: [x for x in MxFacesetMaker.DetectorResolution]).dispose_with(bag)
                self._mx_detector_resolution.set(get_enum_id_by_name(MxFacesetMaker.DetectorResolution, state.get('detector_resolution', None), MxFacesetMaker.DetectorResolution._720p))
                self._mx_detector_resolution.listen(lambda *_: self._mx_media_source.reemit_current_frame())

                self._mx_augment_pyramid = mx.Flag(state.get('augment_pyramid', True)).dispose_with(bag)
                self._mx_augment_pyramid.listen(lambda *_: self._mx_media_source.reemit_current_frame())

                self._mx_detector_minimum_confidence = mx.Number(state.get('detector_minimum_confidence', 0.3), config=mx.Number.Config(min=0.01, max=1.0, step=0.01)).dispose_with(bag)
                self._mx_detector_minimum_confidence.listen(lambda *_: self._mx_media_source.reemit_current_frame())

                self._mx_detector_overlap_threshold = mx.Number(state.get('detector_overlap_threshold', 0.3), config=mx.Number.Config(min=0.01, max=1.0, step=0.01)).dispose_with(bag)
                self._mx_detector_overlap_threshold.listen(lambda *_: self._mx_media_source.reemit_current_frame())

                self._mx_min_face_size = mx.Number(state.get('min_face_size', 40), config=mx.Number.Config(min=4, max=1024, step=1)).dispose_with(bag)
                self._mx_min_face_size.listen(lambda *_: self._mx_media_source.reemit_current_frame())

                self._mx_face_marker_type = mx.StateChoice[self.FaceMarkerType](availuator=lambda: self.FaceMarkerType).dispose_with(bag)
                self._mx_face_marker_type.set(get_enum_id_by_name(self.FaceMarkerType, state.get('face_marker_type', None), self.FaceMarkerType.TDDFAV3))

                self._mx_face_marker_device = mx.StateChoice[lib_ort.DeviceRef](availuator=lambda: ort_devices).dispose_with(bag) \
                                                    .set(face_marker_device)

                self._mx_pass_count = mx.Number(state.get('pass_count', 2), config=mx.Number.Config(min=1, max=3, step=1)).dispose_with(bag)
                self._mx_pass_count.listen(lambda *_: self._mx_media_source.reemit_current_frame())

                self._mx_face_identifier_type = mx.StateChoice[self.FaceIdentifierType](availuator=lambda: self.FaceIdentifierType).dispose_with(bag)
                self._mx_face_identifier_type.set(get_enum_id_by_name(self.FaceIdentifierType, state.get('face_identifier_type', None), self.FaceIdentifierType.IDMMD))

                self._mx_face_identifier_device = mx.StateChoice[lib_ort.DeviceRef](availuator=lambda: ort_devices).dispose_with(bag) \
                                                    .set(face_identifier_device)

                self._mx_face_coverage = mx.Number(state.get('face_coverage', 1.8), config=mx.Number.Config(min=1.0, max=4.0, step=0.05)).dispose_with(bag)
                self._mx_face_coverage.listen(lambda *_: self._mx_media_source.reemit_current_frame())

                self._mx_face_y_offset = mx.Number(state.get('face_y_offset', -0.14), config=mx.Number.Config(min=-0.5, max=0.5, step=0.01)).dispose_with(bag)
                self._mx_face_y_offset.listen(lambda *_: self._mx_media_source.reemit_current_frame())

                self._mx_face_y_axis_offset = mx.Number(state.get('face_y_axis_offset', 0.0), config=mx.Number.Config(min=-1.0, max=1.0, step=0.01)).dispose_with(bag)
                self._mx_face_y_axis_offset.listen(lambda *_: self._mx_media_source.reemit_current_frame())

                self._mx_min_image_size = mx.Number(state.get('min_image_size', 128), config=mx.Number.Config(min=64, max=2048, step=64)).dispose_with(bag)
                self._mx_min_image_size.listen(lambda s: ( self._mx_max_image_size.set(s) if self._mx_max_image_size.get() < s else ...,
                                                            self._mx_media_source.reemit_current_frame() ) )

                self._mx_max_image_size = mx.Number(state.get('max_image_size', 1024), config=mx.Number.Config(min=64, max=2048, step=64)).dispose_with(bag)
                self._mx_max_image_size.listen(lambda s: ( self._mx_min_image_size.set(s) if self._mx_min_image_size.get() > s else ...,
                                                            self._mx_media_source.reemit_current_frame() ) )

                self._mx_border_type = mx.StateChoice[FImage.Border](availuator=lambda: [FImage.Border.CONSTANT, FImage.Border.REPLICATE]).dispose_with(self)
                self._mx_border_type.set( get_enum_id_by_name(FImage.Border, state.get('border_type', None), FImage.Border.CONSTANT) )
                self._mx_border_type.listen(lambda *_: self._mx_media_source.reemit_current_frame())

                self._mx_sort_by_type = mx.StateChoice[MxFacesetMaker.SortBy](availuator=lambda: [*MxFacesetMaker.SortBy]).dispose_with(bag)
                self._mx_sort_by_type.set(get_enum_id_by_name(MxFacesetMaker.SortBy, state.get('sort_by_type', None), MxFacesetMaker.SortBy.Confidence))
                self._mx_sort_by_type.listen(lambda *_: self._mx_media_source.reemit_current_frame())

                self._mx_max_faces =  mx.Number(state.get('max_faces', 0), config=mx.Number.Config(min=0, max=16, step=1)).dispose_with(bag)
                self._mx_max_faces.listen(lambda *_: self._mx_media_source.reemit_current_frame())

                self._mx_max_faces_discard = mx.Flag(state.get('max_faces_discard', False)).dispose_with(bag)
                self._mx_max_faces_discard.listen(lambda *_: self._mx_media_source.reemit_current_frame())

                self._mx_jobs_max = mx.Number(1, config=mx.Number.Config(min=1, max=multiprocessing.cpu_count()*2, step=1)).dispose_with(bag)
                self._mx_jobs_max.set(state.get('jobs_max', self._mx_jobs_max.config.max))

                self._mx_jobs_count = mx.Property[int](0).dispose_with(bag)
                self._mx_jobs_done_per_sec = mx.Property[float](0.0).dispose_with(bag)

                self._mx_preview_frame = mx.Property[MxFacesetMaker.FParsedFrame|None](None).dispose_with(bag)
                self._mx_preview_image_size = mx.Number(state.get('preview_image_size', 160), config=mx.Number.Config(min=32, max=512, step=32)).dispose_with(bag)
                self._mx_preview_image_size.listen(lambda *_: self._mx_media_source.reemit_current_frame())

                self._mx_preview_draw_annotations = mx.Flag(state.get('preview_draw_annotations', False)).dispose_with(bag)
                self._mx_preview_draw_annotations.listen(lambda *_: self._mx_media_source.reemit_current_frame())

                self._mx_export_path = mx.Path( mx.Path.Config(dir=True, allow_open=True, allow_new=True, desc='Export directory'), ).dispose_with(bag)
                self._mx_export_file_format = MxImageFormat(state=state.get('export_file_format_state', None)).dispose_with(bag)
                self._mx_export_dfl_mask = mx.Flag(state.get('export_dfl_mask', False)).dispose_with(bag)

                self._mx_ded_progress = mx.Progress().dispose_with(bag)

                self._mx_export_enabled = mx.Flag(False).dispose_with(bag)
                self._mx_export_enabled.listen(lambda export_enabled: self._mx_media_source.reemit_current_frame() if export_enabled else ...)


                state_upd = lambda is_save_export_paths: state.update(
                        {   'face_detector_type' : self._mx_face_detector_type.get().name,
                            'face_detector_device_state' : self._mx_face_detector_device.get().get_state(),
                            'detector_resolution' : self._mx_detector_resolution.get().name,
                            'augment_pyramid' : self._mx_augment_pyramid.get(),
                            'detector_minimum_confidence' : self._mx_detector_minimum_confidence.get(),
                            'detector_overlap_threshold' : self._mx_detector_overlap_threshold.get(),
                            'min_face_size' : self._mx_min_face_size.get(),

                            'face_marker_type' : self._mx_face_marker_type.get().name,
                            'face_marker_device_state' : self._mx_face_marker_device.get().get_state(),
                            'pass_count' : self._mx_pass_count.get(),
                            'face_identifier_type' : self._mx_face_identifier_type.get().name,
                            'face_identifier_device_state' : self._mx_face_identifier_device.get().get_state(),

                            'face_coverage' : self._mx_face_coverage.get(),
                            'face_y_offset' : self._mx_face_y_offset.get(),
                            'face_y_axis_offset' : self._mx_face_y_axis_offset.get(),
                            'min_image_size' : self._mx_min_image_size.get(),
                            'max_image_size' : self._mx_max_image_size.get(),
                            'border_type' : self._mx_border_type.get().name,

                            'sort_by_type' : self._mx_sort_by_type.get().name,
                            'max_faces' : self._mx_max_faces.get(),
                            'max_faces_discard' : self._mx_max_faces_discard.get(),

                            'jobs_max' : self._mx_jobs_max.get(),

                            'preview_image_size' : self._mx_preview_image_size.get(),
                            'preview_draw_annotations' : self._mx_preview_draw_annotations.get(),

                            'export_path' : self._mx_export_path.get() if is_save_export_paths else None,
                            'export_file_format_state' : self._mx_export_file_format.get_state(),

                            'export_dfl_mask' : self._mx_export_dfl_mask.get(),

                        })

                self._update_state_ev.listen(lambda: state_upd(is_save_export_paths=True)).dispose_with(bag)
                mx.CallOnDispose(lambda: state_upd(is_save_export_paths=False)).dispose_with(bag)

                media_path = self._mx_media_source.mx_media_path.get()

                if (export_path := state.get('export_path', None)) is None:
                    export_path = media_path.parent / f'{media_path.name}_faces'

                self._mx_export_path.new(export_path)

                self._face_detector : YoloV7Face = ...
                self._face_markers : Sequence = []
                self._face_identifier : IDMMD = ...
                
                self._mx_face_detector_type.listen(lambda type, enter: self._on_face_detector_type_device())
                self._mx_face_detector_device.reflect(lambda device, enter: self._on_face_detector_type_device())

                self._mx_face_marker_type.listen(lambda type, enter: self._on_face_marker_type_device())
                self._mx_face_marker_device.reflect(lambda device, enter: self._on_face_marker_type_device())

                self._mx_face_identifier_type.listen(lambda type, enter: self._on_face_identifier_type_device())
                self._mx_face_identifier_device.reflect(lambda device, enter: self._on_face_identifier_type_device())


                self._bg_task()
        else:
            bag.dispose_items()




    @ax.task
    def _on_face_detector_type_device(self):
        yield ax.attach_to(self._fg)
        yield ax.switch_to(self._main_thread)

        face_detector_type = self._mx_face_detector_type.get()
        face_detector_device = self._mx_face_detector_device.get()

        yield ax.switch_to(self._model_creator_thread)

        err = None
        try:
            if face_detector_type == self.FaceDetectorType.YoloV7:
                face_detector = YoloV7Face(face_detector_device)
            else:
                raise ValueError()

        except Exception as e:
            err = e
            face_detector = None

        self._face_detector = face_detector

        yield ax.switch_to(self._main_thread)

        if err is not None:
            self._mx_error.emit(str(err))
        else:
            self._mx_media_source.reemit_current_frame()

    @ax.task
    def _on_face_marker_type_device(self):
        yield ax.attach_to(self._fg)
        yield ax.switch_to(self._main_thread)

        face_marker_type = self._mx_face_marker_type.get()
        face_marker_device = self._mx_face_marker_device.get()

        yield ax.switch_to(self._model_creator_thread)

        err = None
        face_markers = []
        if face_marker_type is not None and face_marker_device is not None:
            try:
                face_markers_cls = []
                if face_marker_type in [self.FaceMarkerType.TDDFAV3]:
                    face_markers_cls.append(TDDFAV3)

                face_markers = [ face_marker_cls(face_marker_device) for face_marker_cls in face_markers_cls]
            except Exception as e:
                err = e

        self._face_markers = face_markers

        yield ax.switch_to(self._main_thread)

        if err is not None:
            self._mx_error.emit(str(err))
        else:
            self._mx_media_source.reemit_current_frame()


    @ax.task
    def _on_face_identifier_type_device(self):
        yield ax.attach_to(self._fg)
        yield ax.switch_to(self._main_thread)

        face_identifier_type = self._mx_face_identifier_type.get()
        face_identifier_device = self._mx_face_identifier_device.get()

        yield ax.switch_to(self._model_creator_thread)

        err = None
        try:
            if face_identifier_type == self.FaceIdentifierType.Not_applied:
                face_identifier = None
            elif face_identifier_type == self.FaceIdentifierType.IDMMD:
                face_identifier = IDMMD(face_identifier_device)
            else:
                raise ValueError()

        except Exception as e:
            err = e
            face_identifier = None

        self._face_identifier = face_identifier

        yield ax.switch_to(self._main_thread)

        if err is not None:
            self._mx_error.emit(str(err))
        else:
            self._mx_media_source.reemit_current_frame()

    @ax.task
    def _to_preview(self, p_frame : FParsedFrame):
        yield ax.switch_to(self._preview_thread)

        TS = self._mx_preview_image_size.get()

        new_p_faces = []

        for p_face in p_frame.faces:

            if (aligned_face := p_face.aligned_face) is not None:
                if (image := p_face.aligned_face_image) is not None:
                    IS, _, _ = image.shape

                    image = image.resize(TS, TS)

                    if self._mx_preview_draw_annotations.get():
                        image = image.u8().bgr()


                        if (mat := aligned_face.mat) is not None:
                            image = image.draw_rect(p_face.face.rect.transform( mat.scale(TS/IS) ), [0,0,255], thickness=1)

                        # if (ysa_range := aligned_face.annotations.get_first_by_class(fd.FAnnoLmrk2DYSARange)) is not None:
                        #     ysa = ysa_range.to_2DYSA()
                        # else:
                        #     ysa = aligned_face.annotations.get_first_by_class(fd.FAnnoLmrk2DYSA)

                        # if ysa is not None:
                        #     image = ysa.transform( FAffMat2().scale(TS/IS) ).draw(image, [0,255,255], radius=1)

                        lmrks = aligned_face.annotations.get_first_by_class_prio([fd.FAnnoLmrk2D106, fd.FAnnoLmrk2D68, fd.FAnnoLmrk2D])

                        if isinstance(lmrks, fd.FAnnoLmrk2D):
                            image = lmrks.transform( FAffMat2().scale(TS/IS) ).draw(image, [0,255,0], radius=1)

                    p_face = p_face.set_aligned_face_image(image)

            new_p_faces.append(p_face)

        p_frame = p_frame.set_faces(new_p_faces)

        yield ax.switch_to(self._main_thread)

        self._mx_preview_frame.set(p_frame)

    @ax.task
    def _bg_task(self):
        yield ax.attach_to(self._fg)
        yield ax.switch_to(self._main_thread)

        while True:
            jobs_len = len(self._jobs)

            if self._mx_jobs_count.get() != jobs_len:
                self._mx_jobs_count.set(jobs_len)

                if jobs_len == 0:
                    self._jobs_sps.reset()
                    self._mx_jobs_done_per_sec.set(0.0)

            if len(self._jobs) != 0:
                job = self._jobs[0]
                if job.finished:
                    self._mx_jobs_done_per_sec.set(self._jobs_sps.step())

                    if job.succeeded:
                        self._to_preview(job.result)
                    else:
                        self._mx_error.emit(str(job.error))

                    self._jobs.pop(0)
                    continue

            yield ax.sleep(0.005)

    @ax.task
    def _detect_faces_p(self, image : FImage, resolution : int|None = None,
                                            pad_to_resolution : bool = False,
                                            augment_pyramid : bool = True,
                                            minimum_confidence : float = 0.3,
                                            nms_threshold : float = 0.3,
                                            min_face_size : int = 40,
                                            ) -> Sequence[fd.FFace]:
        yield ax.switch_to(self._thread_pool)

        while self._face_detector is Ellipsis:
            yield ax.sleep(0)

        if (face_detector := self._face_detector) is not None:
            job = face_detector.prepare_job(image, resolution=resolution, pad_to_resolution=pad_to_resolution, augment_pyramid=augment_pyramid,
                                            minimum_confidence=minimum_confidence,
                                            nms_threshold=nms_threshold, min_face_size=min_face_size)

            yield ax.switch_to(self._face_detector_thread)
            return face_detector.detect_job(job)

        return ()

    @ax.task
    def _detect_faces(self, image : FImage) -> Sequence[fd.FFace]:
        yield ax.propagate(self._detect_faces_p(image,
                                                resolution=_DetectorResolution_to_res[ self._mx_detector_resolution.get() ],
                                                pad_to_resolution=# pad_to_resolution only in ImageSequence mode, because images can vary in resolution that cause very slow extraction
                                                                  self._mx_media_source.mx_source_type.get()==MxMediaSource.SourceType.ImageSequence,
                                                augment_pyramid=self._mx_augment_pyramid.get(),
                                                minimum_confidence=self._mx_detector_minimum_confidence.get(),
                                                nms_threshold=self._mx_detector_overlap_threshold.get(),
                                                min_face_size=self._mx_min_face_size.get(),
                                                ))

    @ax.task
    def _filter_faces(self, frame : fd.FFrame, faces : Sequence[fd.FFace]) -> Sequence[fd.FFace]:
        yield ax.switch_to(self._thread_pool)

        sort_by_type = self._mx_sort_by_type.get()
        max_faces = self._mx_max_faces.get()
        max_faces_discard = self._mx_max_faces_discard.get()

        if (faces_len := len(faces)) != 0:
            if max_faces_discard and faces_len > max_faces:
                return ()

            if sort_by_type == MxFacesetMaker.SortBy.Confidence:        faces = fd.FFace.sorted_by_confidence(faces)
            elif sort_by_type == MxFacesetMaker.SortBy.Largest:         faces = fd.FFace.sorted_by_largest(faces)
            elif sort_by_type == MxFacesetMaker.SortBy.DistFromCenter:  faces = fd.FFace.sorted_by_dist_from(faces, FVec2f(frame.image_size) / 2)
            elif sort_by_type == MxFacesetMaker.SortBy.LeftToRight:     faces = fd.FFace.sorted_by_left_to_right(faces)
            elif sort_by_type == MxFacesetMaker.SortBy.RightToLeft:     faces = fd.FFace.sorted_by_right_to_left(faces)
            elif sort_by_type == MxFacesetMaker.SortBy.TopToBottom:     faces = fd.FFace.sorted_by_top_to_bottom(faces)
            elif sort_by_type == MxFacesetMaker.SortBy.BottomToTop:     faces = fd.FFace.sorted_by_bottom_to_top(faces)

            if max_faces != 0:
                faces = faces[:max_faces]

        #Discard, if more than  are found
        # Pass faces 1

        return faces

    @ax.task
    def _mark_face(self, image : FImage, face : fd.FFace) -> fd.FFace:
        """Gather annotations using face markers"""
        yield ax.switch_to(self._thread_pool)

        annotations : Sequence[fd.FAnno] = []

        detector_id = face.detector_id or 'YoloV7Face'
        uni_rect = FRectf(1,1)

        for face_marker in self._face_markers:
            # Calc g_face_rect in global space adjusted for particular face_marker

            # Using face detection rect
            face_align_mat = FAffMat2.estimate(face.rect.inflate_to_square(), uni_rect)

            # optimal adjustings for particular face_marker, found manually
            if isinstance(face_marker, TDDFAV3):
                if detector_id == 'YoloV7Face':
                    marker_coverage = 1.2
                #elif detector_id == 'SCRFD':
                #    marker_coverage = 1.3
            else:
                raise Exception()

            # Calc adjusted g_face_rect
            g_face_rect = uni_rect.transform( FAffMat2().translate(-0.5, -0.5)
                                                        .scale(marker_coverage)
                                                        .translate(0.5, 0.5) ).transform(face_align_mat.inverted)

            # Calc mat to transform g_face_rect to face_marker's space in size of input_size
            input_size = face_marker.input_size
            img_to_align_mat = FAffMat2.estimate(g_face_rect, FRectf(input_size,input_size))

            # warp_affine image
            feed_aligned_face = image.warp_affine(img_to_align_mat, input_size, input_size, border=FImage.Border.REPLICATE)

            yield ax.switch_to(self._face_marker_thread)

            align_to_img_mat = img_to_align_mat.inverted

            # Feed aligned_face to face_marker
            if isinstance(face_marker, TDDFAV3):
                result = face_marker.extract(feed_aligned_face)
                yield ax.switch_to(self._thread_pool)

                annotations += [result.anno_lmrks_ysa_range.transform(align_to_img_mat),
                                result.anno_lmrks_2d68.transform(align_to_img_mat),
                                result.anno_pose]

            # Update Face. New annotations should be placed first
            face = face.set_annotations( fd.FAnnoList(annotations).extend(face.annotations) )

        return face



    @ax.task
    def _align_face(self, face : fd.FFace) -> fd.FAlignedFace|None:
        yield ax.switch_to(self._thread_pool)
        return face.align(  coverage = self._mx_face_coverage.get(),
                            uy_offset = self._mx_face_y_offset.get(),
                            y_axis_offset = self._mx_face_y_axis_offset.get(),
                            min_image_size = self._mx_min_image_size.get(),
                            max_image_size = self._mx_max_image_size.get() )

    @ax.task
    def _identify_face(self, image : FImage, face : fd.FFace) -> fd.FFace:
        """Gather annotations using Face identifier"""
        yield ax.switch_to(self._thread_pool)

        if (face_identifier := self._face_identifier) is not None:

            if (anno_lmrks := face.annotations.get_first_by_class(fd.FAnnoLmrk2DYSARange)) is not None:
                anno_lmrks = anno_lmrks.to_2DYSA(y_axis_offset=0.10)
            else:
                anno_lmrks = face.annotations.get_first_by_class_prio([fd.FAnnoLmrk2D106, fd.FAnnoLmrk2D68, fd.FAnnoLmrk2D])

            input_size = face_identifier.input_size

            # Get align mat
            mat = anno_lmrks.get_align_mat(coverage=1.0, uy_offset=0, output_size=FVec2i(input_size, input_size))

            # Transform image with mat
            feed_face_image = image.warp_affine(mat, input_size, input_size, interp=FImage.Interp.LANCZOS4)

            yield ax.switch_to(self._face_identifier_thread)

            anno_id = face_identifier.extract(feed_face_image)

            # Update Face. New annotations should be placed first
            face = face.set_annotations( fd.FAnnoList([anno_id]).extend(face.annotations)  )

        return face

    @ax.task
    def _parse_face(self, image : FImage, frame : fd.FFrame, face : fd.FFace) -> FParsedFace:
        yield ax.switch_to(self._thread_pool)

        pass_count = self._mx_pass_count.get()
        for n_pass in range(pass_count):
            yield ax.wait(fut := self._mark_face(image, face))
            if not fut.succeeded:
                yield ax.cancel(Exception(f'{frame.media_path}: {fut.error}'))
            face = fut.result

            if n_pass < pass_count-1:
                # Prepare for next pass

                if (anno_lmrks := face.annotations.get_first_by_class(fd.FAnnoLmrk2DYSARange)) is not None:
                    anno_lmrks = anno_lmrks.to_2DYSA(y_axis_offset=0.0)
                else:
                    anno_lmrks = face.annotations.get_first_by_class_prio([ fd.FAnnoLmrk2DYSA, fd.FAnnoLmrk2D106, fd.FAnnoLmrk2D68, fd.FAnnoLmrk2D])

                if isinstance(anno_lmrks, fd.FAnnoLmrk2D):
                    # Get mat from lmrks to align the face with enough coverage and target size
                    size = FVec2i(640, 640)
                    mat = anno_lmrks.get_align_mat(coverage=2.0, uy_offset=-0.14, output_size=size)

                    # Warp frame image
                    feed_face_image = image.warp_affine(mat, size, interp=FImage.Interp.LANCZOS4)

                    # yield ax.switch_to(self._main_thread)
                    # import cv2
                    # cv2.imshow('', feed_face_image.HWC())
                    # cv2.waitKey(100)

                    # Detect faces
                    yield ax.wait(fut := self._detect_faces_p(feed_face_image, augment_pyramid=True))
                    if not fut.succeeded:
                        yield ax.cancel(Exception(f'{frame.media_path}: {fut.error}'))
                    faces = fut.result

                    if len(faces) == 0:
                        # No faces detected, break multipass sequence
                        break

                    # Transform faces back to image space
                    faces = [x.transform(mat.inverted) for x in faces]

                    # Get closest face to the current face by dist from rect centers
                    face = sorted(faces, key=lambda x, pc=face.rect.pc: (x.rect.pc-pc).length)[0]

                    # Do the pass again with new face
                else:
                    # No suitable landmarks - break multipass sequence
                    break

        yield ax.wait(fut := self._identify_face(image, face))
        if not fut.succeeded:
            yield ax.cancel(Exception(f'{frame.media_path}: {fut.error}'))
        face = fut.result

        yield ax.wait(fut := self._align_face(face))
        if not fut.succeeded:
            yield ax.cancel(Exception(f'{frame.media_path}: {fut.error}'))
        aligned_face = fut.result


        p_face = self.FParsedFace(face)
        if aligned_face is not None:
            # Transform image with mat image,
            aligned_face_image = image.warp_affine(aligned_face.mat, aligned_face.image_size, interp=FImage.Interp.LANCZOS4, border=self._mx_border_type.get())
            p_face = p_face.set_aligned_face(aligned_face).set_aligned_face_image(aligned_face_image)

        return p_face


    @ax.task
    def _parse_frame(self, image : FImage, frame : fd.FFrame) -> FParsedFrame:
        yield ax.switch_to(self._thread_pool)
        p_frame = self.FParsedFrame(image, frame)

        # Launch detect faces task
        yield ax.wait(fut := self._detect_faces(image))
        if not fut.succeeded:
            yield ax.cancel(Exception(f'{frame.media_path}: {fut.error}'))
        faces = fut.result
        # Launch filter faces task
        yield ax.wait(fut := self._filter_faces(frame, faces))
        if not fut.succeeded:
            yield ax.cancel(Exception(f'{frame.media_path}: {fut.error}'))

        faces = fut.result
        if len(faces) != 0:
            # parse faces
            yield ax.wait(futs := [self._parse_face(image, frame, face) for face in faces])

            p_faces = []
            for fut in futs:
                if fut.succeeded:
                    p_face = fut.result
                    if p_face.aligned_face is not None and \
                       p_face.aligned_face_image is not None:
                        p_faces.append(p_face)
                else:
                    yield ax.cancel(fut.error)

            # Update frame
            p_frame = p_frame.set_faces(p_faces)

        return p_frame




    @ax.task
    def _export_face(self, p_frame : FParsedFrame, p_face : FParsedFace, name : str, export_path : Path):
        if (aligned_face := p_face.aligned_face) is not None and \
           (aligned_face_image := p_face.aligned_face_image) is not None:
            yield ax.switch_to(self._thread_pool)

            filepath = aligned_face_image.save(export_path / f'{name}.unk',
                                               fmt_type=self._mx_export_file_format.mx_image_format_type.get(),
                                               quality=self._mx_export_file_format.mx_quality.get())

            # Prepare and embed meta
            meta = fd.FEmbedAlignedFaceInfo(aligned_face, p_face.face, p_frame.frame)
            meta.embed_to(filepath)


    @ax.task
    def _export_frame(self, p_frame : FParsedFrame):
        yield ax.switch_to(self._thread_pool)

        if (export_path := self._mx_export_path.get()) is not None and \
           len(p_faces := p_frame.faces) != 0:

            if not export_path.exists():
                export_path.mkdir(parents=True, exist_ok=True)

            frame_name = f'{frame_idx:07}' if (frame_idx := p_frame.frame.frame_idx) is not None else p_frame.frame.media_path.stem
            faces_names = [ f'{frame_name}_{i:02}' for i in range(len(p_faces)) ]

            futs = [ self._export_face(p_frame, p_face, faces_names[i], export_path)
                     for i, p_face in enumerate(p_faces)  ]

            if self._mx_media_source.mx_source_type.get() == MxMediaSource.SourceType.ImageSequence and \
               self._mx_export_dfl_mask.get():

                export_dfl_mask_path = export_path / 'dfl_mask'
                if not export_dfl_mask_path.exists():
                    export_dfl_mask_path.mkdir(parents=True, exist_ok=True)

                if (dfljpg := DFLJPG.load(p_frame.frame.media_path)) is not None:
                    xseg_mask = None

                    if (image := p_frame.image) is not None:

                        if (xseg_mask := dfljpg.get_xseg_mask()) is not None:
                            # Resize to source frame size
                            xseg_mask = FImage.from_numpy(xseg_mask).resize(image.width, image.height)
                        else:
                            # Has no applied xseg mask
                            # Bake from polys
                            if (polys := dfljpg.get_seg_ie_polys()) is not None and polys.has_polys():
                                xseg_mask = np.zeros ((image.height,image.width,1), dtype=np.float32)
                                polys.overlay_mask(xseg_mask)
                                xseg_mask = FImage.from_numpy(xseg_mask)

                    if xseg_mask is not None:

                        for i, p_face in enumerate(p_faces):

                            if (aligned_face := p_face.aligned_face) is not None and \
                               (mat := aligned_face.mat) is not None and \
                               (aligned_face_image := p_face.aligned_face_image) is not None:

                                futs.append( self._export_source_mask(  xseg_mask,
                                                                        mat,
                                                                        aligned_face_image.size,
                                                                        export_dfl_mask_path / f'{faces_names[i]}.png') )

            yield ax.wait(futs)

            for fut in futs:
                if not fut.succeeded:
                    yield ax.cancel(fut.error)

    @ax.task
    def _export_source_mask(self, source_mask : FImage, mat : FAffMat2, size : FVec2i, export_path : Path,):
        yield ax.switch_to(self._thread_pool)
        aligned_mask = source_mask.warp_affine(mat, size)
        aligned_mask.u8().save(export_path)


    @ax.task
    def _process_job(self, image : FImage, frame : fd.FFrame) -> FParsedFrame:
        yield ax.attach_to(self._fg)
        yield ax.switch_to(self._thread_pool)

        yield ax.wait(fut := self._parse_frame(image, frame))
        if not fut.succeeded:
            yield ax.cancel(Exception(f'{frame.media_path}: {fut.error}'))
        p_frame = fut.result

        if self._mx_export_enabled.get():
            # Launch _export_frame task
            yield ax.wait(fut := self._export_frame(p_frame))
            if not fut.succeeded:
                yield ax.cancel(Exception(f'{frame.media_path}: {fut.error}'))

        return p_frame


    @ax.task
    def _on_media_source_frame(self, fr : MxMediaSource.Frame):
        yield ax.attach_to(self._fg)
        yield ax.switch_to(self._main_thread)

        if isinstance(fr, (MxMediaSource.VideoFrame, MxMediaSource.ImageFrame)):
            image = fr.image
            frame = fd.FFrame(image.size).set_media_url(fr.media_path.as_uri())

            if isinstance(fr, MxMediaSource.VideoFrame):
                frame = frame.set_frame_idx(fr.frame_idx)

            self._jobs.append(self._process_job(image, frame))


        while len(self._jobs) >= self._mx_jobs_max.get():
            # MxMediaSource description:
            # new will be emitted ASAP when this Future(Task) is finished
            # thus we finish task immediately, except parallel job count is full
            yield ax.sleep(0.001)





    class FParsedFace:
        def __init__(self, face : fd.FFace):
            self._face = face
            self._aligned_face : fd.FAlignedFace|None = None
            self._aligned_face_image : FImage|None = None

        def clone(self) -> Self:
            f = self.__class__.__new__(self.__class__)
            f._face = self._face
            f._aligned_face = self._aligned_face
            f._aligned_face_image = self._aligned_face_image
            return f

        @property
        def face(self) -> fd.FFace: return self._face
        @property
        def aligned_face(self) -> fd.FAlignedFace|None: return self._aligned_face
        @property
        def aligned_face_image(self) -> FImage|None: return self._aligned_face_image

        def set_face(self, face : fd.FFace) -> Self:                                f = self.clone(); f._face = face; return f
        def set_aligned_face(self, aligned_face : fd.FAlignedFace|None) -> Self:    f = self.clone(); f._aligned_face = aligned_face; return f
        def set_aligned_face_image(self, aligned_face_image : FImage|None) -> Self: f = self.clone(); f._aligned_face_image = aligned_face_image; return f


    class FParsedFrame:
        def __init__(self, image : FImage, frame : fd.FFrame):
            self._image = image
            self._frame = frame
            self._faces : Sequence[MxFacesetMaker.FParsedFace] = ()

        def clone(self) -> Self:
            f = self.__class__.__new__(self.__class__)
            f._image = self._image
            f._frame = self._frame
            f._faces = self._faces
            return f

        @property
        def image(self) -> FImage: return self._image
        @property
        def frame(self) -> fd.FFrame: return self._frame
        @property
        def faces(self) -> Sequence[MxFacesetMaker.FParsedFace]: return self._faces

        def set_image(self, image : FImage) -> Self:                f = self.clone(); f._image = image; return f
        def set_frame(self, frame : fd.FFrame) -> Self:             f = self.clone(); f._frame = frame; return f
        def set_faces(self, faces : Sequence[MxFacesetMaker.FParsedFace]) -> Self: f = self.clone(); f._faces = faces; return f


_DetectorResolution_to_res = {  MxFacesetMaker.DetectorResolution.Source: None,
                                MxFacesetMaker.DetectorResolution._480p: 640,
                                MxFacesetMaker.DetectorResolution._720p: 1280,
                                MxFacesetMaker.DetectorResolution._1080p: 1920, }