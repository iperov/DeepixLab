from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Callable, Sequence, Union

from core import ax, mx
from core.lib import path as lib_path
from core.lib.collections import FDict, HFDict, get_enum_id_by_name
from core.lib.ffmpeg import VideoSource
from core.lib.image import FImage, ImageFormatSuffixes

from ..Timeline import MxTimeline


class MxMediaSource(mx.Disposable):
    """
    MediaSource is not a player.

    MediaSource provides frames for processing ASAP.
    """
    class SourceType(StrEnum):
        VideoFile = '@(Video_file)'
        ImageSequence = '@(Image_sequence)'
        #Dataset = '@(Dataset)'

    @dataclass(frozen=True)
    class Frame:
        pass

    @dataclass(frozen=True)
    class VideoFrame(Frame):
        media_path : Path
        frame_idx : int
        image : FImage

    @dataclass(frozen=True)
    class ImageFrame(Frame):
        media_path : Path
        image : FImage


    def __init__(self,  on_frame : Callable[[MxMediaSource.Frame], ax.Future] = None,
                        state : FDict = None):
        """

            on_frame(Frame) -> ax.Future
                Frame is base class of VideoFrame|ImageFrame depends on source_type
                check by isinstance

                When Future is finished, next frame is ready to be emitted soon.
        """
        super().__init__()
        self._on_frame = on_frame
        state = FDict(state)

        self._fg = ax.FutureGroup().dispose_with(self)
        self._main_thread = ax.get_current_thread()
        self._cap_thread = ax.Thread('cap_thread').dispose_with(self)

        self._vid_src : VideoSource = None
        self._imagespaths : Sequence[Path] = None
        #self._fsip : FSIP = None

        self._mx_tl : MxTimeline = None
        self._mx_pix_hdr : mx.Flag = None

        self._mx_error = mx.TextEmitter().dispose_with(self)

        self._mx_media_path : mx.IPath_v = None

        self._mx_preview_image : mx.Property[FImage|None] = None

        self._mx_source_type = mx.StateChoice[MxMediaSource.SourceType](availuator=lambda: [*MxMediaSource.SourceType]).dispose_with(self)
        self._mx_source_type.set( get_enum_id_by_name(MxMediaSource.SourceType, state.get('source_type', None), MxMediaSource.SourceType.VideoFile) )

        self._mx_source_type.reflect(lambda source_type, enter, bag=mx.Disposable().dispose_with(self), state=HFDict(state):
                                     self._ref_mx_source_type(source_type, enter, bag, state=state.pop('mx_source_type', None) if enter else None))

    def get_state(self) -> FDict:
        source_type = self._mx_source_type.get()

        state = { 'source_type' : source_type.name }

        if source_type in [self.SourceType.VideoFile,
                           self.SourceType.ImageSequence,
                           ]:
            state['mx_source_type'] = \
                (mx_source_type_state := { 'media_path' : self._mx_media_path.get() })

            if self._mx_media_path.mx_opened.get():
                mx_source_type_state['mx_media_path'] = \
                    (mx_media_path_state := {'mx_tl' : self._mx_tl.get_state()})

                if source_type == self.SourceType.VideoFile:
                    mx_media_path_state = mx_media_path_state | {'pix_hdr' : self._mx_pix_hdr.get()}

        return FDict(state)

    @property
    def mx_error(self) -> mx.ITextEmitter_v: return self._mx_error
    @property
    def mx_source_type(self) -> mx.IStateChoice_v[SourceType]: return self._mx_source_type
    @property
    def mx_media_path(self) -> mx.IPath_v:
        """avail & renewed when mx_source_type is changed"""
        return self._mx_media_path

    @property
    def mx_timeline(self) -> Union[MxTimeline, None]:
        """Avail when mx_media_path.mx_opened"""
        return self._mx_tl

    @property
    def mx_pix_hdr(self) -> mx.IFlag_v:
        """Avail when mx_media_path.mx_opened and mx_source_type is VideoSource"""
        return self._mx_pix_hdr

    @property
    def mx_preview_image(self) -> mx.IProperty_rv[FImage|None]|None:
        """Avail when mx_media_path.mx_opened"""
        return self._mx_preview_image


    def reemit_current_frame(self):
        if self._mx_media_path.get() is not None:
            if not (mx_tl := self._mx_tl).mx_playing.get():
                mx_tl.mx_frame_idx.set(mx_tl.mx_frame_idx.get())


    def _ref_mx_source_type(self, source_type : SourceType, enter : bool, bag : mx.Disposable, state : FDict|None) -> SourceType:
        state = FDict(state)

        if not enter:
            bag.dispose_items()
        else:
            sub_bag=mx.Disposable().dispose_with(bag)

            if source_type == self.SourceType.VideoFile:

                self._mx_media_path = mx.Path(mx.Path.Config(allow_open=True, extensions=VideoSource.SUPPORTED_SUFFIXES, desc='@(Video_file)'),
                                                on_close=lambda: self._mx_media_path_on_close(sub_bag),
                                                on_open=lambda path, state=HFDict(state): self._mx_media_path_on_open(path, sub_bag, state=state.pop('mx_media_path', None)),
                                              ).dispose_with(bag)

            elif source_type == self.SourceType.ImageSequence:
                self._mx_media_path = mx.Path(mx.Path.Config(allow_open=True, dir=True, desc='@(Image_sequence)'),
                                                on_close=lambda: self._mx_media_path_on_close(sub_bag),
                                                on_open=lambda path, state=HFDict(state): self._mx_media_path_on_open(path, sub_bag, state=state.pop('mx_media_path', None)),
                                                ).dispose_with(bag)

            self._mx_preview_image = mx.Property[FImage|None](None).dispose_with(bag)

            if (media_path := state.get('media_path', None)) is not None:
                self._mx_media_path.open(media_path)


    def _mx_media_path_on_close(self, sub_bag : mx.Disposable):
        sub_bag.dispose_items()
        self._fg.cancel_all()


    def _mx_media_path_on_open(self, path : Path, bag : mx.Disposable, state : FDict|None) -> bool:
        state = FDict(state)
        source_type = self._mx_source_type.get()
        if source_type is MxMediaSource.SourceType.VideoFile:
            err = None
            try:
                vid_src = VideoSource.open(path)
            except Exception as e:
                err = e

            if err is None:
                self._vid_src = vid_src.dispose_with(bag)
                self._mx_tl = MxTimeline(vid_src.frame_count, on_frame_idx=self._tl_on_frame_idx, state=state.get('mx_tl', None)).dispose_with(bag)

                self._mx_pix_hdr = mx.Flag( state.get('pix_hdr', False) ).dispose_with(bag)
                self._mx_pix_hdr.listen(self._on_pix_hdr)
                return True
            else:
                self._mx_error.emit(f'@(Error) VideoSource: {err}')

        elif source_type is MxMediaSource.SourceType.ImageSequence:
            imagespaths = self._imagespaths = lib_path.get_files_paths(path, ImageFormatSuffixes)

            if len(imagespaths) != 0:
                self._mx_tl = MxTimeline(len(imagespaths), on_frame_idx=self._tl_on_frame_idx, state=state.get('mx_tl', None)).dispose_with(bag)

                return True
    
        return False

    @ax.task
    def _on_pix_hdr(self, pix_hdr : bool):
        yield ax.attach_to(self._fg)
        yield ax.switch_to(self._cap_thread)

        self._vid_src.set_pix32(pix_hdr)

        yield ax.switch_to(self._main_thread)

        self.reemit_current_frame()

    @ax.task
    def _update_preview_frame(self, img : FImage):
        yield ax.attach_to(self._fg)
        yield ax.switch_to(self._main_thread)
        if self._mx_media_path.mx_opened.get():
            self._mx_preview_image.set(img)

    @ax.task
    def _tl_on_frame_idx(self, frame_idx : int):
        yield ax.attach_to(self._fg)
        yield ax.switch_to(self._cap_thread)

        err = None
        try:
            source_type = self._mx_source_type.get()
            if source_type is MxMediaSource.SourceType.VideoFile:
                fr = self.VideoFrame(media_path = self._vid_src.path,
                                     frame_idx = frame_idx,
                                     image = self._vid_src.get_frame(frame_idx), )

            elif source_type is MxMediaSource.SourceType.ImageSequence:
                media_path = self._imagespaths[frame_idx]
                fr = self.ImageFrame(media_path = media_path,
                                     image = FImage.from_file(media_path) )


        except Exception as e:
            err = e

        if err is None:
            fut = self._on_frame(fr)

            self._update_preview_frame(fr.image)

            yield ax.wait(fut)
        else:
            yield ax.switch_to(self._main_thread)

            self._mx_error.emit(f'@(Error) MxMediaSource: {err}')
            self._mx_media_path.close()
