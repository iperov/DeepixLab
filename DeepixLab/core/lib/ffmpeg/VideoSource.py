from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ... import mx
from ..image import FImage
from .ffmpeg import probe, run


class VideoSource(mx.Disposable):
    SUPPORTED_SUFFIXES = ['.avi','.flv','.mkv','.mp4','.m4v','.mov','.mjpeg','.webm']

    @staticmethod
    def open(path : Path|str, fps = None) -> VideoSource:
        """

            fps(None)   interpret with desired fps
                        default is auto
                        example: 1fps for 10sec video will result 10 frames count

        raises Exception
        """
        path = Path(path)

        if not path.exists():
            raise Exception(f'{path} does not exist.')

        if not path.is_file():
            raise Exception(f'{path} is not a file.')

        if not path.suffix in VideoSource.SUPPORTED_SUFFIXES:
            raise Exception(f'Supported video files: {VideoSource.SUPPORTED_SUFFIXES}')

        # Analyze probe_info
        err = None
        try:
            probe_info = probe(path)
        except Exception as e:
            err = e

        if err is None:
            stream = None
            for s in probe_info['streams']:
                if s['codec_type'] == 'video':
                    stream = s
                    break

            if stream is not None:
                stream_tags = stream.get('tags', None)
                stream_format = probe_info.get('format', None)

                v_idx = 0
                if (v_width := stream.get('width', None)) is not None:
                    v_width = int(v_width)
                if (v_height := stream.get('height', None)) is not None:
                    v_height = int(v_height)

                v_fps = stream.get('avg_frame_rate', None)
                if v_fps is None:
                    v_fps = stream.get('r_frame_rate', None)
                if v_fps is not None:
                    v_fps = eval(v_fps)

                v_start_time = stream.get('start_time', None)
                v_duration = stream.get('duration', None)

                if v_duration is None:
                    if stream_tags is not None:
                        for key, value in stream_tags.items():
                            if 'number_of_frames' in key.lower():
                                v_duration = int(value) / v_fps

                if stream_format is not None:
                    if v_start_time is None:
                        v_start_time = stream_format.get('start_time', None)
                    if v_duration is None:
                        v_duration = stream_format.get('duration', None)
                v_start_time = float(v_start_time) if v_start_time is not None else None
                v_duration = float(v_duration) if v_duration is not None else None
            else:
                err = 'No video tracks.'

            if any( x is None for x in [v_idx, v_width, v_height, v_start_time, v_duration, v_fps] ):#
                err = 'Wrong metadata'

        if err is not None:
            raise Exception(f'Incorrect video file: {err}')

        if fps is None:
            fps = v_fps

        frame_time = 1 / fps
        v_frame_count = int(math.ceil( v_duration / frame_time ) )

        return VideoSource(path, v_idx, v_start_time, v_duration, v_frame_count, v_fps, v_width, v_height, )

    def __init__(self, path : Path, v_idx : int, v_start_time : float, v_duration : float, v_frame_count : int, v_fps : float, v_width : int, v_height : int):
        """use VideoSource.open"""
        super().__init__()

        self._path = path
        self._v_idx = v_idx
        self._v_start_time = v_start_time
        self._v_duration = v_duration
        self._v_frame_count = v_frame_count
        self._v_fps = v_fps
        self._v_width = v_width
        self._v_height = v_height

        self._pix32 = False
        self._ffmpeg_proc = None
        self._frame = -1

    @property
    def frame_count(self) -> int: return self._v_frame_count
    @property
    def path(self) -> Path: return self._path
    @property
    def pix32(self) -> bool: return self._pix32

    def set_pix32(self, pix32 : bool):
        self._pix32 = pix32
        if self._ffmpeg_proc is not None:
            self._ffmpeg_restart()

    def get_frame(self, idx : int) -> FImage:
        #print('get_frame', idx )
        self._ffmpeg_seek_frame(idx)
        if (image := self._ffmpeg_next_frame()) is None:
            image = FImage.zeros(self._v_height, self._v_width, self._ch)
        return image

    def __dispose__(self):
        self._ffmpeg_stop()
        super().__dispose__()


    def _ffmpeg_stop(self) -> str | None:
        """stop and return error log"""
        #print('_ffmpeg_stop at ', self._frame)
        if self._ffmpeg_proc is not None:
            try:
                self._ffmpeg_proc.kill()
            except:
                ...
            self._ffmpeg_proc = None
            return None

            # # Read all errors
            # lines = []
            # if (stderr := self._ffmpeg_proc.stderr) is not None:

            #     import code
            #     code.interact(local=dict(globals(), **locals()))

            #     while not stderr.closed and stderr.readable():
            #         if len(line := stderr.readline()) != 0:
            #             lines.append(line.decode('utf-8').rstrip())
            #         else:
            #             break
            # self._ffmpeg_proc.stderr.close()
            # self._ffmpeg_proc.stdout.close()
            #'\n'.join(lines)
        return None

    def _ffmpeg_restart(self) -> bool:
        #print('_ffmpeg_restart', self._frame )
        self._ffmpeg_stop()

        if self._pix32:
            self._pix_fmt = 'bgr48le'
            self._ch = 3
            self._dtype = np.uint16
            self._ch_bytes = 2
        else:
            self._pix_fmt = 'bgr24'
            self._ch = 3
            self._dtype = np.uint8
            self._ch_bytes = 1

        try:
            args = [#'-loglevel', 'error', #warning
                    '-ss', str(self._v_start_time + self._frame/self._v_fps) ,
                    '-i', str(self._path),
                    '-r', str(self._v_fps),
                    '-fps_mode', 'cfr',
                    '-f', 'rawvideo',
                    '-pix_fmt', self._pix_fmt,
                    '-map', f'0:v:{self._v_idx}',
                    'pipe:' ]

            self._ffmpeg_proc = run(args, pipe_stdout=True,  quiet_stderr=True)
        except:
            ...


    def _ffmpeg_seek_frame(self, idx : int):
        idx = max(0, min(idx, self._v_frame_count-1))

        if self._ffmpeg_proc is not None:
            frame_diff = idx - self._frame
            if frame_diff > 0 and frame_diff <= 120:
                for _ in range(frame_diff):
                    self._ffmpeg_next_frame_bytes()

        if self._frame != idx:
            self._frame = idx
            self._ffmpeg_restart()

    def _ffmpeg_next_frame_bytes(self) -> bytes|None:
        if self._ffmpeg_proc is not None and \
           (self._frame >= 0 and self._frame < self._v_frame_count):

            try:
                buf = self._ffmpeg_proc.stdout.read(self._v_height*self._v_width*self._ch*self._ch_bytes)
            except:
                buf = []

            if len(buf) != 0:
                self._frame += 1
                return buf
            else:
                self._ffmpeg_stop()

        return None

    def _ffmpeg_next_frame(self) -> FImage|None:
        if (buf := self._ffmpeg_next_frame_bytes()) is not None:
            img = np.ndarray( (self._v_height, self._v_width, self._ch), dtype=self._dtype, buffer=buf)
            if img.dtype == np.uint16:
                img = np.divide(img, 65535.0, dtype=np.float32)
            return FImage.from_numpy(img)

        return None

        img = None

        if self._ffmpeg_proc is not None and \
           (self._frame >= 0 and self._frame < self._v_frame_count):

            try:

                buf = self._ffmpeg_proc.stdout.read(self._v_height*self._v_width*self._ch*self._ch_bytes)
                if len(buf) != 0:
                    self._frame += 1

                else:
                    self._ffmpeg_stop()
            except:
                ...








