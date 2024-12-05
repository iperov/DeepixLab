from __future__ import annotations

import shutil
from enum import StrEnum
from pathlib import Path

from common.ImageFormat import MxImageFormat
from core import ax, mx
from core.lib import path as lib_path
from core.lib.collections import FDict
from core.lib.image import (FImage, ImageFormatSuffixes, ImageFormatType,
                            Patcher)

from .MxModel import MxModel


class MxExport(mx.Disposable):

    def __init__(self, model : MxModel, state : FDict = None):
        super().__init__()
        self._model = model
        state = FDict(state)

        self._fg = ax.FutureGroup().dispose_with(self)
        self._main_thread = ax.get_current_thread()
        self._export_thread = ax.Thread().dispose_with(self)
        self._export_thread_pool = ax.ThreadPool().dispose_with(self)
        self._export_fg = ax.FutureGroup().dispose_with(self)

        self._mx_error = mx.TextEmitter().dispose_with(self)
        self._mx_progress = mx.Progress().dispose_with(self)
        self._mx_input_path  = mx.Path(config=mx.Path.Config(dir=True), on_open=self._on_input_path_open).dispose_with(self)
        self._mx_output_path = mx.Path(config=mx.Path.Config(dir=True, allow_open=False, allow_new=True)).dispose_with(self)

        self._mx_patch_mode = mx.Flag(state.get('patch_mode', False)).dispose_with(self)
        self._mx_sample_count = mx.Number(state.get('sample_count', 2), mx.Number.Config(min=1, max=4)).dispose_with(self)
        self._mx_fix_borders = mx.Flag(state.get('fix_borders', False)).dispose_with(self)

        self._mx_levels_min = mx.Number(state.get('levels_min', 0.0), mx.Number.Config(min=0.0, max=1.0, step=0.01)).dispose_with(self)
        self._mx_levels_min.listen(lambda s: self._mx_levels_max.set(s) if self._mx_levels_max.get() < s else ...)
        self._mx_levels_max = mx.Number(state.get('levels_max', 1.0), mx.Number.Config(min=0.0, max=1.0, step=0.01)).dispose_with(self)
        self._mx_levels_max.listen(lambda s: self._mx_levels_min.set(s) if self._mx_levels_min.get() > s else ...)


        self._mx_image_format = MxImageFormat(default_format_type=ImageFormatType.PNG, state=state.get('image_format_state', None)).dispose_with(self)

        self._mx_delete_output_directory = mx.Flag(state.get('delete_output_directory', True)).dispose_with(self)

        if (input_path := state.get('input_path', None)) is not None:
            self._mx_input_path.open(input_path)
        if (output_image_path := state.get('output_image_path', None)) is not None:
            self._mx_output_path.new(output_image_path)

    def get_state(self) -> FDict:
        return FDict({  'input_path' : self._mx_input_path.get(),
                        'output_path' : self._mx_output_path.get(),
                        'patch_mode' : self._mx_patch_mode.get(),
                        'sample_count' : self._mx_sample_count.get(),
                        'fix_borders' : self._mx_fix_borders.get(),
                        'levels_min' : self._mx_levels_min.get(),
                        'levels_max' : self._mx_levels_max.get(),
                        'image_format_state' : self._mx_image_format.get_state(),
                        'delete_output_directory' : self._mx_delete_output_directory.get(),
                        })

    @property
    def mx_model(self) -> MxModel: return self._model
    @property
    def mx_error(self) -> mx.ITextEmitter_v: return self._mx_error
    @property
    def mx_progress(self) -> mx.IProgress_rv: return self._mx_progress
    @property
    def mx_input_path(self) -> mx.IPath_v: return self._mx_input_path
    @property
    def mx_output_path(self) -> mx.IPath_v: return self._mx_output_path
    @property
    def mx_patch_mode(self) -> mx.IFlag_v: return self._mx_patch_mode
    @property
    def mx_sample_count(self) -> mx.INumber_v: return self._mx_sample_count
    @property
    def mx_fix_borders(self) -> mx.IFlag_v: return self._mx_fix_borders
    @property
    def mx_levels_min(self) -> mx.INumber_v: return self._mx_levels_min
    @property
    def mx_levels_max(self) -> mx.INumber_v: return self._mx_levels_max
    @property
    def mx_image_format(self) -> MxImageFormat: return self._mx_image_format
    @property
    def mx_delete_output_directory(self) -> mx.IFlag_v: return self._mx_delete_output_directory

    def _on_input_path_open(self, path : Path):
        self._mx_output_path.new( path / 'dc' )
        return path


    @ax.task
    def stop(self):
        """
        Stop current export.

        Avail when mx_progress.mx_active == True
        """
        yield ax.switch_to(self._export_thread)
        yield ax.attach_to(self._export_fg, cancel_all=True)
        yield ax.switch_to(self._main_thread)
        self._mx_progress.finish()


    @ax.task
    def start(self):
        """
        Start export.

        Avail when mx_output_path is not None
        """
        yield ax.switch_to(self._main_thread)

        if (input_path  := self._mx_input_path.get()) is None or \
           (output_path := self._mx_output_path.get()) is None:
            yield ax.cancel()
        yield ax.attach_to(self._export_fg, cancel_all=True)

        self._mx_progress.start().set_inf()

        yield ax.switch_to(self._export_thread)

        err = None
        if self._mx_delete_output_directory.get() and output_path.exists():
            try:
                shutil.rmtree(output_path)
            except Exception as e:
                err = e

        if err is None:
            try:
                output_path.mkdir(parents=True, exist_ok=True)
                image_paths = lib_path.get_files_paths(input_path, extensions=ImageFormatSuffixes)
            except Exception as e:
                err=e

            yield ax.switch_to(self._main_thread)

            self._mx_progress.set(0, len(image_paths))

            if err is None:
                for value in  ax.FutureGenerator( ( ( self._infer_path(image_path, output_path), image_path )
                                                    for image_path in image_paths),
                                                    max_parallel=self._export_thread_pool.count*2, max_buffer=self._export_thread_pool.count*2, ):
                    if value is not None:
                        self._mx_progress.inc()

                        fut, image_path = value
                        if fut.succeeded:
                            ...
                        else:
                            err = f'{image_path} : {fut.error}'
                            break
                    else:
                        yield ax.sleep(0)


        yield ax.switch_to(self._main_thread)

        self._mx_progress.finish()

        if err is not None:
            self._mx_error.emit(str(err))

    @ax.task
    def _infer_path(self, image_path : Path, output_root : Path) -> FImage:
        yield ax.attach_to(self._fg, detach_parent=False)
        yield ax.switch_to(self._main_thread)

        model = self._model
        H, W, _, _ = model.shape
        patch_mode = self._mx_patch_mode.get()
        sample_count = self._mx_sample_count.get()
        fix_borders = self._mx_fix_borders.get()

        yield ax.switch_to(self._export_thread_pool)

        err = None
        try:
            image = FImage.from_file(image_path)
        except Exception as e:
            err = e

        if err is None:
            if patch_mode:
                if image.width >= W and image.height >= H:

                    patcher = Patcher(image, W, H, sample_count=sample_count, use_padding=fix_borders)

                    for i in range(patcher.patch_count):
                        yield ax.wait(fut := model.process(MxModel.Job( inputs=MxModel.Job.Inputs(input_image=[patcher.get_patch(i)]),
                                                                        outputs=MxModel.Job.Outputs(output_image=True))))
                        if fut.succeeded:
                            if (result_outputs := fut.result.result.outputs) is not None:
                                patcher.merge_patch(i, result_outputs.output_image[0] )
                            else:
                                err = Exception()
                        else:
                            err = fut.error
                            break

                    if err is None:
                        output_image = patcher.get_merged_image()
                else:
                    err = Exception(f'{image_path} : Image size is less than model resolution')
            else:
                yield ax.wait(fut := model.process(MxModel.Job(inputs=MxModel.Job.Inputs(input_image=[image.resize(W, H, interp=FImage.Interp.LANCZOS4)]),
                                                                    outputs=MxModel.Job.Outputs(output_image=True))))
                if fut.succeeded:
                    if (result_outputs := fut.result.result.outputs) is not None:
                        output_image = result_outputs.output_image[0]
                    else:
                        err = Exception()
                else:
                    err = fut.error


        if err is None:
            try:
                output_image = output_image.resize(image.width, image.height, interp=FImage.Interp.LANCZOS4)

                levels_min, levels_max = self._mx_levels_min.get(), self._mx_levels_max.get()
                if levels_min != 0.0 or levels_max != 1.0:
                    output_image = output_image.f32().levels(in_b=[levels_min], in_w=[levels_max], in_g=[1.0], out_b=[0.0], out_w=[1.0])

                output_image.save(output_root / f'{image_path.stem}.unk',   fmt_type=self._mx_image_format.mx_image_format_type.get(),
                                                                            quality=self._mx_image_format.mx_quality.get())

            except Exception as e:
                err = e

        if err is not None:
            yield ax.cancel(err)

