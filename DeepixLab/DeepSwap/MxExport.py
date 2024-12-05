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
    class FileFormat(StrEnum):
        JPEG = 'JPEG'
        JPEG2000 = 'JPEG2000'
        JPEG2000_16 = 'JPEG2000 16-bit'
        PNG = 'PNG'
        PNG_16 = 'PNG 16-bit'
        TIFF_16 = 'TIFF 16-Bit'


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
    def mx_image_format(self) -> MxImageFormat: return self._mx_image_format
    @property
    def mx_fix_borders(self) -> mx.IFlag_v: return self._mx_fix_borders
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

        yield ax.switch_to(self._export_thread)
        yield ax.attach_to(self._export_fg, cancel_all=True)

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

            self._mx_progress.start().set(0, len(image_paths))

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

            self._mx_progress.finish()

        yield ax.switch_to(self._main_thread)

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
                img_fmt = self._mx_image_format.image_format
                quality = self._mx_image_format.quality
                output_path = output_root / f'{image_path.stem}{img_fmt.suffix}'
                H, W, _ = image.shape

                output_image = output_image.resize(W, H, interp=FImage.Interp.LANCZOS4)
                output_image.save(output_path, quality=quality)
            except Exception as e:
                err = e

        if err is not None:
            yield ax.cancel(err)


        #     yield ax.wait(step_task := model.step(MxModel.StepRequest(  dst_image_np=[image_np],
        #                                                                 pred_swap_image=True,
        #                                                                 pred_swap_mask=True,
        #                                                                 pred_swap_enhance=True,
        #                                                                 )))

        #     if step_task.succeeded:
        #         pred_swap_image_np = step_task.result.pred_swap_enhance_np[0] if step_task.result.pred_swap_enhance_np is not None else step_task.result.pred_swap_image_np[0]
        #         pred_swap_mask_np = step_task.result.pred_swap_mask_np[0]
        #     else:
        #         err = step_task.error

        # if err is None:
        #     try:
        #         H, W, _ = image_np.shape

        #         pred_swap_image_np.resize(W, H, interp=FImage.Interp.LANCZOS4).save(output_image_path)
        #         pred_swap_mask_np.resize(W, H, interp=FImage.Interp.LANCZOS4).save(output_mask_path)
        #     except Exception as e:
        #         err = e

        # if err is not None:
        #     yield ax.cancel(err)

