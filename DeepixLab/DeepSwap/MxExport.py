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
        self._mx_output_swap_path = mx.Path(config=mx.Path.Config(dir=True, allow_open=False, allow_new=True)).dispose_with(self)
        self._mx_output_swap_guide_path = mx.Path(config=mx.Path.Config(dir=True, allow_open=False, allow_new=True)).dispose_with(self)
        self._mx_output_rec_guide_path = mx.Path(config=mx.Path.Config(dir=True, allow_open=False, allow_new=True)).dispose_with(self)

        self._mx_image_format = MxImageFormat(default_format_type=ImageFormatType.PNG, state=state.get('image_format_state', None)).dispose_with(self)

        self._mx_delete_output_directory = mx.Flag(state.get('delete_output_directory', True)).dispose_with(self)

        if (input_path := state.get('input_path', None)) is not None:
            self._mx_input_path.open(input_path)
        if (output_swap_path := state.get('output_swap_path', None)) is not None:
            self._mx_output_swap_path.new(output_swap_path)
        if (output_swap_guide_path := state.get('output_swap_guide_path', None)) is not None:
            self._mx_output_swap_guide_path.new(output_swap_guide_path)
        if (output_rec_guide_path := state.get('output_rec_guide_path', None)) is not None:
            self._mx_output_rec_guide_path.new(output_rec_guide_path)


    def get_state(self) -> FDict:
        return FDict({  'input_path' : self._mx_input_path.get(),
                        'output_swap_path'       : self._mx_output_swap_path.get(),
                        'output_swap_guide_path' : self._mx_output_swap_guide_path.get(),
                        'output_rec_guide_path'  : self._mx_output_rec_guide_path.get(),
                        
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
    def mx_output_swap_path(self) -> mx.IPath_v: return self._mx_output_swap_path
    @property
    def mx_output_swap_guide_path(self) -> mx.IPath_v: return self._mx_output_swap_guide_path
    @property
    def mx_output_rec_guide_path(self) -> mx.IPath_v: return self._mx_output_rec_guide_path
    
    @property
    def mx_image_format(self) -> MxImageFormat: return self._mx_image_format
    @property
    def mx_delete_output_directory(self) -> mx.IFlag_v: return self._mx_delete_output_directory

    def _on_input_path_open(self, path : Path):
        self._mx_output_swap_path.new( path / 'swap' )
        self._mx_output_swap_guide_path.new( path / 'swap_guide' )
        self._mx_output_rec_guide_path.new( path / 'rec_guide' )
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
           (output_swap_path := self._mx_output_swap_path.get()) is None or \
           (output_swap_guide_path := self._mx_output_swap_guide_path.get()) is None or \
           (output_rec_guide_path := self._mx_output_rec_guide_path.get()) is None:
            yield ax.cancel()
      
        yield ax.switch_to(self._export_thread)
        yield ax.attach_to(self._export_fg, cancel_all=True)

        err = None
        if self._mx_delete_output_directory.get():
            if output_swap_path.exists():
                try:
                    shutil.rmtree(output_swap_path)
                except Exception as e:
                    err = e
            
            if output_swap_guide_path.exists():
                try:
                    shutil.rmtree(output_swap_guide_path)
                except Exception as e:
                    err = e
        
            if output_rec_guide_path.exists():
                try:
                    shutil.rmtree(output_rec_guide_path)
                except Exception as e:
                    err = e
                    
        if err is None:
            try:
                output_swap_path.mkdir(parents=True, exist_ok=True)
                output_swap_guide_path.mkdir(parents=True, exist_ok=True)
                output_rec_guide_path.mkdir(parents=True, exist_ok=True)
                image_paths = lib_path.get_files_paths(input_path, extensions=ImageFormatSuffixes)
            except Exception as e:
                err=e

            yield ax.switch_to(self._main_thread)

            self._mx_progress.start().set(0, len(image_paths))

            if err is None:
                for value in  ax.FutureGenerator( ( ( self._infer_path(image_path, output_swap_path, output_swap_guide_path, output_rec_guide_path), image_path )
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
    def _infer_path(self, image_path : Path, output_swap_path : Path, output_swap_guide_path : Path, output_rec_guide_path : Path) -> FImage:
        yield ax.attach_to(self._fg, detach_parent=False)
        yield ax.switch_to(self._main_thread)

        model = self._model
        H, W, _ = model.input_shape

        yield ax.switch_to(self._export_thread_pool)

        err = None
        try:
            image = FImage.from_file(image_path)
        except Exception as e:
            err = e

        if err is None:
        
            yield ax.wait(fut := model.process(MxModel.Job(inputs=MxModel.Job.Inputs(dst_input_image=[image.resize(W, H, interp=FImage.Interp.LANCZOS4)]),
                                                                outputs=MxModel.Job.Outputs(pred_dst_guide=True,
                                                                                            pred_swap_image=True,
                                                                                            pred_swap_enhance=True,
                                                                                            pred_swap_guide=True,
                                                                                            ))))
            if fut.succeeded:
                if (result_outputs := fut.result.result.outputs) is not None:
                    pred_dst_guide = result_outputs.pred_dst_guide[0]
                    pred_swap_guide = result_outputs.pred_swap_guide[0]
                    
                    if result_outputs.pred_swap_enhance is not None:
                        pred_swap_image = result_outputs.pred_swap_enhance[0]
                    else:
                        pred_swap_image = result_outputs.pred_swap_image[0]
                else:
                    err = Exception()
            else:
                err = fut.error


        if err is None:
            try:
                img_fmt = self._mx_image_format.image_format
                quality = self._mx_image_format.quality

                pred_swap_image.save(output_swap_path / f'{image_path.stem}{img_fmt.suffix}', quality=quality)
                pred_swap_guide.save(output_swap_guide_path / f'{image_path.stem}{img_fmt.suffix}', quality=quality)
                pred_dst_guide.save(output_rec_guide_path / f'{image_path.stem}{img_fmt.suffix}', quality=quality)
                
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

