from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from common.SSI import SSI
from core import ax, mx
from core.lib import path as lib_path
from core.lib.collections import FDict, MxState, get_enum_id_by_name
from core.lib.image import FImage, ImageFormatSuffixes, Patcher

from .MxDataGenerator import MxDataGenerator
from .MxModel import MxModel


class MxPreview(mx.Disposable):
    """
    Manages preview.
    """
    class SourceType(StrEnum):
        DataGenerator = '@(Data_generator)'
        Directory = '@(Directory)'

    def __init__(self, gen : MxDataGenerator,
                       model : MxModel, state : FDict = None):
        super().__init__()
        self._gen = gen
        self._model = model

        self._state = state = MxState(state).dispose_with(self)

        self._main_thread = ax.get_current_thread()
        self._bg_thread = ax.Thread().dispose_with(self)

        self._mx_error = mx.TextEmitter().dispose_with(self)

        self._mx_source_type = mx.StateChoice[MxPreview.SourceType](availuator=lambda: MxPreview.SourceType).dispose_with(self)
        self._mx_source_type.set( get_enum_id_by_name(MxPreview.SourceType, state.get('source_type', None), MxPreview.SourceType.DataGenerator) )

        self._mx_ssi_sheet = mx.Property[SSI.Sheet](SSI.Sheet()).dispose_with(self)
        self._mx_ssi_sheet.set( SSI.Sheet.from_state(state.get('ssi_sheet', None)) )

        state.listen(lambda hfdict: hfdict.update({'source_type' : self._mx_source_type.get().name,
                                                   'ssi_sheet'   : self._mx_ssi_sheet.get().get_state() }) )

        self._mx_source_type.reflect(lambda source_type, enter, sub_bag=mx.Disposable().dispose_with(self),
                                                                sub_state=state.create_child('source_type_state').dispose_with(self):
                                     self._ref_mx_source_type(source_type, enter, sub_bag, sub_state))


    def get_state(self) -> FDict: return self._state.update().to_f_dict()

    @property
    def mx_model(self) -> MxModel: return self._model
    @property
    def mx_error(self) -> mx.ITextEmitter_v: return self._mx_error
    @property
    def mx_ssi_sheet(self) -> mx.IProperty_rv[SSI.Sheet]: return self._mx_ssi_sheet
    @property
    def mx_source_type(self) -> mx.IStateChoice_v[SourceType]: return self._mx_source_type
    @property
    def mx_directory_path(self) -> mx.IPath_v:
        """Avail when mx_source_type == Directory"""
        return self._mx_directory_path
    @property
    def mx_directory_image_idx(self) -> mx.INumber_v:
        """Control current image from directory.

        Avail when `mx_source_type == Directory` and `mx_directory_path is not None`
        """
        return self._mx_directory_image_idx
    @property
    def mx_patch_mode(self) -> mx.IFlag_v:
        """Avail when `mx_source_type == Directory` and `mx_directory_path is not None`"""
        return self._mx_patch_mode
    @property
    def mx_sample_count(self) -> mx.INumber_v:
        """Avail when mx_patch_mode == True"""
        return self._mx_sample_count
    @property
    def mx_fix_borders(self) -> mx.IFlag_v:
        """Avail when mx_patch_mode == True"""
        return self._mx_fix_borders


    def _ref_mx_source_type(self, source_type, enter : bool, bag : mx.Disposable, state : MxState):
        if enter:
            if source_type == self.SourceType.DataGenerator:
                self._data_gen_fg = ax.FutureGroup().dispose_with(bag)

            elif source_type == self.SourceType.Directory:

                sub_bag = mx.Disposable().dispose_with(bag)
                self._mx_directory_path = mx.Path(  config=mx.Path.Config(dir=True),
                                                    on_close=lambda: sub_bag.dispose_items(),
                                                    on_open=lambda path, sub_state=state.create_child('directory_path_state').dispose_with(bag):
                                                                   self._on_directory_path_open(path, sub_bag, sub_state),
                                                    ).dispose_with(bag)

                state.listen(lambda hfdict: hfdict.update({'directory_path' : self._mx_directory_path.get()}) ).dispose_with(bag)

                if (directory_path := state.get('directory_path', None)) is not None:
                    self._mx_directory_path.open(directory_path)

            mx.CallOnDispose(state.update).dispose_with(bag)
        else:
            bag.dispose_items()

    def _on_directory_path_open(self, path : Path, bag : mx.Disposable, state : MxState) -> bool:
        try:
            self._imagespaths = imagespaths = lib_path.get_files_paths(path, extensions=ImageFormatSuffixes)
        except Exception as e:
            self._mx_error.emit(str(e))
            return False

        self._directory_fg = ax.FutureGroup().dispose_with(bag)

        self._mx_directory_image_idx = mx.Number(state.get('directory_image_idx', 0), mx.Number.Config(min=0, max=len(imagespaths)-1)).dispose_with(bag)
        self._mx_directory_image_idx.listen(lambda _: self.update_directory_sample())

        self._mx_patch_mode = mx.Flag(state.get('patch_mode', False)).dispose_with(bag)
        self._mx_patch_mode.listen(lambda _: self.update_directory_sample())

        self._mx_sample_count = mx.Number(state.get('sample_count', 2), mx.Number.Config(min=1, max=4)).dispose_with(bag)
        self._mx_sample_count.listen(lambda _: self.update_directory_sample())

        self._mx_fix_borders = mx.Flag(state.get('fix_borders', False)).dispose_with(bag)
        self._mx_fix_borders.listen(lambda _: self.update_directory_sample())

        state.listen(lambda hfdict: hfdict.update({'directory_image_idx' : self._mx_directory_image_idx.get(),
                                                    'patch_mode' : self._mx_patch_mode.get(),
                                                    'sample_count' : self._mx_sample_count.get(),
                                                    'fix_borders' : self._mx_fix_borders.get(), } ) )
        mx.CallOnDispose(lambda: state.update).dispose_with(bag)

        self._mx_directory_image_idx.reflect(lambda *_: self.update_directory_sample())

        return True

    @ax.task
    def generate_one(self):
        """avail when `mx_source_type == DataGenerator`"""
        yield ax.switch_to(self._main_thread)
        yield ax.attach_to(self._data_gen_fg, max_tasks=1)

        model = self._model
        yield ax.wait(fut := self._gen.generate( (1,)+model.shape, asap=True))

        if not fut.succeeded:
            yield ax.cancel(fut.error)

        gen_result = fut.result

        yield ax.wait(fut := self._model.process(MxModel.Job(   inputs = MxModel.Job.Inputs(input_image=gen_result.input_image),
                                                                outputs = MxModel.Job.Outputs(output_image=True))))

        if not fut.succeeded:
            self._mx_error.emit(str(fut.error))
            yield ax.cancel(fut.error)

        job_result = fut.result.result
        sections = {}
        if (job_outputs := job_result.outputs) is not None:
            sections['image'] = SSI.Grid( { (0,0) : SSI.Image(image=gen_result.input_image[0], caption=gen_result.image_paths[0].name),
                                            (0,1) : SSI.Image(image=job_outputs.output_image[0] if job_outputs.output_image is not None else None, caption=f'{gen_result.image_paths[0].name}'),
                                            } )
        self._mx_ssi_sheet.set(SSI.Sheet(sections=sections))

    @ax.task
    def update_directory_sample(self):
        """
        Avail when `mx_source_type == Directory` and `mx_directory_path is not None`
        """
        yield ax.switch_to(self._main_thread)
        yield ax.attach_to(self._directory_fg, cancel_all=True)

        imagespaths = self._imagespaths
        if len(imagespaths) == 0:
            yield ax.cancel()

        idx = max(0, min(self._mx_directory_image_idx.get(), len(imagespaths)-1))
        patch_mode = self._mx_patch_mode.get()
        sample_count = self._mx_sample_count.get()
        fix_borders = self._mx_fix_borders.get()

        model = self._model
        H,W,IC,_ = model.shape
        imagepath = imagespaths[idx]

        yield ax.switch_to(self._bg_thread)

        err = None
        try:
            image_np = FImage.from_file(imagepath)
        except Exception as e:
            err = e

        if err is None:
            if patch_mode:

                patcher = Patcher(image_np, W, H, sample_count=sample_count, use_padding=fix_borders)

                for i in range(patcher.patch_count):
                    yield ax.wait(job_task := model.process( MxModel.Job(inputs=MxModel.Job.Inputs(input_image=[patcher.get_patch(i).ch(IC)]),
                                                                         outputs=MxModel.Job.Outputs(output_image=True),)))
                    if job_task.succeeded:
                        if (result_outputs := job_task.result.result.outputs) is not None:
                            patcher.merge_patch(i, result_outputs.output_image[0] )
                        else:
                            err = Exception()
                    else:
                        err = job_task.error
                        break

                if err is None:
                    output_image = patcher.get_merged_image()

            else:
                yield ax.wait(job_task := model.process(MxModel.Job(inputs = MxModel.Job.Inputs(input_image=[image_np.resize(W,H, interp=FImage.Interp.LANCZOS4).ch(IC)] ),
                                                                    outputs = MxModel.Job.Outputs(output_image=True))))

                if job_task.succeeded:
                    job_result = job_task.result.result
                    if (job_outputs := job_result.outputs) is not None:
                        output_image = job_outputs.output_image[0]
                    else:
                        err = Exception()
                else:
                    err = job_task.error

        yield ax.switch_to(self._main_thread)

        if err is None:
            sections = {'@(Image)' : SSI.Grid( {  (0,0) : SSI.Image(image=image_np, caption=imagepath.name),
                                               (0,1) : SSI.Image(image=output_image, caption='@(Predicted_image)'), } ),
                        }

            self._mx_ssi_sheet.set(SSI.Sheet(sections=sections))

        else:
            self._mx_error.emit(str(err))
            yield ax.cancel(error=err)