from pathlib import Path

from common.FileStateManager import MxFileStateManager
from core import ax, mx
from core.lib.collections import FDict

from .MxDataGenerator import MxDataGenerator
from .MxExport import MxExport
from .MxModel import MxModel
from .MxModelTrainer import MxModelTrainer
from .MxPreview import MxPreview


class MxManager(mx.Disposable):

    def __init__(self, open_path : Path|None = None):
        super().__init__()
        self._fg = ax.FutureGroup().dispose_with(self)
        self._main_thread = ax.get_current_thread()

        self._mx_data_generator : MxDataGenerator = None
        self._mx_model : MxModel = None
        self._mx_model_trainer : MxModelTrainer = None
        self._mx_preview : MxPreview = None

        file_state_mgr = self._mx_file_state_mgr = MxFileStateManager( file_suffix='.dpc',
                                                        on_close=self._on_close,
                                                        on_load=self._on_load,
                                                        get_state=self._get_state).dispose_with(self)
        if open_path is not None:
            file_state_mgr.mx_path.open(open_path)

    @property
    def mx_file_state_manager(self) -> MxFileStateManager: return self._mx_file_state_mgr

    @property
    def mx_data_generator(self) -> MxDataGenerator:
        """avail only when .file_state_manager.mx_state is Initialized """
        return self._mx_data_generator
    @property
    def mx_model(self) -> MxModel:
        """avail only when .file_state_manager.mx_state is Initialized """
        return self._mx_model
    @property
    def mx_model_trainer(self) -> MxModelTrainer:
        """avail only when .file_state_manager.mx_state is Initialized """
        return self._mx_model_trainer
    @property
    def mx_preview(self) -> MxPreview:
        """avail only when .file_state_manager.mx_state is Initialized """
        return self._mx_preview
    @property
    def mx_export(self) -> MxExport:
        """avail only when .file_state_manager.mx_state is Initialized """
        return self._mx_export

    def _on_close(self):
        self._disp_bag.dispose()

    def _on_load(self, state : FDict):
        disp_bag = self._disp_bag = mx.Disposable()

        self._mx_data_generator = MxDataGenerator(state=state.get('data_generator', None)).dispose_with(disp_bag)
        self._mx_model          = MxModel( state=state.get('model', None)).dispose_with(disp_bag)
        self._mx_model_trainer  = MxModelTrainer(self._mx_data_generator, self._mx_model, state=state.get('model_trainer', None)).dispose_with(disp_bag)
        self._mx_preview        = MxPreview(self._mx_data_generator, self._mx_model, state=state.get('preview', None)).dispose_with(disp_bag)
        self._mx_export         = MxExport(self._mx_model, state=state.get('export', None)).dispose_with(disp_bag)

    @ax.task
    def _get_state(self) -> FDict:
        yield ax.attach_to(self._fg)

        model_t = self._mx_model.get_state()
        yield ax.wait(model_t)
        if not model_t.succeeded:
            yield ax.cancel(model_t.error)

        return FDict({'data_generator' : self._mx_data_generator.get_state(),
                      'model' : model_t.result,
                      'model_trainer' : self._mx_model_trainer.get_state(),
                      'preview' : self._mx_preview.get_state(),
                      'export' : self._mx_export.get_state(),
                     })