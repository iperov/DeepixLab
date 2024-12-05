from pathlib import Path

from core import mx

from .MxDatasetEditor import MxDatasetEditor


class MxManager(mx.Disposable):
    def __init__(self, open_path_once : Path|None=None):
        super().__init__()
        self._mx_dataset_editor = MxDatasetEditor(open_path_once=open_path_once).dispose_with(self)

    @property
    def mx_dataset_editor(self) -> MxDatasetEditor: return self._mx_dataset_editor