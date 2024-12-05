from __future__ import annotations

from core import qx

from .MxManager import MxManager
from .QxDatasetEditor import QxDatasetEditor


class QxManager(qx.QVBox):
    def __init__(self, mgr : MxManager):
        super().__init__()
        self.add(QxDatasetEditor(mgr.mx_dataset_editor))

