from common.FileStateManager import QxFileStateManager
from core import mx, qx

from .MxManager import MxManager
from .QxDataGenerator import QxDataGenerator
from .QxExport import QxExport
from .QxModel import QxModel
from .QxModelTrainer import QxModelTrainer
from .QxPreview import QxPreview


class QxManager(qx.QVBox):
    def __init__(self, mgr : MxManager):
        super().__init__()
        self._mgr = mgr

        file_state_manager = mgr.mx_file_state_manager
        file_state_manager.mx_initialized.reflect(lambda initialized, enter, bag=mx.Disposable().dispose_with(self):
                                                   self._ref_file_state_manager_initialized(initialized, enter, bag)).dispose_with(self)

    def _ref_file_state_manager_initialized(self, initialized : bool, enter : bool, bag : mx.Disposable):
        if enter:
            q_file_state_manager = QxFileStateManager(self._mgr.mx_file_state_manager).dispose_with(bag)
            
            if initialized:

                self.add(qx.QSplitter().dispose_with(bag).set_orientation(qx.Orientation.Horizontal)

                    .add(
                            qx.QVBox().set_spacing(4)
                            .add(qx.QHeaderVBox().set_text('@(File_state_manager)').inline(lambda c: c.content_vbox.add(q_file_state_manager)).v_compact())
                            
                            .add(qx.QVScrollArea().set_widget(
                                qx.QVBox().set_spacing(4)
                                    .add(qx.QHeaderVBox().set_text('SRC @(Data_generator)').inline(lambda c: c.content_vbox.add(QxDataGenerator(self._mgr.mx_src_data_generator))).v_compact())
                                    .add(qx.QHeaderVBox().set_text('DST @(Data_generator)').inline(lambda c: c.content_vbox.add(QxDataGenerator(self._mgr.mx_dst_data_generator))).v_compact())
                                    .add(qx.QHeaderVBox().set_text('@(Model)').inline(lambda c: c.content_vbox.add(QxModel(self._mgr.mx_model))).v_compact())
                                    .add(qx.QHeaderVBox().set_text('@(Export)').inline(lambda c: c.content_vbox.add(QxExport(self._mgr.mx_export))).v_compact())
                                    .add(qx.QWidget()))))

                    .add(qx.QSplitter().set_orientation(qx.Orientation.Vertical)
                        .add(qx.QVBox()
                                .add(qx.QHeaderVBox().set_text('@(Trainer)').inline(lambda c: c.content_vbox.add(QxModelTrainer(self._mgr.mx_model_trainer))))
                                .add(qx.QWidget()))
                        
                        .add(QxPreview(self._mgr.mx_preview))
                        
                        )
                        
                        )

            else:
                self.add(q_file_state_manager)

        else:
            bag.dispose_items()