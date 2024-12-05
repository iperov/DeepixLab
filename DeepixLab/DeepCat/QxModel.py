from core import mx, qx

from .MxModel import MxModel


class QxModel(qx.QVBox):
    def __init__(self, model : MxModel):
        super().__init__()
        self._model = model

        self._model_settings_vbox = qx.QVBox()

        (self

            .add(qx.QGrid().set_spacing(1)
                    .row(0)
                        .add(qx.QLabel().set_text('@(Device)'), align=qx.Align.RightF)
                        .add(qx.QComboBoxMxStateChoice(model.mx_device).set_font(qx.FontDB.FixedWidth))
                    .grid(),
                align=qx.Align.CenterH)

            .add(qx.QVBox()
                    .add(self._model_settings_vbox , align=qx.Align.CenterH)
                    .add_spacer(4)
                    .add(qx.QHBox()
                            .add(qx.QPushButton().set_text('@(Current_config)').inline(lambda btn: btn.mx_clicked.listen(lambda: model.revert_model_config())))
                            .add(qx.QPushButton().set_text('@(Apply_config)').inline(lambda btn: btn.mx_clicked.listen(lambda: model.apply_model_config()))) )

                    .add(qx.QPushButton().set_text('@(Reset_model)').inline(lambda btn: btn.mx_clicked.listen(lambda: model.reset_model())))
                    
                    .add(qx.QHBox()
                            .add(qx.QPushButton().set_text('@(Reset) Encoder').inline(lambda btn: btn.mx_clicked.listen(lambda: model.reset_encoder())))
                            .add(qx.QPushButton().set_text('@(Reset) Decoder').inline(lambda btn: btn.mx_clicked.listen(lambda: model.reset_decoder()))))

                    , align=qx.Align.CenterH) )

        model.mx_mode.reflect(lambda mode, enter, bag=mx.Disposable().dispose_with(self):
                                self._ref_mode(mode, enter, bag) )

    def _ref_mode(self, mode : MxModel.Config.Mode, enter, bag : mx.Disposable):
        if enter:
            model = self._model

            self._model_settings_vbox.add(grid := qx.QGrid().dispose_with(bag).set_spacing(1))
            grid_row = (grid
                .row(0)
                    .add(qx.QLabel().set_text('@(Mode)'), align=qx.Align.RightF)
                    .add(qx.QComboBoxMxStateChoice(model.mx_mode).set_font(qx.FontDB.FixedWidth))
                .next_row()
                    .add(qx.QLabel().set_text('@(Input_channel_type)'), align=qx.Align.RightF)
                    .add(qx.QComboBoxMxStateChoice(model.mx_input_channel_type).set_font(qx.FontDB.FixedWidth))
                .next_row()
                    .add(qx.QLabel().set_text('@(Output_channel_type)'), align=qx.Align.RightF)
                    .add(qx.QComboBoxMxStateChoice(model.mx_output_channel_type).set_font(qx.FontDB.FixedWidth))
                .next_row()
                    .add(qx.QLabel().set_text('@(Resolution)'), align=qx.Align.RightF)
                    .add(qx.QDoubleSpinBoxMxNumber(model.mx_resolution), align=qx.Align.LeftF)
                .next_row()
                    .add(qx.QLabel().set_text('@(Base_dimension)'), align=qx.Align.RightF)
                    .add(qx.QDoubleSpinBoxMxNumber(model.mx_base_dim), align=qx.Align.LeftF))

            if mode ==  MxModel.Config.Mode.Segmentator:
                (grid_row.next_row()
                    .add(qx.QLabel().set_text('@(Generalization_level)'), align=qx.Align.RightF)
                    .add(qx.QDoubleSpinBoxMxNumber(model.mx_generalization_level), align=qx.Align.LeftF))

            elif mode ==  MxModel.Config.Mode.Enhancer:
                (grid_row.next_row()
                    .add(qx.QLabel().set_text('@(Depth)'), align=qx.Align.RightF)
                    .add(qx.QDoubleSpinBoxMxNumber(model.mx_depth), align=qx.Align.LeftF))

        else:
            bag.dispose_items()