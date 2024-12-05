from core import qx

from .MxModel import MxModel


class QxModel(qx.QVBox):
    def __init__(self, model : MxModel):
        super().__init__()
        self._model = model

        (self
            .add(qx.QGrid().set_spacing(1)
                    .row(0)
                        .add(qx.QLabel().set_text('@(Device)'), align=qx.Align.RightF)
                        .add(qx.QComboBoxMxStateChoice(model.mx_device).set_font(qx.FontDB.FixedWidth))
                    .grid(),
                align=qx.Align.CenterH)

            .add_spacer(8)

            .add(qx.QVBox()
                    .add(qx.QVBox()
                        

                        .add(qx.QHBox()
                            .add(qx.QGrid().set_spacing(1)
                                    .row(0)
                                        .add(qx.QLabel().set_text('@(Swapper)').set_font(qx.FontDB.FixedWidth), col_span=2, align=qx.Align.CenterF)
                                    .next_row()
                                        .add(qx.QLabel().set_text('@(Resolution)'), align=qx.Align.RightF)
                                        .add(qx.QDoubleSpinBoxMxNumber(model.mx_resolution))
                                    .next_row()
                                        .add(qx.QLabel().set_text('@(Dimension)'), align=qx.Align.RightF)
                                        .add(qx.QDoubleSpinBoxMxNumber(model.mx_swapper_dim))
                                    .grid()
                                    , align=qx.Align.TopF)
                            .add_spacer(8)
                            .add(qx.QGrid().set_spacing(1)
                                    .row(0)
                                        .add(qx.QLabel().set_text('@(Enhancer)').set_font(qx.FontDB.FixedWidth), col_span=2, align=qx.Align.CenterF)
                                    .next_row()
                                        .add(qx.QLabel().set_text('@(Resolution)'), align=qx.Align.RightF)
                                        .add(qx.QDoubleSpinBoxMxNumber(model.mx_enhancer_resolution))
                                    .next_row()
                                        .add(qx.QLabel().set_text('@(Depth)'), align=qx.Align.RightF)
                                        .add(qx.QDoubleSpinBoxMxNumber(model.mx_enhancer_depth))
                                    .next_row()
                                        .add(qx.QLabel().set_text('@(Dimension)'), align=qx.Align.RightF)
                                        .add(qx.QDoubleSpinBoxMxNumber(model.mx_enhancer_dim))
                                    .next_row()
                                        .add(qx.QLabel().set_text('GAN @(Dimension)'), align=qx.Align.RightF)
                                        .add(qx.QDoubleSpinBoxMxNumber(model.mx_enhancer_gan_dim))
                                    .next_row()
                                        .add(qx.QLabel().set_text('GAN @(Patch_size) %'), align=qx.Align.RightF)
                                        .add(qx.QDoubleSpinBoxMxNumber(model.mx_enhancer_gan_patch_size_per))
                                    .grid()
                                    , align=qx.Align.TopF)
                                , align=qx.Align.CenterH)

                        .add_spacer(8)
                        
                        .add(qx.QGrid().set_spacing(1)
                                .row(0)
                                    .add(qx.QLabel().set_text('@(Stage)'), align=qx.Align.RightF)
                                    .add(qx.QComboBoxMxStateChoice(model.mx_stage).set_font(qx.FontDB.FixedWidth))
                                .grid()
                                , align=qx.Align.CenterH)

                        .add_spacer(8)

                        .add(qx.QHBox()
                            .add(qx.QPushButton().set_text('@(Current_config)').inline(lambda btn: btn.mx_clicked.listen(lambda: model.revert_model_config())))
                            .add(qx.QPushButton().set_text('@(Apply_config)').inline(lambda btn: btn.mx_clicked.listen(lambda: model.apply_model_config()))))
                        , align=qx.Align.CenterH)

                    .add_spacer(4)
                    
                    .add(qx.QPushButton().set_text('@(Reset) all').inline(lambda btn: btn.mx_clicked.listen(lambda: model.reset_all()))
                        , align=qx.Align.CenterH)
                        
                    .add(qx.QHBox()
                        .add(qx.QPushButton().set_text('@(Reset) encoder').inline(lambda btn: btn.mx_clicked.listen(lambda: model.reset_encoder())))
                        , align=qx.Align.CenterH)
                    
                    .add(qx.QHBox()
                        .add(qx.QPushButton().set_text('@(Reset) inter_src').inline(lambda btn: btn.mx_clicked.listen(lambda: model.reset_inter_src())))
                        .add(qx.QPushButton().set_text('@(Reset) inter_dst').inline(lambda btn: btn.mx_clicked.listen(lambda: model.reset_inter_dst())))
                        , align=qx.Align.CenterH)
                    
                    .add(qx.QHBox()
                        .add(qx.QPushButton().set_text('@(Reset) decoder').inline(lambda btn: btn.mx_clicked.listen(lambda: model.reset_decoder())))
                        .add(qx.QPushButton().set_text('@(Reset) decoder_guide').inline(lambda btn: btn.mx_clicked.listen(lambda: model.reset_decoder_guide())))
                        , align=qx.Align.CenterH)
                        
                    .add(qx.QHBox()
                        .add(qx.QPushButton().set_text('@(Reset) enhancer').inline(lambda btn: btn.mx_clicked.listen(lambda: model.reset_enhancer())))
                        .add(qx.QPushButton().set_text('@(Reset) enhancer gan').inline(lambda btn: btn.mx_clicked.listen(lambda: model.reset_enhancer_gan())))
                        , align=qx.Align.CenterH)
                        
                    , align=qx.Align.CenterH) )