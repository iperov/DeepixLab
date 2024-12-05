from common.Graph import QxGraph
from core import qx

from .MxModelTrainer import MxModelTrainer


class QxModelTrainer(qx.QVBox):
    def __init__(self, trainer : MxModelTrainer):
        super().__init__()
        self._trainer = trainer
        
        trainer.mx_error.listen(lambda text: wnd.q_info_bar.add_text(f"<span style='color: red'>@(Error)</span>: {text}") if (wnd := self.get_first_parent_by_class(qx.QMainWIndow)) is not None else ...).dispose_with(self)
    
        (self
               
            .add(qx.QHBox().v_compact()
                    
                    .add(qx.QGrid().set_spacing(1)
                            .row(0)
                                .add(qx.QLabel().set_text('@(Train)'), align=qx.Align.RightF)
                                .add(qx.QLabel().set_text('SRC'), align=qx.Align.CenterF)
                                .add(qx.QLabel().set_text('DST'), align=qx.Align.CenterF)
                            .next_row()
                                .add(qx.QLabel().set_text('Image'), align=qx.Align.RightF)
                                .add(qx.QCheckBoxMxFlag(trainer.mx_train_src_image), align=qx.Align.CenterF)
                                .add(qx.QCheckBoxMxFlag(trainer.mx_train_dst_image), align=qx.Align.CenterF)
                            .next_row()
                                .add(qx.QLabel().set_text('Guide'), align=qx.Align.RightF)
                                .add(qx.QCheckBoxMxFlag(trainer.mx_train_src_guide), align=qx.Align.CenterF)
                                .add(qx.QCheckBoxMxFlag(trainer.mx_train_dst_guide), align=qx.Align.CenterF)
                            .next_row()
                                .add(qx.QLabel().set_text('Guide-limited area'), align=qx.Align.RightF)
                                .add(qx.QCheckBoxMxFlag(trainer.mx_src_guide_limited_area), align=qx.Align.CenterF)
                                .add(qx.QCheckBoxMxFlag(trainer.mx_dst_guide_limited_area), align=qx.Align.CenterF)
                            
                            .grid(), align=qx.Align.CenterF)

                    .add_spacer(32)
                    
                    .add(qx.QGrid().set_spacing(1)
                            
                            .row(0)
                                .add(qx.QLabel(), align=qx.Align.RightF, col_span=4) #.set_text('@(Train)')
                            .next_row()
                                .add(qx.QCheckBoxMxFlag(trainer.mx_train_encoder), align=qx.Align.CenterF)
                                .add(qx.QLabel().set_text('Encoder'), align=qx.Align.LeftF)
                            .next_row()
                                .add(qx.QCheckBoxMxFlag(trainer.mx_train_inter_src), align=qx.Align.CenterF)
                                .add(qx.QLabel().set_text('Inter src'), align=qx.Align.LeftF)
                                .add(qx.QCheckBoxMxFlag(trainer.mx_train_inter_dst), align=qx.Align.CenterF)
                                .add(qx.QLabel().set_text('Inter dst'), align=qx.Align.LeftF)
                            .next_row()
                                .add(qx.QCheckBoxMxFlag(trainer.mx_train_decoder), align=qx.Align.CenterF)
                                .add(qx.QLabel().set_text('Decoder'), align=qx.Align.LeftF)
                                .add(qx.QCheckBoxMxFlag(trainer.mx_train_decoder_guide), align=qx.Align.CenterF)
                                .add(qx.QLabel().set_text('Decoder guide'), align=qx.Align.LeftF)
                                    
                            
                            .grid(), align=qx.Align.CenterF)
                    
                    .add_spacer(16)
                    

                    


                    .add(qx.QGrid().set_spacing(1)
                            .row(0)
                                .add(qx.QLabel().set_text('@(Batch_size)'), align=qx.Align.RightF)
                                .add(qx.QDoubleSpinBoxMxNumber(trainer.mx_batch_size))
                            .next_row()
                                .add(qx.QLabel().set_text('@(Batch_acc)'), align=qx.Align.RightF)
                                .add(qx.QDoubleSpinBoxMxNumber(trainer.mx_batch_acc))
                            .next_row()
                                .add(qx.QLabel().set_text('@(Learning_rate)'), align=qx.Align.RightF)
                                .add(qx.QDoubleSpinBoxMxNumber(trainer.mx_learning_rate))
                                .add(qx.QLabel().set_font(qx.FontDB.FixedWidth).set_text('* 1e-6'), align=qx.Align.LeftF)

                            .grid(), align=qx.Align.CenterF)

                    .add_spacer(8)

                    .add(qx.QGrid().set_spacing(1)
                        .row(0)
                            .add(qx.QLabel().set_text('MSE'), align=qx.Align.RightF)
                            .add(qx.QDoubleSpinBoxMxNumber(trainer.mx_mse_power))
                        .next_row()
                            .add(qx.QLabel().set_text('DSSIM'), align=qx.Align.RightF)
                            .add(qx.QDoubleSpinBoxMxNumber(trainer.mx_dssim_power))
                            
                        .grid(), align=qx.Align.CenterF)

                    , align=qx.Align.CenterF)  
                    
            .add_spacer(8)    
               
            .add(qx.QGrid().set_spacing(1).v_compact()
                 .row(0)
                    .add(qx.QLabel().set_text('@(Iteration_time)'), align=qx.Align.RightF)
                    .add(qx.QLabel().set_font(qx.FontDB.Digital)
                            .inline(lambda lbl: trainer.mx_iteration_time.reflect(lambda time: lbl.set_text(f'{time:.3f}')).dispose_with(lbl)))
                            
                    .add(qx.QLabel().set_text('@(second)'), align=qx.Align.LeftF)
                    .grid(),align=qx.Align.CenterH)
                    

            .add(qx.QOnOffPushButtonMxFlag(trainer.mx_training).v_compact()
                    .inline(lambda btn: btn.off_button.set_text('@(Start_training)'))
                    .inline(lambda btn: btn.on_button.set_text('@(Stop_training)')))
                    
            .add_spacer(4)
            
            .add(qx.QHeaderVBox().set_text('@(Metrics)').inline(lambda c: c.content_vbox.add(QxGraph(trainer.mx_metrics_graph).v_expand() ))) #.v_compact(200)
            
            )
        
        
