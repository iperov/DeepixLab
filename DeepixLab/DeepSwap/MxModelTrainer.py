import itertools

from common.Graph import MxGraph
from core import ax, mx
from core.lib.collections import FDict
from core.lib.time import FPSCounter

from .MxDataGenerator import MxDataGenerator
from .MxModel import MxModel


class MxModelTrainer(mx.Disposable):
    def __init__(self, src_gen : MxDataGenerator, dst_gen : MxDataGenerator, model : MxModel, state : FDict = None):
        super().__init__()
        state = FDict(state)

        self._main_thread = ax.get_current_thread()
        self._bg_thread = ax.Thread(name='training_thread').dispose_with(self)
        self._fg = ax.FutureGroup().dispose_with(self)
        self._training_fg = ax.FutureGroup().dispose_with(self)

        self._src_gen = src_gen
        self._dst_gen = dst_gen
        self._model = model

        self._mx_error = mx.TextEmitter().dispose_with(self)
        self._mx_iteration_time = mx.Property[float](0.0).dispose_with(self)
        self._mx_metrics_graph = MxGraph(state=state.get('metrics_graph_state', None)).dispose_with(self)

        self._mx_batch_size = mx.Number(state.get('batch_size', 16), config=mx.Number.Config(min=1, max=64, step=1)).dispose_with(self)
        self._mx_batch_acc = mx.Number(state.get('batch_acc', 1), config=mx.Number.Config(min=1, max=512, step=1)).dispose_with(self)
        self._mx_learning_rate = mx.Number(state.get('learning_rate', 50), config=mx.Number.Config(min=0, max=250, step=1)).dispose_with(self)
        self._mx_mse_power = mx.Number(state.get('mse_power', 1.0), config=mx.Number.Config(min=0.0, max=1.0, step=0.1)).dispose_with(self)
        self._mx_dssim_power = mx.Number(state.get('dssim_power', 1.0), config=mx.Number.Config(min=0.0, max=1.0, step=0.1)).dispose_with(self)
        
        self._mx_train_encoder  = mx.Flag(state.get('train_encoder', True)).dispose_with(self)
        self._mx_train_inter_src  = mx.Flag(state.get('train_inter_src', True)).dispose_with(self)
        self._mx_train_inter_dst  = mx.Flag(state.get('train_inter_dst', True)).dispose_with(self)
        self._mx_train_decoder  = mx.Flag(state.get('train_decoder', True)).dispose_with(self)
        self._mx_train_decoder_guide  = mx.Flag(state.get('train_decoder_guide', True)).dispose_with(self)
        
        self._mx_train_src_image  = mx.Flag(state.get('train_src_image', True)).dispose_with(self)
        self._mx_train_src_guide = mx.Flag(state.get('train_src_guide', True)).dispose_with(self)
        self._mx_src_guide_limited_area = mx.Flag(state.get('src_guide_limited_area', True)).dispose_with(self)

        self._mx_train_dst_image  = mx.Flag(state.get('train_dst_image', True)).dispose_with(self)
        self._mx_train_dst_guide = mx.Flag(state.get('train_dst_guide', True)).dispose_with(self)
        self._mx_dst_guide_limited_area = mx.Flag(state.get('dst_guide_limited_area', True)).dispose_with(self)
        
        self._mx_training = mx.Flag(state.get('training', False)).dispose_with(self)
        self._mx_training.reflect(self._training_task)

    def get_state(self) -> FDict:
        return FDict(  {'batch_size'        : self._mx_batch_size.get(),
                        'batch_acc'         : self._mx_batch_acc.get(),
                        'learning_rate'     : self._mx_learning_rate.get(),
                        'mse_power'         : self._mx_mse_power.get(),
                        'dssim_power'       : self._mx_dssim_power.get(),
                        'metrics_graph_state' : self._mx_metrics_graph.get_state(),
                        
                        'train_encoder'      : self._mx_train_encoder.get(),
                        'train_inter_src'    : self._mx_train_inter_src.get(),
                        'train_inter_dst'    : self._mx_train_inter_dst.get(),
                        'train_decoder'      : self._mx_train_decoder.get(),
                        'train_decoder_guide' : self._mx_train_decoder_guide.get(),
                        
                        'train_src_image'   : self._mx_train_src_image.get(),
                        'train_src_guide'    : self._mx_train_src_guide.get(),
                        'src_guide_limited_area' : self._mx_src_guide_limited_area.get(),

                        'train_dst_image'   : self._mx_train_dst_image.get(),
                        'train_dst_guide'    : self._mx_train_dst_guide.get(),
                        'dst_guide_limited_area' : self._mx_dst_guide_limited_area.get(),

                        'training'          : self._mx_training.get(),      })

    @property
    def mx_error(self) -> mx.ITextEmitter_v: return self._mx_error
    @property
    def mx_iteration_time(self) -> mx.IProperty_rv[float]: return self._mx_iteration_time
    @property
    def mx_metrics_graph(self) -> MxGraph: return self._mx_metrics_graph

    @property
    def mx_batch_size(self) -> mx.INumber_v: return self._mx_batch_size
    @property
    def mx_batch_acc(self) -> mx.INumber_v: return self._mx_batch_acc
    @property
    def mx_learning_rate(self) -> mx.INumber_v: return self._mx_learning_rate
    @property
    def mx_mse_power(self) -> mx.INumber_v: return self._mx_mse_power
    @property
    def mx_dssim_power(self) -> mx.INumber_v: return self._mx_dssim_power
  
    @property
    def mx_train_encoder(self) -> mx.IFlag_v: return self._mx_train_encoder
    @property
    def mx_train_inter_src(self) -> mx.IFlag_v: return self._mx_train_inter_src
    @property
    def mx_train_inter_dst(self) -> mx.IFlag_v: return self._mx_train_inter_dst
    @property
    def mx_train_decoder(self) -> mx.IFlag_v: return self._mx_train_decoder
    @property
    def mx_train_decoder_guide(self) -> mx.IFlag_v: return self._mx_train_decoder_guide
     
    @property
    def mx_train_src_image(self) -> mx.IFlag_v: return self._mx_train_src_image
    @property
    def mx_train_src_guide(self) -> mx.IFlag_v: return self._mx_train_src_guide
    @property
    def mx_src_guide_limited_area(self) -> mx.IFlag_v: return self._mx_src_guide_limited_area
    @property
    def mx_train_dst_image(self) -> mx.IFlag_v: return self._mx_train_dst_image
    @property
    def mx_train_dst_guide(self) -> mx.IFlag_v: return self._mx_train_dst_guide
    @property
    def mx_dst_guide_limited_area(self) -> mx.IFlag_v: return self._mx_dst_guide_limited_area
    
    @property
    def mx_training(self) -> mx.IFlag_v: return self._mx_training



    @ax.task
    def _training_task(self, _):
        yield ax.switch_to(self._main_thread)
        if not self._mx_training.get():
            return
        
        yield ax.attach_to(self._fg)
        yield ax.attach_to(self._training_fg, max_tasks=1)
        yield ax.switch_to(self._bg_thread)
        
        @ax.task
        def _send_iteration_time(iteration_time):
            yield ax.switch_to(self._main_thread)
            self._mx_iteration_time.set(iteration_time)

        model = self._model
        src_gen = self._src_gen
        dst_gen = self._dst_gen
        
        src_data_gen = ax.FutureGenerator( ( (src_gen.generate(self._mx_batch_size.get(), model.input_shape, model.target_shape), None)
                                            for _ in itertools.count() ),
                                            max_parallel=src_gen.workers_count*2, max_buffer=src_gen.workers_count*2, ordered=False)
        dst_data_gen = ax.FutureGenerator( ( (dst_gen.generate( self._mx_batch_size.get(), model.input_shape, model.target_shape), None)
                                            for _ in itertools.count() ),
                                            max_parallel=dst_gen.workers_count*2, max_buffer=dst_gen.workers_count*2, ordered=False)
        @ax.task
        def train_iter() -> MxModel.Job:
            train_src_image = self._mx_train_src_image.get()
            train_src_guide = self._mx_train_src_guide.get()
            train_dst_image = self._mx_train_dst_image.get()
            train_dst_guide = self._mx_train_dst_guide.get()
            
            if train_src_image or train_src_guide:
                while (data_gen_value := src_data_gen.next()) is None:
                    yield ax.sleep(0)
                fut, _ = data_gen_value
                if not fut.succeeded:
                    yield ax.cancel(fut.error)
                data = fut.result
                src_input_image  = data.input_image
                src_target_image = data.target_image
                src_target_guide  = data.target_guide
            else:
                src_input_image  = None
                src_target_image = None
                src_target_guide  = None
                
            if train_dst_image or train_dst_guide:
                while (data_gen_value := dst_data_gen.next()) is None:
                    yield ax.sleep(0)
                fut, _ = data_gen_value
                if not fut.succeeded:
                    yield ax.cancel(fut.error)
                data = fut.result
                dst_input_image  = data.input_image
                dst_target_image = data.target_image
                dst_target_guide  = data.target_guide
            else:
                dst_input_image  = None
                dst_target_image = None
                dst_target_guide  = None
                
            yield ax.propagate(model.process(MxModel.Job(inputs = MxModel.Job.Inputs(
                                                                    src_input_image  = src_input_image,
                                                                    src_target_image = src_target_image,
                                                                    src_target_guide  = src_target_guide,
                                                                    dst_input_image  = dst_input_image,
                                                                    dst_target_image = dst_target_image,
                                                                    dst_target_guide  = dst_target_guide,
                                                                    ),

                                                        training = MxModel.Job.Training(
                                                                    train_encoder = self._mx_train_encoder.get(),
                                                                    train_inter_src = self._mx_train_inter_src.get(),
                                                                    train_inter_dst = self._mx_train_inter_dst.get(),
                                                                    train_decoder = self._mx_train_decoder.get(),
                                                                    train_decoder_guide = self._mx_train_decoder_guide.get(),
                                                            
                                                                    train_src_image = train_src_image,
                                                                    train_src_guide = train_src_guide,
                                                                    src_guide_limited_area = self._mx_src_guide_limited_area.get(),

                                                                    train_dst_image = train_dst_image,
                                                                    train_dst_guide = train_dst_guide,
                                                                    dst_guide_limited_area = self._mx_dst_guide_limited_area.get(),

                                                                    dssim_power = self._mx_dssim_power.get(),
                                                                    mse_power = self._mx_mse_power.get(),

                                                                    batch_acc = self._mx_batch_acc.get(),
                                                                    lr = self._mx_learning_rate.get() * 1e-6,  ))) )

        train_gen = ax.FutureGenerator(( (train_iter(), i) for i in itertools.count() ),
                                        max_parallel=2, max_buffer=2)

        eff_iter_fps = FPSCounter(samples=240)

        err = None
        while err is None and self._mx_training.get():

            if (value := train_gen.next()) is not None:
                fut, i = value

                if fut.succeeded:
                    job = fut.result
                    eff_iteration_time = (1.0 / fps) if (fps := eff_iter_fps.step()) != 0 else 0

                    if (result_metrics := job.result.metrics) is not None:
                        iteration_time = result_metrics.time

                        self._mx_metrics_graph.add({'@(Error) src' : result_metrics.error_src,
                                                    '@(Error) dst' : result_metrics.error_dst,
                                                    '@(Iteration_time)' : iteration_time if i >= 2 else 0, # Discard first iteration time because it's too long often
                                                    '@(Effective_iteration_time)' : eff_iteration_time, } )
                        _send_iteration_time(iteration_time)

                else:
                    err = '@(Training_error): ' + str(fut.error)
                    break
            else:
                yield ax.sleep(0)

        yield ax.switch_to(self._main_thread)

        if err is not None:
            self._mx_error.emit(err)

        self._mx_training.set(False)
