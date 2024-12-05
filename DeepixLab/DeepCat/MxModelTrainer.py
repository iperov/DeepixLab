import itertools

from common.Graph import MxGraph
from core import ax, mx
from core.lib.collections import FDict
from core.lib.time import FPSCounter

from .MxDataGenerator import MxDataGenerator
from .MxModel import MxModel


class MxModelTrainer(mx.Disposable):
    def __init__(self, gen : MxDataGenerator, model : MxModel, state : FDict = None):
        super().__init__()
        state = FDict(state)

        self._main_thread = ax.get_current_thread()
        self._bg_thread = ax.Thread(name='training_thread').dispose_with(self)
        self._fg = ax.FutureGroup().dispose_with(self)
        self._training_fg = ax.FutureGroup().dispose_with(self)

        self._gen = gen
        self._model = model

        self._mx_error = mx.TextEmitter().dispose_with(self)
        self._mx_iteration_time = mx.Property[float](0.0).dispose_with(self)
        self._mx_metrics_graph = MxGraph(state=state.get('metrics_graph_state', None)).dispose_with(self)

        self._mx_batch_size = mx.Number(state.get('batch_size', 16), config=mx.Number.Config(min=1, max=64, step=1)).dispose_with(self)
        self._mx_batch_acc = mx.Number(state.get('batch_acc', 1), config=mx.Number.Config(min=1, max=512, step=1)).dispose_with(self)
        self._mx_learning_rate = mx.Number(state.get('learning_rate', 50), config=mx.Number.Config(min=0, max=250, step=1)).dispose_with(self)
        self._mx_mae_power = mx.Number(state.get('mae_power', 0.0), config=mx.Number.Config(min=0.0, max=1.0, step=0.1)).dispose_with(self)
        self._mx_mse_power = mx.Number(state.get('mse_power', 1.0), config=mx.Number.Config(min=0.0, max=1.0, step=0.1)).dispose_with(self)
        self._mx_dssim_power = mx.Number(state.get('dssim_power', 0.0), config=mx.Number.Config(min=0.0, max=1.0, step=0.1)).dispose_with(self)

        self._mx_training = mx.Flag(state.get('training', False)).dispose_with(self)
        self._mx_training.reflect(self._training_task)

    def get_state(self) -> FDict:
        return FDict(  {'batch_size'        : self._mx_batch_size.get(),
                        'batch_acc'         : self._mx_batch_acc.get(),
                        'learning_rate'     : self._mx_learning_rate.get(),
                        'mae_power'         : self._mx_mae_power.get(),
                        'mse_power'         : self._mx_mse_power.get(),
                        'dssim_power'       : self._mx_dssim_power.get(),
                        'metrics_graph_state' : self._mx_metrics_graph.get_state(),
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
    def mx_mae_power(self) -> mx.INumber_v: return self._mx_mae_power
    @property
    def mx_mse_power(self) -> mx.INumber_v: return self._mx_mse_power
    @property
    def mx_dssim_power(self) -> mx.INumber_v: return self._mx_dssim_power

    @property
    def mx_training(self) -> mx.IFlag_v: return self._mx_training


    @ax.task
    def _training_task(self, _):
        yield ax.switch_to(self._main_thread)
        yield ax.attach_to(self._fg)
        yield ax.attach_to(self._training_fg, max_tasks=1)
        yield ax.switch_to(self._bg_thread)

        @ax.task
        def _send_iteration_time(iteration_time):
            yield ax.switch_to(self._main_thread)
            self._mx_iteration_time.set(iteration_time)

        model = self._model
        gen = self._gen
        data_gen = ax.FutureGenerator( ( (gen.generate( (self._mx_batch_size.get(),)+model.shape ), None)
                                          for _ in itertools.count() ),
                                      max_parallel=gen.workers_count*2, max_buffer=gen.workers_count*2, ordered=False)

        @ax.task
        def train_iter() -> MxModel.Job:
            while (data_gen_value := data_gen.next()) is None:
                yield ax.sleep(0)
            fut, _ = data_gen_value
            if fut.succeeded:
                data = fut.result
                yield ax.propagate(model.process(MxModel.Job(inputs = MxModel.Job.Inputs(
                                                                        input_image = data.input_image,
                                                                        output_target_image = data.target_image, ),

                                                            training = MxModel.Job.Training(
                                                                        mae_power = self._mx_mae_power.get(),
                                                                        mse_power = self._mx_mse_power.get(),
                                                                        dssim_power = self._mx_dssim_power.get(),

                                                                        batch_acc = self._mx_batch_acc.get(),
                                                                        lr = self._mx_learning_rate.get() * 1e-6,  ))) )
            else:
                yield ax.cancel(fut.error)

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

                        self._mx_metrics_graph.add({'@(Error) ' : result_metrics.error,
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
