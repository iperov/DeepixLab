from __future__ import annotations

import dataclasses
from enum import StrEnum
from functools import cached_property
from typing import Sequence, Tuple

import numpy as np
import torch
import torch.amp
import torch.autograd
import torch.linalg
import torch.nn.functional as F

from core import ax, mx
from core.lib import math as lib_math
from core.lib import time as lib_time
from core.lib import torch as lib_torch
from core.lib.collections import FDict, get_enum_id_by_name
from core.lib.functools import cached_method
from core.lib.image import FImage
from core.lib.torch import functional as xF
from core.lib.torch.model import Enhancer, XSeg
from core.lib.torch.modules import PatchDiscriminator
from core.lib.torch.optim import AdaBelief, Optimizer


class Config:
    """Describes model configuration"""

    class Mode(StrEnum):
        Segmentator = '@(Segmentator)'
        Enhancer = '@(Enhancer)'

    class ChannelType(StrEnum):
        Luminance = '@(Luminance)'
        Color = '@(Color)'

    @staticmethod
    def from_state(state : FDict=None) -> Config:
        state = FDict(state)

        config_cls = None
        if (config_cls_type := state.get('config_cls_type', None)) is not None:
            config_cls = globals().get(config_cls_type, None)
        if config_cls is None:
            config_cls= ConfigXSeg

        input_channel_type = get_enum_id_by_name(Config.ChannelType, state.get('input_channel_type', None), Config.ChannelType.Luminance)
        output_channel_type = get_enum_id_by_name(Config.ChannelType, state.get('output_channel_type', None), Config.ChannelType.Luminance)
        resolution = state.get('resolution', 256)
        base_dim = state.get('base_dim', 32)

        kwargs = {  'input_channel_type' : input_channel_type,
                    'output_channel_type' : output_channel_type,
                    'resolution' : resolution,
                    'base_dim' : base_dim, }

        if issubclass(config_cls, ConfigXSeg):
            kwargs['generalization_level'] = state.get('generalization_level', 6)
        elif issubclass(config_cls, ConfigEnhancer ):
            kwargs['depth'] = state.get('depth', 4)

        return config_cls(**kwargs)

    def __init__(self,  input_channel_type : ChannelType = ChannelType.Luminance,
                        output_channel_type : ChannelType = ChannelType.Luminance,
                        resolution : int = 256,
                        base_dim : int = 32,
                    ):
        """
            resolution  [64..1024]:64
            base_dim    [16..256]:8
        """
        self._input_channel_type = input_channel_type
        self._output_channel_type = output_channel_type
        self._resolution = min(max(64, round(resolution / 64) * 64), 1024)
        self._base_dim = min(max(16, round(base_dim / 8) * 8), 256)

    @property
    def input_channel_type(self) -> ChannelType: return self._input_channel_type
    @property
    def output_channel_type(self) -> ChannelType: return self._output_channel_type
    @property
    def resolution(self) -> int: return self._resolution
    @property
    def base_dim(self) -> int: return self._base_dim

    def get_state(self) -> FDict:
        return FDict({  'config_cls_type' : self.__class__.__name__,
                        'input_channel_type' : self._input_channel_type.name,
                        'output_channel_type' : self._output_channel_type.name,
                        'resolution' : self._resolution,
                        'base_dim'   : self._base_dim, })

class ConfigXSeg(Config):
    def __init__(self,  generalization_level : int = 6, **kwargs):
        super().__init__(**kwargs)
        self._generalization_level = min(max(0, generalization_level), 6 )

    @property
    def generalization_level(self) -> int: return self._generalization_level

    def get_state(self) -> FDict: return super().get_state() | FDict({'generalization_level'  : self._generalization_level})

class ConfigEnhancer(Config):
    def __init__(self,  depth : int = 4, **kwargs):
        super().__init__(**kwargs)
        self._depth = min(max(3, depth), 5)

    @property
    def depth(self) -> int: return self._depth

    def get_state(self) -> FDict: return super().get_state() | FDict({'depth'  : self._depth})


class MxModel(mx.Disposable):
    Config = Config

    def __init__(self, state : FDict|None = None):
        super().__init__()
        state = FDict(state)

        # Async things
        self._fg = ax.FutureGroup().dispose_with(self)
        self._main_thread = ax.get_current_thread()
        self._prepare_thread_pool = ax.ThreadPool(name='prepare_thread_pool').dispose_with(self)
        self._model_thread = ax.Thread('model_thread', priority=ax.Thread.Priority.HIGH).dispose_with(self)

        # Model related. Changed in _model_thread.
        self._device = lib_torch.Device.from_state(state.get('device_state', None))
        self._iteration : int = state.get('iteration', 0)
        self._mm = lib_torch.ModuleManager(state.get('mm_state', None))
        self._config : MxModel.Config = None

        # Controls
        self._mx_device = mx.StateChoice[lib_torch.Device](availuator=lambda: [lib_torch.get_cpu_device()]+lib_torch.get_avail_gpu_devices()).dispose_with(self)

        self._mx_mode = mx.StateChoice[MxModel.Config.Mode](availuator=lambda: Config.Mode).dispose_with(self)
        self._mx_mode.listen(lambda mode, enter, bag=mx.Disposable().dispose_with(self):
                                      self._ref_mode(mode, enter, bag)).dispose_with(self)

        self._update_config( MxModel.Config.from_state(state.get('config_state', None)) )
        self.revert_model_config()

    def _ref_mode(self, mode : MxModel.Config.Mode, enter : bool, bag : mx.Disposable):
        if enter:
            config = self._config

            self._mx_input_channel_type  = mx.StateChoice[Config.ChannelType](availuator=lambda: Config.ChannelType).dispose_with(bag)
            self._mx_input_channel_type.set(config.input_channel_type)
            self._mx_output_channel_type = mx.StateChoice[Config.ChannelType](availuator=lambda: Config.ChannelType).dispose_with(bag)
            self._mx_output_channel_type.set(config.output_channel_type)

            self._mx_resolution           = mx.Number(config.resolution, config=mx.Number.Config(min=64, max=1024, step=64)).dispose_with(bag)
            self._mx_base_dim             = mx.Number(config.base_dim, config=mx.Number.Config(min=16, max=256, step=8)).dispose_with(bag)

            if mode == MxModel.Config.Mode.Segmentator:

                self._mx_generalization_level = mx.Number(config.generalization_level if isinstance(config, ConfigXSeg) else 6,
                                                          config=mx.Number.Config(min=0, max=6, step=1) ).dispose_with(bag)

            elif mode == MxModel.Config.Mode.Enhancer:
                self._mx_depth  = mx.Number(config.depth if isinstance(config, ConfigEnhancer) else 4,
                                            config=mx.Number.Config(min=3, max=5, step=1)).dispose_with(bag)
        else:
            bag.dispose_items()

    @ax.task
    def get_state(self) -> FDict:
        yield ax.attach_to(self._fg)
        yield ax.switch_to(self._model_thread)

        return FDict({  'device_state' : self._device.get_state(),
                        'iteration'    : self._iteration,
                        'mm_state'     : self._mm.get_state(),
                        'config_state' : self._config.get_state(), })

    @property
    def mx_device(self) -> mx.IStateChoice_v[lib_torch.Device]: return self._mx_device
    @property
    def mx_mode(self) -> mx.IStateChoice_v[Config.Mode]: return self._mx_mode

    # Below renew on changed mode"""
    @property
    def mx_input_channel_type(self) -> mx.IStateChoice_v[Config.ChannelType]: return self._mx_input_channel_type
    @property
    def mx_output_channel_type(self) -> mx.IStateChoice_v[Config.ChannelType]: return self._mx_output_channel_type
    @property
    def mx_resolution(self) -> mx.INumber_v: return self._mx_resolution
    @property
    def mx_base_dim(self) -> mx.INumber_v: return self._mx_base_dim
    @property
    def mx_depth(self) -> mx.INumber_v:
        """avail if mode == MxModel.Config.Mode.Enhancer"""
        return self._mx_depth
    @property
    def mx_generalization_level(self) -> mx.INumber_v:
        """avail if mode == MxModel.Config.Mode.Segmentator"""
        return self._mx_generalization_level


    @ax.task
    def apply_model_config(self):
        """Apply mx config to actual model."""
        yield ax.attach_to(self._fg)
        yield ax.switch_to(self._main_thread)

        device = self._mx_device.get()

        mode = self._mx_mode.get()
        if mode == Config.Mode.Segmentator:
            config_cls = ConfigXSeg
        elif mode == Config.Mode.Enhancer:
            config_cls = ConfigEnhancer

        kwargs = {  'input_channel_type'  : self._mx_input_channel_type.get(),
                    'output_channel_type' : self._mx_output_channel_type.get(),
                    'resolution'          : self._mx_resolution.get(),
                    'base_dim'            : self._mx_base_dim.get(), }

        if issubclass(config_cls, ConfigXSeg):
            kwargs['generalization_level'] = self._mx_generalization_level.get()
        elif issubclass(config_cls, ConfigEnhancer):
            kwargs['depth'] = self._mx_depth.get()

        config = config_cls(**kwargs)

        yield ax.switch_to(self._model_thread)

        self._device = device
        self._update_config(config)
        self.revert_model_config()

    @ax.task
    def revert_model_config(self):
        """Revert mx config to actual from current model config."""
        yield ax.attach_to(self._fg)
        yield ax.switch_to(self._model_thread)

        device = self._device
        config = self._config

        yield ax.switch_to(self._main_thread)

        self._mx_device.set(device)

        self._mx_mode.set(None)
        if isinstance(config, ConfigXSeg):
            self._mx_mode.set(Config.Mode.Segmentator)
        elif isinstance(config, ConfigEnhancer):
            self._mx_mode.set(Config.Mode.Enhancer)


    def _update_config(self, config : MxModel.Config):
        # in model thread
        old_config, self._config = self._config, config

        # Redefine module factory
        if isinstance(config, ConfigXSeg):
            d = { 'model' : lambda: XSeg(in_ch=_ch_type_to_ch[config.input_channel_type],
                                         out_ch=_ch_type_to_ch[config.output_channel_type],
                                         base_dim=config.base_dim,
                                         depth=6,
                                         generalization_level=config.generalization_level)  }

        elif isinstance(config, ConfigEnhancer):
            d = { 'model' : lambda: Enhancer(in_ch=_ch_type_to_ch[config.input_channel_type],
                                             out_ch=_ch_type_to_ch[config.output_channel_type],
                                             base_dim=config.base_dim,
                                             depth=config.depth),
                  'gan' : lambda: PatchDiscriminator(   in_ch=_ch_type_to_ch[config.output_channel_type],
                                                        out_ch=_ch_type_to_ch[config.output_channel_type],
                                                        patch_size=int((config.resolution*2) * 12.5/100.0),
                                                        base_dim=32,
                                                        max_downs=5),
                                               }

        self._mm.set_module_factory(lambda mm, key: d[key]())
        self._mm.set_optimizer_factory(lambda mm, mod: AdaBelief(mod.parameters()))

        if old_config is not None:
            # Config is changed. Determine which models should be resetted/adjusted.

            is_reset = type(config) != type(old_config) \
                            or config.input_channel_type != old_config.input_channel_type \
                            or config.output_channel_type != old_config.output_channel_type \
                            or config.resolution != old_config.resolution \
                            or config.base_dim != old_config.base_dim

            if isinstance(config, ConfigEnhancer):
                is_reset = is_reset or config.depth != old_config.depth

            if is_reset:
                self.reset_model('model')
            else:
                if isinstance(config, ConfigXSeg):
                    model : XSeg = self._mm.get_module('model')
                    model.set_generalization_level(config.generalization_level)

                    if config.generalization_level < old_config.generalization_level:
                        for i in range(config.generalization_level, old_config.generalization_level):
                            model.reset_shortcut(i)

    @ax.task
    def reset_model(self, *key_or_list):
        yield ax.attach_to(self._fg);
        yield ax.switch_to(self._model_thread);
        self._mm.reset('model', 'gan')

    @ax.task
    def reset_encoder(self):
        yield ax.attach_to(self._fg);
        yield ax.switch_to(self._model_thread);
        model = self._mm.get_module('model')
        model.reset_encoder()

    @ax.task
    def reset_decoder(self):
        yield ax.attach_to(self._fg);
        yield ax.switch_to(self._model_thread);
        model = self._mm.get_module('model')
        model.reset_decoder()

    @dataclasses.dataclass
    class Job:
        @dataclasses.dataclass
        class Inputs:
            input_image : Sequence[FImage]|None = None
            output_target_image : Sequence[FImage]|None = None

            _input_image_nd  : np.ndarray|None = None
            _output_target_image_nd  : np.ndarray|None = None

        @dataclasses.dataclass
        class Outputs:
            output_image : bool = False

        @dataclasses.dataclass
        class Training:
            dssim_power : float = 1.0
            mse_power : float = 1.0

            batch_acc : int = 1
            lr : float = 5e-5
            lr_dropout : float = 0.3

        @dataclasses.dataclass
        class Result:
            @dataclasses.dataclass
            class Outputs:
                output_image : Sequence[FImage]|None = None

            @dataclasses.dataclass
            class Metrics:
                time : float = 0.0
                error : float = 0.0

            outputs : Outputs|None = None
            metrics : Metrics|None = None

        inputs : Inputs
        outputs : Outputs|None = None
        training : Training|None = None
        result : Result = None

        def prepare(self, config : MxModel.Config):
            """Thread-safe data preparation. Can be called from non-model thread."""
            inputs = self.inputs

            in_ch = _ch_type_to_ch[config.input_channel_type]
            out_ch = _ch_type_to_ch[config.output_channel_type]

            inputs._input_image_nd         = np.stack([x.ch(in_ch).f32().CHW() for x in inputs.input_image]) if inputs.input_image is not None else None
            inputs._output_target_image_nd = np.stack([x.ch(out_ch).f32().CHW() for x in inputs.output_target_image]) if inputs.output_target_image is not None else None

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        """returns (height, width, input_ch, output_ch)"""
        config = self._config
        return (config.resolution, config.resolution, _ch_type_to_ch[config.input_channel_type], _ch_type_to_ch[config.output_channel_type])

    @ax.task
    def process(self, job : MxModel.Job) -> MxModel.Job:
        """
        Process job.
        Task errors: Exception

        Depends on parent task.
        Canceling this task will interrupt the job ASAP.
        """
        yield ax.attach_to(self._fg, detach_parent=False)
        yield ax.switch_to(self._prepare_thread_pool)
        config = self._config
        job.prepare(config)
        yield ax.switch_to(self._model_thread)

        try:
            MxModel._JobProcess(job, self, config)
            return job
        except Exception as e:
            yield ax.cancel(e)

    class _JobProcess:
        def __init__(self, job : MxModel.Job, model : MxModel, config : MxModel.Config):
            self.job = job
            self.model = model
            self.config = config

            job_inputs   = job.inputs
            job_outputs  = job.outputs
            job_training = job.training
            job.result = MxModel.Job.Result()
            job.result.metrics = result_metrics = MxModel.Job.Result.Metrics()

            iteration_time = lib_time.measure()

            in_ch = _ch_type_to_ch[config.input_channel_type]
            out_ch = _ch_type_to_ch[config.output_channel_type]

            # Check data
            if not all( self.resolution == x.shape[2] == x.shape[3] for x in [job_inputs._input_image_nd, job_inputs._output_target_image_nd] if x is not None) or \
               not all( in_ch == x.shape[1] for x in [job_inputs._input_image_nd] if x is not None) or \
               not all( out_ch == x.shape[1] for x in [job_inputs._output_target_image_nd] if x is not None):
                raise Exception('@(Invalid_input_data_explained)')

            if job_training is not None:
                # Training, optimization, metrics.
                losses = []
                dis_losses = []

                if (output_target_image_t := self.get_output_target_image_t()) is not None and \
                   (pred_output_image_t := self.get_pred_output_image_t()) is not None:

                    if (mse_power := job_training.mse_power) != 0.0:

                        if isinstance(config, ConfigEnhancer):
                            losses.append( torch.mean(mse_power*5*torch.abs(pred_output_image_t-output_target_image_t), (1,2,3)) )

                            for logit in (logits := self.gan_forward(pred_output_image_t, net_grad=False)):
                                losses.append( 0.1*F.binary_cross_entropy_with_logits(logit, torch.ones_like(logit), reduction='none').mean((1,2,3))/len(logits) )

                            for logit in (logits := self.gan_forward(output_target_image_t)):
                                dis_losses.append( F.binary_cross_entropy_with_logits(logit, torch.ones_like(logit), reduction='none').mean((1,2,3))/len(logits) )

                            for logit in (logits := self.gan_forward(pred_output_image_t.detach())):
                                dis_losses.append( F.binary_cross_entropy_with_logits(logit, torch.zeros_like(logit), reduction='none').mean((1,2,3))/len(logits) )

                        else:
                            losses.append( torch.mean(mse_power*10*torch.square(pred_output_image_t-output_target_image_t), (1,2,3)) )

                if isinstance(config, ConfigXSeg):

                    if (output_target_image_u_t := self.get_output_target_image_u_t()) is not None and \
                       (pred_output_image_u_t := self.get_pred_output_image_u_t()) is not None:
                        for (power, div) in [(job_training.dssim_power, 16),
                                             (job_training.dssim_power, 32), ]:
                            if power != 0.0:
                                losses.append( power*xF.dssim(pred_output_image_u_t, output_target_image_u_t, kernel_size=lib_math.next_odd(self.resolution//div), use_padding=False).mean([-1]) )

                if len(losses) != 0:
                    loss_t = sum(losses)
                    torch.autograd.backward(loss_t, torch.ones_like(loss_t))
                    result_metrics.error = float( loss_t.detach().mean().cpu().numpy() )

                if len(dis_losses) != 0:
                    dis_loss_t = sum(dis_losses)
                    torch.autograd.backward(dis_loss_t, torch.ones_like(dis_loss_t))

                if (model._iteration % job_training.batch_acc) == (job_training.batch_acc-1):
                    for opt in list(self.use_optimizer.get_cached_dict(self).values()):
                        opt.step(iteration=model._iteration, grad_mult=1.0 / job_training.batch_acc, lr=job_training.lr, lr_dropout=job_training.lr_dropout, release_grad=True)
                model._iteration += 1

            if job_outputs is not None:
                # Collect outputs
                job.result.outputs = result_outputs = MxModel.Job.Result.Outputs()

                if job_outputs.output_image:
                    if (pred_output_image_u_t := self.get_pred_output_image_u_t()) is not None:
                        result_outputs.output_image = [ FImage.from_numpy(x, channels_last=False) for x in pred_output_image_u_t.detach().clip(0, 1).cpu().numpy() ]

            # Metrics
            result_metrics.time = iteration_time.elapsed()


        # Cached properties
        @cached_property
        def resolution(self) -> int: return self.model._config.resolution

        # Train specific things?
        @cached_property
        def train_image_target(self) -> bool:   return self.job.training is not None and \
                                                       self.job.inputs._output_target_image_nd is not None

        # Train particular models?
        @cached_property
        def train_model(self) -> bool:  return self.train_image_target

        # Optimizers
        @cached_method
        def use_optimizer(self, key) -> Optimizer:
            opt = self.model._mm.get_optimizer(key)
            if (self.model._iteration % self.job.training.batch_acc) == 0:
                opt.zero_grad()
            return opt

        # Model forwards
        def module_forward(self, name : str, train : bool, inp, net_grad=True) -> torch.Tensor:
            with torch.set_grad_enabled(train):
                model = self.model._mm.get_module(name, device=self.model._device, train=train)
                if train:
                    self.use_optimizer(name)

                if not net_grad:
                    for param in (params := [x for x in model.parameters() if x.requires_grad]):
                        param.requires_grad_(False)

                result = model(inp)

                if not net_grad:
                    for param in params:
                        param.requires_grad_(True)

                return result

        def model_forward(self, x) -> torch.Tensor: return self.module_forward('model', self.train_model, x)

        def gan_forward(self, x, net_grad=True) -> torch.Tensor: return self.module_forward('gan', self.train_model, x, net_grad=net_grad)


        # Cached tensors
        @cached_method
        def get_input_image_t(self) -> torch.Tensor|None:
            x = torch.tensor(job_inputs._input_image_nd, device=self.model._device.device) if (job_inputs := self.job.inputs) is not None and job_inputs._input_image_nd is not None else None
            if x is not None:
                x = x * 2.0 - 1.0
            return x

        @cached_method
        def get_output_target_image_u_t(self) -> torch.Tensor|None:
            return torch.tensor(job_inputs._output_target_image_nd, device=self.model._device.device) if (job_inputs := self.job.inputs) is not None and job_inputs._output_target_image_nd is not None else None

        @cached_method
        def get_output_target_image_t(self) -> torch.Tensor|None:
            return (output_target_image_u_t * 2.0 - 1.0) if (output_target_image_u_t := self.get_output_target_image_u_t()) is not None else None

        @cached_method
        def get_pred_output_image_t(self) -> torch.Tensor|None:
            return self.model_forward(image_t) if (image_t := self.get_input_image_t()) is not None else None

        @cached_method
        def get_pred_output_image_u_t(self) -> torch.Tensor|None:
            return (pred_output_image_t / 2.0 + 0.5) if (pred_output_image_t := self.get_pred_output_image_t()) is not None else None


_ch_type_to_ch = {  MxModel.Config.ChannelType.Luminance : 1,
                    MxModel.Config.ChannelType.Color : 3}

