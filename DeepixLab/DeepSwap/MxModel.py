from __future__ import annotations

import dataclasses
import math
from enum import StrEnum
from functools import cached_property
from typing import Callable, Sequence, Tuple

import numpy as np
import torch
import torch.amp
import torch.autograd
import torch.linalg
import torch.nn as nn
import torch.nn.functional as F

from core import ax, mx
from core.lib import math as lib_math
from core.lib import time as lib_time
from core.lib import torch as lib_torch
from core.lib.collections import FDict , get_enum_id_by_name
from core.lib.functools import cached_method
from core.lib.image import FImage
from core.lib.torch import functional as xF
from core.lib.torch.init import xavier_uniform, cai
from core.lib.torch.model import Enhancer
from core.lib.torch.modules import PatchDiscriminator
from core.lib.torch.optim import AdaBelief, Optimizer


class Config:
    """Describes model configuration"""

    class Stage(StrEnum):
        Swapper = '@(Swapper)'
        Enhancer = '@(Enhancer)'

    class Swapper:
        def __init__(self, dim : int = 512):
            """
                dim     [32..1024]:8
            """
            self._dim = min(max(32, round(dim / 8) * 8), 1024)

        @property
        def dim(self) -> int: return self._dim

        @staticmethod
        def from_state(state : FDict=None) -> Config.Swapper:
            state = FDict(state)
            return Config.Swapper(dim=state.get('dim', 512),)
        def get_state(self) -> FDict: return FDict({'dim':self._dim})

    class GAN:
        def __init__(self,  base_dim : int = 32,
                            patch_size_per : float = 12.5):
            """
                dim     [16..256]:8
                patch_size_per  [1..100]
            """
            self._base_dim = min(max(16, round(base_dim / 8) * 8), 256)
            self._patch_size_per = min(max(1, patch_size_per), 100)

        @property
        def base_dim(self) -> int: return self._base_dim
        @property
        def patch_size_per(self) -> float: return self._patch_size_per

        @staticmethod
        def from_state(state : FDict=None) -> Config.GAN:
            state = FDict(state)
            return Config.GAN(base_dim=state.get('base_dim', 32),
                                            patch_size_per=state.get('patch_size_per', 12.5), )
        def get_state(self) -> FDict:
            return FDict({'base_dim'      : self._base_dim,
                          'patch_size_per': self._patch_size_per, })

    class Enhancer:
        def __init__(self,  resolution : int = 320,
                            depth : int = 4,
                            base_dim : int = 32,
                            gan : Config.GAN = None, ):
            """
                resolution  [64..640]:18
                depth       [3..5]:1
                base_dim    [16..256]:8
            """
            self._resolution = min(max(64, round(resolution / 8) * 8), 640)
            self._depth = min(max(3, depth), 5)
            self._base_dim = min(max(16, round(base_dim / 8) * 8), 256)
            self._gan = gan or Config.GAN()

        @property
        def resolution(self) -> int: return self._resolution
        @property
        def depth(self) -> int: return self._depth
        @property
        def base_dim(self) -> int: return self._base_dim
        @property
        def gan(self) -> Config.GAN: return self._gan

        @staticmethod
        def from_state(state : FDict=None) -> Config.Enhancer:
            state = FDict(state)
            return Config.Enhancer( resolution=state.get('resolution', 320),
                                    depth=state.get('depth', 4),
                                    base_dim=state.get('base_dim', 32),
                                    gan=Config.GAN.from_state(state.get('gan', {})),
                                    )
        def get_state(self) -> FDict:
            return FDict({  'resolution'    : self._resolution,
                            'depth'         : self._depth,
                            'base_dim'      : self._base_dim,
                            'gan'           : self._gan.get_state(), })


    def __init__(self,  resolution : int = 160,
                        swapper : Swapper = None,
                        enhancer : Enhancer = None,
                        stage : Stage = Stage.Swapper,
                    ):
        """
            resolution  [64..512]:32
        """
        self._resolution = min(max(64, round(resolution / 32) * 32), 512)
        self._swapper = swapper or Config.Swapper()
        self._enhancer = enhancer or Config.Enhancer()
        self._stage = stage

    @property
    def resolution(self) -> int: return self._resolution
    @property
    def swapper(self) -> Swapper: return self._swapper
    @property
    def enhancer(self) -> Config.Enhancer: return self._enhancer
    @property
    def stage(self) -> Stage: return self._stage

    @staticmethod
    def from_state(state : FDict=None) -> Config:
        state = FDict(state)
        
        return Config(  resolution=state.get('resolution', 160),
                        swapper=Config.Swapper.from_state(state.get('swapper', {})),
                        enhancer=Config.Enhancer.from_state(state.get('enhancer', {})),
                        stage=get_enum_id_by_name(Config.Stage, state.get('stage', None), Config.Stage.Swapper),
                        )
    def get_state(self) -> FDict:
        return FDict({  'resolution'  : self._resolution,
                        'swapper'     : self._swapper.get_state(),
                        'enhancer'    : self._enhancer.get_state(),
                        'stage'       : self._stage.name,
                        })
        

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
        self._config : Config = None
        self._update_config( Config.from_state(state.get('config_state', None)) )
        
        # Controls
        self._mx_device = mx.StateChoice[lib_torch.Device](availuator=lambda: lib_torch.get_avail_devices()).dispose_with(self)
        
        self._mx_resolution              = mx.Number(self._config.resolution, config=mx.Number.Config(min=64, max=320, step=32)).dispose_with(self)
        self._mx_swapper_dim             = mx.Number(self._config.swapper.dim, config=mx.Number.Config(min=32, max=1024, step=8)).dispose_with(self)
        self._mx_enhancer_resolution     = mx.Number(self._config.enhancer.resolution, config=mx.Number.Config(min=64, max=640, step=8)).dispose_with(self)
        self._mx_enhancer_depth          = mx.Number(self._config.enhancer.depth, config=mx.Number.Config(min=3, max=5, step=1)).dispose_with(self)
        self._mx_enhancer_dim            = mx.Number(self._config.enhancer.base_dim, config=mx.Number.Config(min=16, max=256, step=8)).dispose_with(self)
        self._mx_enhancer_gan_dim        = mx.Number(self._config.enhancer.gan.base_dim, config=mx.Number.Config(min=8, max=256, step=8)).dispose_with(self)
        self._mx_enhancer_gan_patch_size_per = mx.Number(self._config.enhancer.gan.patch_size_per, config=mx.Number.Config(min=1, max=100, step=0.5)).dispose_with(self)
        self._mx_stage = mx.StateChoice[Config.Stage](availuator=lambda: Config.Stage).dispose_with(self)

        self.revert_model_config()


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
    def mx_resolution(self) -> mx.INumber_v: return self._mx_resolution
    @property
    def mx_swapper_dim(self) -> mx.INumber_v: return self._mx_swapper_dim
    @property
    def mx_enhancer_resolution(self) -> mx.INumber_v: return self._mx_enhancer_resolution
    @property
    def mx_enhancer_depth(self) -> mx.INumber_v: return self._mx_enhancer_depth
    @property
    def mx_enhancer_dim(self) -> mx.INumber_v: return self._mx_enhancer_dim
    @property
    def mx_enhancer_gan_dim(self) -> mx.INumber_v: return self._mx_enhancer_gan_dim
    @property
    def mx_enhancer_gan_patch_size_per(self) -> mx.INumber_v: return self._mx_enhancer_gan_patch_size_per
    @property
    def mx_stage(self) -> mx.IStateChoice_v[Config.Stage]: return self._mx_stage

    @ax.task
    def apply_model_config(self):
        """Apply mx config to actual model."""
        yield ax.attach_to(self._fg)
        yield ax.switch_to(self._main_thread)

        device = self._mx_device.get()
        config = MxModel.Config(resolution  = self._mx_resolution.get(),
                                swapper     = MxModel.Config.Swapper(dim=self._mx_swapper_dim.get()),
                                enhancer    = MxModel.Config.Enhancer(  resolution=self._mx_enhancer_resolution.get(),
                                                                        depth=self._mx_enhancer_depth.get(),
                                                                        base_dim=self._mx_enhancer_dim.get(),
                                                                        gan=MxModel.Config.GAN( base_dim=self._mx_enhancer_gan_dim.get(),
                                                                                                patch_size_per=self._mx_enhancer_gan_patch_size_per.get())),
                                stage = self._mx_stage.get())

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
        self._mx_resolution.set(config.resolution)
        self._mx_swapper_dim.set(config.swapper.dim)
        self._mx_enhancer_resolution.set(config.enhancer.resolution)
        self._mx_enhancer_depth.set(config.enhancer.depth)
        self._mx_enhancer_dim.set(config.enhancer.base_dim)
        self._mx_enhancer_gan_dim.set(config.enhancer.gan.base_dim)
        self._mx_enhancer_gan_patch_size_per.set(config.enhancer.gan.patch_size_per)
        self._mx_stage.set(config.stage)



    def _update_config(self, config : Config):
        # in model thread
        old_config, self._config = self._config, config
        
        # Redefine module factory
        in_ch = 3
        out_ch = 3
        out_guide_ch = 3
        base_dim = 64
        n_layers = 5
        inter_res = 7

        def get_layer_resolution():
            return [ round(self._config.resolution / math.pow(self._config.resolution/inter_res, 1/n_layers)**i) for i in range(n_layers+1) ]
        
        # Uniformly base_dim -> swapper_dim
        ch_mod  = math.pow(config.swapper.dim/base_dim, 1/n_layers)
        layer_encoder_dim = [ round(base_dim * ch_mod**i) for i in range(n_layers+1) ]
        # Modify by sin(x*pi/2)
        layer_encoder_dim = [ x * ( 1.0 + math.sin( (i/n_layers)*math.pi/2 ) - (i/n_layers) )  for i, x in enumerate(layer_encoder_dim) ]
        # Fix to nearest 8
        layer_encoder_dim = [ int(8*round(x/8.0)) for x in layer_encoder_dim ]

        layer_decoder_dim = layer_encoder_dim[::-1]
        layer_decoder_guide_dim = [ int(8*round( (x/3) / 8.0)) for x in layer_decoder_dim]

        inter_in_ch = layer_encoder_dim[-1]
        inter_out_ch = max(layer_decoder_dim[0], layer_decoder_guide_dim[0])

        layer_decoder_dim[0] = inter_out_ch*2
        layer_decoder_guide_dim[0] = inter_out_ch*2
        
        class SimpleAtten(nn.Module):
            def __init__(self, ch):
                super().__init__()
                self._c0 = nn.Conv2d(ch, ch, 3, 1, 1)
                self._c1 = nn.Conv2d(ch, ch, 3, 1, 1)

            def forward(self, inp):
                a = F.leaky_relu(self._c0(inp), 0.1)
                a = F.leaky_relu(self._c1(a), 0.1)

                _, _, H, W = a.size()
                d = (a - a.mean(dim=[2,3], keepdim=True)).pow(2)
                a = d / (4 * (d.sum(dim=[2,3], keepdim=True) / (W * H - 1) + 1e-4)) + 0.5

                return inp*torch.sigmoid(a)
            
        class ResidualBlock(nn.Module):
            def __init__(self, ch, mid_ch = None, atten=False):
                """emb should match mid_ch"""
                super().__init__()
                if mid_ch is None:
                    mid_ch = ch

                self._c0 = nn.Conv2d(ch, mid_ch, 3, 1, 1)
                self._c1 = nn.Conv2d(mid_ch, ch, 3, 1, 1)
                self._atten = SimpleAtten(ch) if atten else None

            def forward(self, inp, emb=None):
                x = inp
                x = self._c0(x)
                if emb is not None:
                    x = x + emb
                x = F.leaky_relu(x, 0.2)
                x = self._c1(x)
                if self._atten is not None:
                    x = self._atten(x)
                x = F.leaky_relu(x + inp, 0.2)
                return x

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self._in = nn.Conv2d(in_ch, layer_encoder_dim[0], 1, 1, 0)
                down_conv_list = self._down_conv_list = nn.ModuleList()
                for up_ch, down_ch in list(zip(layer_encoder_dim[:-1], layer_encoder_dim[1:])):
                    down_conv_list.append( nn.Conv2d(up_ch, down_ch, 5, 1, 2) )
                cai(self) 
                
            def forward(self, inp : torch.Tensor):
                x = inp * 2.0 - 1.0
                x = self._in(x)

                for down_conv, down_res in zip(self._down_conv_list, get_layer_resolution()[1:]):#, self._down_r_list
                    x = down_conv(x)
                    x = F.interpolate(x, (down_res,down_res), mode='bilinear', align_corners=True)
                    x = F.leaky_relu(x, 0.1)

                return x #* (x.square().mean(dim=[1,2,3], keepdim=True) + 1e-06).rsqrt()

        class Inter(nn.Module):
            def __init__(self):
                super().__init__()
                self._fc1 = nn.Linear(inter_in_ch*inter_res*inter_res, config.swapper.dim)
                self._fc2 = nn.Linear(config.swapper.dim, inter_out_ch*inter_res*inter_res)
                xavier_uniform(self)
                

            def forward(self, inp):
                x = inp
                x = x.reshape(-1, inter_in_ch*inter_res*inter_res)
                x = self._fc1(x)
                
                # https://arxiv.org/pdf/1912.10233.pdf
                # Latent Variables on Spheres for Autoencoders in High Dimensions
                x = x - x.sum(-1, keepdim=True) / x.shape[-1]
                x = x / torch.sqrt( (x**2).sum(-1, keepdim=True) )
                
                x = self._fc2(x)
                x = x.reshape(-1, inter_out_ch,inter_res,inter_res)
                return x

        class Decoder(nn.Module):
            def __init__(self, out_ch, layers_dim, use_residuals : bool = True):
                super().__init__()

                up_conv_list = self._up_conv_list = nn.ModuleList()
                residual_list = self._residual_list = nn.ModuleList()
                for down_ch, up_ch in list(zip(layers_dim[:-1], layers_dim[1:])):
                    up_conv_list.append(nn.Conv2d(down_ch, up_ch, 5, 1, 2))
                    residual_list.append(ResidualBlock(up_ch, mid_ch=up_ch*2) if use_residuals else nn.Identity())

                self._out = nn.Conv2d(layers_dim[-1], out_ch, 1, 1, 0)
                xavier_uniform(self)
                                         
            def forward(self, inp):
                x = inp
                for up_res, up_conv, residual in zip(get_layer_resolution()[-2::-1], self._up_conv_list, self._residual_list):
                    x = F.interpolate(x, (up_res, up_res), mode='bilinear', align_corners=True)
                    x = F.leaky_relu(up_conv(x), 0.1)
                    x = residual(x)

                return self._out(x)
            
        d = {   'encoder'       : lambda: Encoder(),
                'inter_src'     : lambda: Inter(),
                'inter_dst'     : lambda: Inter(),
                'decoder'       : lambda: Decoder(out_ch, layer_decoder_dim),
                'decoder_guide' : lambda: Decoder(out_guide_ch, layer_decoder_guide_dim), 
                
                'enhancer'      : lambda: Enhancer(in_ch=out_ch, out_ch=out_ch, base_dim=config.enhancer.base_dim, depth=config.enhancer.depth),
                'enhancer_gan'  : lambda: PatchDiscriminator(   in_ch=out_ch,
                                                                out_ch=1,
                                                                patch_size=int(config.enhancer.resolution * config.enhancer.gan.patch_size_per/100.0),
                                                                base_dim=config.enhancer.gan.base_dim,
                                                                max_downs=5), 
            }

        self._mm.set_module_factory(lambda mm, key: d[key]())
        self._mm.set_optimizer_factory(lambda mm, mod: AdaBelief(mod.parameters()))
        
        if old_config is not None:
            # Config is changed. Determine which models should be resetted/adjusted.
            reset_encoder = config.swapper.dim != old_config.swapper.dim

            reset_inter = reset_encoder
            reset_decoder = reset_inter
            reset_decoder_guide = reset_inter

            reset_enhancer = config.enhancer.depth != old_config.enhancer.depth \
                          or config.enhancer.base_dim != old_config.enhancer.base_dim

            reset_enhancer_gan = reset_enhancer or config.enhancer.resolution != old_config.enhancer.resolution \
                                                or config.enhancer.gan.base_dim != old_config.enhancer.gan.base_dim \
                                                or config.enhancer.gan.patch_size_per != old_config.enhancer.gan.patch_size_per

            if reset_encoder:      self.reset_encoder()
            if reset_inter:        self.reset_inter_src(); self.reset_inter_dst()
            if reset_decoder:      self.reset_decoder()
            if reset_decoder_guide: self.reset_decoder_guide()
            if reset_enhancer:     self.reset_enhancer()
            if reset_enhancer_gan: self.reset_enhancer_gan()
            
    @ax.task
    def reset_all(self): self.reset_encoder();self.reset_inter_src();self.reset_inter_dst();self.reset_decoder();self.reset_decoder_guide();self.reset_enhancer();self.reset_enhancer_gan()
    @ax.task
    def reset_encoder(self):        yield ax.attach_to(self._fg); yield ax.switch_to(self._model_thread); self._mm.reset('encoder')
    @ax.task
    def reset_inter_src(self):      yield ax.attach_to(self._fg); yield ax.switch_to(self._model_thread); self._mm.reset('inter_src')
    @ax.task
    def reset_inter_dst(self):      yield ax.attach_to(self._fg); yield ax.switch_to(self._model_thread); self._mm.reset('inter_dst')
    @ax.task
    def reset_decoder(self):        yield ax.attach_to(self._fg); yield ax.switch_to(self._model_thread); self._mm.reset('decoder')
    @ax.task
    def reset_decoder_guide(self):   yield ax.attach_to(self._fg); yield ax.switch_to(self._model_thread); self._mm.reset('decoder_guide')
    @ax.task
    def reset_enhancer(self):       yield ax.attach_to(self._fg); yield ax.switch_to(self._model_thread); self._mm.reset('enhancer')
    @ax.task
    def reset_enhancer_gan(self):   yield ax.attach_to(self._fg); yield ax.switch_to(self._model_thread); self._mm.reset('enhancer_gan')


    @dataclasses.dataclass
    class Job:
 
        @dataclasses.dataclass
        class Inputs:
            src_input_image : Sequence[FImage]|None = None
            src_target_image : Sequence[FImage]|None = None
            src_target_guide : Sequence[FImage]|None = None
            
            dst_input_image : Sequence[FImage]|None = None
            dst_target_image : Sequence[FImage]|None = None
            dst_target_guide : Sequence[FImage]|None = None

        @dataclasses.dataclass
        class Outputs:
            pred_src_image : bool = False 
            pred_src_guide : bool = False
            pred_dst_image : bool = False
            pred_dst_guide : bool = False
            pred_swap_image : bool = False
            pred_swap_guide : bool = False
            pred_src_enhance : bool = False
            pred_swap_enhance : bool = False

        @dataclasses.dataclass
        class Training:
            train_encoder : bool = True
            train_inter_src : bool = True
            train_inter_dst : bool = True
            train_decoder : bool = True
            train_decoder_guide : bool = True
            
            train_src_image : bool = True
            train_src_guide : bool = True
            src_guide_limited_area : bool = True

            train_dst_image : bool = True
            train_dst_guide : bool = True
            dst_guide_limited_area : bool = True

            dssim_power : float = 1.0
            mse_power : float = 1.0

            batch_acc : int = 1
            lr : float = 5e-5
            lr_dropout : float = 0.3

        @dataclasses.dataclass
        class Result:
            @dataclasses.dataclass
            class Outputs:
                pred_src_image : Sequence[FImage]|None = None
                pred_src_guide : Sequence[FImage]|None = None
                pred_dst_image : Sequence[FImage]|None = None
                pred_dst_guide : Sequence[FImage]|None = None
                pred_swap_image : Sequence[FImage]|None = None
                pred_swap_guide : Sequence[FImage]|None = None
                pred_src_enhance : Sequence[FImage]|None = None
                pred_swap_enhance : Sequence[FImage]|None = None

            @dataclasses.dataclass
            class Metrics:
                time : float = 0.0
                error_src : float = 0.0
                error_dst : float = 0.0

            outputs : Outputs|None = None
            metrics : Metrics|None = None

        inputs : Inputs
        outputs : Outputs|None = None
        training : Training|None = None
        result : Result = None



    @property
    def input_shape(self) -> Tuple[int, int, int]: 
        """input shape [H,W,C]"""
        config = self._config
        resolution = config.resolution
        return (resolution, resolution, 3)
    
    @property
    def target_shape(self) -> Tuple[int, int, int]:
        """[H,W,C]"""
        config = self._config
        resolution = config.resolution
        if config.stage == Config.Stage.Enhancer:
            resolution = config.enhancer.resolution
        return (resolution, resolution, 3)


    @ax.task
    def process(self, job : MxModel.Job) -> MxModel.Job:
        """
        Process job.
        Task errors: Exception

        Depends on parent task.
        Canceling this task will interrupt the job ASAP.
        """
        yield ax.attach_to(self._fg, detach_parent=False)
        yield ax.switch_to(self._model_thread)

        try:
            MxModel._JobProcess(job, self, self._config)
            return job
        except Exception as e:
            yield ax.cancel(e)

    class _JobProcess:
        def __init__(self, job : MxModel.Job, model : MxModel, config : Config):
            self.job = job
            self.model = model
            self.config = config

            job_inputs   = job.inputs
            job_outputs  = job.outputs
            job_training = job.training
            job.result = MxModel.Job.Result()
            job.result.metrics = result_metrics = MxModel.Job.Result.Metrics()

            iteration_time = lib_time.measure()

            if job_training is not None:
                # Training, optimization, metrics.
                src_losses = []
                dst_losses = []
                dis_losses = []
                
                if config.stage == MxModel.Config.Stage.Swapper:
                    
                    if self.train_src_image:
                        if (src_target_masked_image_t := self.get_src_target_image_t(masked=job_training.src_guide_limited_area)) is not None and \
                           (pred_src_masked_image_t := self.get_pred_src_image_t(masked=job_training.src_guide_limited_area)) is not None:
                            for (power, div) in [ (job_training.dssim_power, 16),
                                                  (job_training.dssim_power, 32), ]:
                                if power != 0.0:
                                    src_losses.append( power*xF.dssim(pred_src_masked_image_t, src_target_masked_image_t, kernel_size=lib_math.next_odd(self.resolution//div), use_padding=False).mean([-1]) )

                            if (mse_power := job_training.mse_power) != 0.0:
                                src_losses.append( torch.mean(mse_power*10*torch.square(pred_src_masked_image_t-src_target_masked_image_t), (1,2,3)) )
                    
                    if self.train_dst_image:
                        if (dst_target_masked_image_t := self.get_dst_target_image_t(masked=job_training.dst_guide_limited_area)) is not None and \
                           (pred_dst_masked_image_t := self.get_pred_dst_image_t(masked=job_training.dst_guide_limited_area)) is not None:

                            for (power, div) in [ (job_training.dssim_power, 16),
                                                  (job_training.dssim_power, 32), ]:
                                if power != 0.0:
                                    dst_losses.append( power*xF.dssim(pred_dst_masked_image_t, dst_target_masked_image_t, kernel_size=lib_math.next_odd(self.resolution//div), use_padding=False).mean([-1]) )

                            if (mse_power := job_training.mse_power) != 0.0:
                                dst_losses.append( torch.mean(mse_power*10*torch.square(pred_dst_masked_image_t-dst_target_masked_image_t), (1,2,3)) )
                    
                    if self.train_src_guide:
                        if (pred_src_guide_t := self.get_pred_src_guide_t()) is not None and \
                           (src_target_guide_t := self.get_src_target_guide_t(res=pred_src_guide_t.shape[2])) is not None:
                                src_losses.append( torch.mean(10*torch.square(pred_src_guide_t-src_target_guide_t), (1,2,3)) )

                    if self.train_dst_guide:
                        if (pred_dst_guide_t := self.get_pred_dst_guide_t()) is not None and \
                           (dst_target_guide_t := self.get_dst_target_guide_t(res=pred_dst_guide_t.shape[2])) is not None:
                                dst_losses.append( torch.mean(10*torch.square(pred_dst_guide_t-dst_target_guide_t), (1,2,3)) )

                elif model._config.stage == Config.Stage.Enhancer:
                    if  (src_target_image_t := self.get_src_target_image_t(masked=False)) is not None and \
                        (pred_src_image_t   := self.get_pred_src_image_t(masked=False, res=src_target_image_t.shape[2], detach=True)) is not None:

                            if job_training.src_guide_limited_area and (src_target_mask_blur_t := self.get_src_target_mask_blur_t(res=src_target_image_t.shape[2])) is not None:
                                src_target_image_t = src_target_image_t*src_target_mask_blur_t + pred_src_image_t*(1-src_target_mask_blur_t)

                            pred_src_enhanced_image_t = self.enhancer_forward(pred_src_image_t)

                            src_losses.append( torch.mean( 5*torch.abs(pred_src_enhanced_image_t-src_target_image_t), (1,2,3)) )

                            for logit in (logits := self.enhancer_gan_forward(pred_src_enhanced_image_t, net_grad=False)):
                                src_losses.append( 0.1*F.binary_cross_entropy_with_logits(logit, torch.ones_like(logit), reduction='none').mean((1,2,3))/len(logits) )

                            for logit in (logits := self.enhancer_gan_forward(src_target_image_t)):
                                dis_losses.append( F.binary_cross_entropy_with_logits(logit, torch.ones_like(logit), reduction='none').mean((1,2,3))/len(logits) )

                            for logit in (logits := self.enhancer_gan_forward(pred_src_enhanced_image_t.detach())):
                                dis_losses.append( F.binary_cross_entropy_with_logits(logit, torch.zeros_like(logit), reduction='none').mean((1,2,3))/len(logits) )

                if len(src_losses) != 0:
                    src_loss_t = sum(src_losses)
                    torch.autograd.backward(src_loss_t, torch.ones_like(src_loss_t))
                    result_metrics.error_src = float( src_loss_t.detach().mean().cpu().numpy() )
                
                if len(dst_losses) != 0:
                    dst_loss_t = sum(dst_losses)
                    torch.autograd.backward(dst_loss_t, torch.ones_like(dst_loss_t))
                    result_metrics.error_dst = float( dst_loss_t.detach().mean().cpu().numpy() )

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

                if job_outputs.pred_src_image:
                    if (pred_src_image_t := self.get_pred_src_image_t(masked=False)) is not None:
                        result_outputs.pred_src_image = [ FImage.from_numpy(x) for x in pred_src_image_t.detach().permute(0,2,3,1).clip(0, 1).cpu().numpy() ]

                if job_outputs.pred_src_guide:
                    if (pred_src_guide_t := self.get_pred_src_guide_t()) is not None:
                        result_outputs.pred_src_guide = [ FImage.from_numpy(x) for x in pred_src_guide_t.detach().permute(0,2,3,1).clip(0, 1).cpu().numpy() ]

                if job_outputs.pred_dst_image:
                    if (pred_dst_image_t := self.get_pred_dst_image_t(masked=False)) is not None:
                        result_outputs.pred_dst_image = [ FImage.from_numpy(x) for x in pred_dst_image_t.detach().permute(0,2,3,1).clip(0, 1).cpu().numpy() ]

                if job_outputs.pred_dst_guide:
                    if (pred_dst_guide_t := self.get_pred_dst_guide_t()) is not None:
                        result_outputs.pred_dst_guide = [ FImage.from_numpy(x) for x in pred_dst_guide_t.detach().permute(0,2,3,1).clip(0, 1).cpu().numpy() ]

                if job_outputs.pred_swap_image:
                    if (pred_swap_image_t := self.get_pred_swap_image_t()) is not None:
                        result_outputs.pred_swap_image = [ FImage.from_numpy(x) for x in pred_swap_image_t.detach().permute(0,2,3,1).clip(0, 1).cpu().numpy() ]

                if job_outputs.pred_swap_guide:
                    if (pred_swap_guide_t := self.get_pred_swap_guide_t()) is not None:
                        result_outputs.pred_swap_guide = [ FImage.from_numpy(x) for x in pred_swap_guide_t.detach().permute(0,2,3,1).clip(0, 1).cpu().numpy() ]
                
                if config.stage == MxModel.Config.Stage.Enhancer:
                    if job_outputs.pred_src_enhance:
                        if (pred_src_enhanced_image_t := self.get_pred_src_enhanced_image_t()) is not None:
                            result_outputs.pred_src_enhance = [ FImage.from_numpy(x) for x in pred_src_enhanced_image_t.detach().permute(0,2,3,1).clip(0, 1).cpu().numpy() ]

                    if job_outputs.pred_swap_enhance:
                        if (pred_swap_enhanced_image_t := self.get_pred_swap_enhanced_image_t()) is not None:
                            result_outputs.pred_swap_enhance = [ FImage.from_numpy(x) for x in pred_swap_enhanced_image_t.detach().permute(0,2,3,1).clip(0, 1).cpu().numpy() ]

            # Metrics
            result_metrics.time = iteration_time.elapsed()


        # Cached properties
        @cached_property
        def resolution(self) -> int: return self.config.resolution
        @cached_property
        def enhancer_resolution(self) -> int: return self.config.enhancer.resolution
        @cached_property
        def target_resolution(self) -> int: return self.enhancer_resolution if self.config.stage == MxModel.Config.Stage.Enhancer else self.resolution

        # Train specific things?
        @cached_property
        def train_src_image(self) -> bool:   return self.config.stage == MxModel.Config.Stage.Swapper and (job_training := self.job.training) is not None and job_training.train_src_image
        @cached_property
        def train_src_guide(self) -> bool:    return self.config.stage == MxModel.Config.Stage.Swapper and (job_training := self.job.training) is not None and job_training.train_src_guide 
        @cached_property
        def train_src_enhance(self) -> bool: return self.config.stage == MxModel.Config.Stage.Enhancer and (job_training := self.job.training) is not None and job_training.train_src_image
        @cached_property
        def train_dst_image(self) -> bool:   return self.config.stage == MxModel.Config.Stage.Swapper and (job_training := self.job.training) is not None and job_training.train_dst_image
        @cached_property
        def train_dst_guide(self) -> bool:    return self.config.stage == MxModel.Config.Stage.Swapper and (job_training := self.job.training) is not None and job_training.train_dst_guide 

        # Train particular models?
        @cached_property
        def train_encoder(self) -> bool:      return (job_training := self.job.training) is not None and job_training.train_encoder and (self.train_src_image or self.train_dst_image or self.train_src_guide or self.train_dst_guide)
        @cached_property
        def train_inter_src(self) -> bool:    return (job_training := self.job.training) is not None and job_training.train_inter_src and (self.train_src_image or self.train_src_guide)
        @cached_property
        def train_inter_dst(self) -> bool:    return (job_training := self.job.training) is not None and job_training.train_inter_dst and (self.train_dst_image or self.train_dst_guide)
        @cached_property
        def train_decoder(self) -> bool:      return (job_training := self.job.training) is not None and job_training.train_decoder and (self.train_src_image or self.train_dst_image)
        @cached_property
        def train_decoder_guide(self) -> bool: return (job_training := self.job.training) is not None and job_training.train_decoder_guide and (self.train_src_guide or self.train_dst_guide)
        @cached_property
        def train_enhancer(self) -> bool:     return self.train_src_enhance
        """
        @cached_property
        def grad_encoder(self) -> bool:      return (job_training := self.job.training) is not None and job_training.train_encoder
        @cached_property
        def grad_inter_src(self) -> bool:    return (job_training := self.job.training) is not None and job_training.train_inter_src
        @cached_property
        def grad_inter_dst(self) -> bool:    return (job_training := self.job.training) is not None and job_training.train_inter_dst
        @cached_property
        def grad_decoder(self) -> bool:      return (job_training := self.job.training) is not None and job_training.train_decoder
        @cached_property
        def grad_decoder_guide(self) -> bool: return (job_training := self.job.training) is not None and job_training.train_decoder_guide
        """
       
        # Optimizers
        @cached_method
        def use_optimizer(self, key) -> Optimizer:
            opt = self.model._mm.get_optimizer(key)
            if (self.model._iteration % self.job.training.batch_acc) == 0:
                opt.zero_grad()
            return opt

        # Common utilities
        def image_seq_to_tensor(self, seq : Sequence[FImage], C_ok : Sequence[int]=None, H_ok : Sequence[int]=None, W_ok : Sequence[int]=None) -> torch.Tensor:
            # Load images to device asap. Doing all necessary transformations on device. raise on error
            try:
                x = torch.stack([ torch.tensor(seq[i].HWC(), device=self.model._device.device) for i in range(len(seq)) ]).permute(0,3,1,2)
            except:
                raise Exception('@(Invalid_input_data_explained)')
            if x.dtype == torch.uint8:
                x = x.type(torch.float32) / 255.0
            _,C,H,W = x.shape
            if (C_ok is not None and C not in C_ok) or \
               (H_ok is not None and H not in H_ok) or \
               (W_ok is not None and W not in W_ok):
                raise Exception('@(Invalid_input_data_explained)')
            return x
        
        def with_module(self, name : str, train : bool, func : Callable, net_grad=True) -> torch.Tensor:
            #with torch.set_grad_enabled(train):
            
            model = self.model._mm.get_module(name, device=self.model._device, train=train)
            if train:
                self.use_optimizer(name)

            if not net_grad:
                for param in (params := [x for x in model.parameters() if x.requires_grad]):
                    param.requires_grad_(False)

            result = func(model)

            if not net_grad:
                for param in params:
                    param.requires_grad_(True)

            return result
            
        def encoder_forward(self, x) -> torch.Tensor: return self.with_module('encoder', self.train_encoder, lambda encoder: encoder(x))
        def inter_src_forward(self, x) -> torch.Tensor: return self.with_module('inter_src', self.train_inter_src, lambda inter_src: inter_src(x))
        def inter_dst_forward(self, x) -> torch.Tensor: return self.with_module('inter_dst', self.train_inter_dst, lambda inter_dst: inter_dst(x))
        def decoder_forward(self, x) -> torch.Tensor: return self.with_module('decoder', self.train_decoder, lambda decoder: decoder(x))
        def decoder_guide_forward(self, x) -> torch.Tensor: return self.with_module('decoder_guide', self.train_decoder_guide, lambda decoder_guide: decoder_guide(x))
        def enhancer_forward(self, x) -> torch.Tensor: return self.with_module('enhancer', self.train_enhancer, lambda enhancer: enhancer(x))
        def enhancer_gan_forward(self, x, net_grad=True) -> torch.Tensor: return self.with_module('enhancer_gan', self.train_enhancer, lambda enhancer_gan: enhancer_gan(x), net_grad=net_grad)


        # Cached tensors
        @cached_method
        def get_src_input_image_t(self) -> torch.Tensor|None: return self.image_seq_to_tensor(src_input_image, [3], [self.resolution], [self.resolution]) if (job_inputs := self.job.inputs) is not None and (src_input_image := job_inputs.src_input_image) is not None else None
        @cached_method
        def get_src_target_guide_t_0(self) -> torch.Tensor|None: return self.image_seq_to_tensor(src_target_guide, [3], [self.target_resolution], [self.target_resolution]) if (job_inputs := self.job.inputs) is not None and (src_target_guide := job_inputs.src_target_guide) is not None else None
        
        @cached_method
        def get_src_target_guide_t(self, res : int = None) -> torch.Tensor|None:
            if (src_target_guide_t := self.get_src_target_guide_t_0()) is not None and res is not None and res != src_target_guide_t.shape[2]:
                src_target_guide_t = F.interpolate(src_target_guide_t, size=res, mode='bilinear', align_corners=True)
            return src_target_guide_t
        
        @cached_method
        def get_src_target_mask_t(self, res : int = None) -> torch.Tensor|None:
            if (src_target_guide_t := self.get_src_target_guide_t(res=res)) is not None:
                # guide to gray
                
                gray = torch.tensor([[[[0.1140]],[[0.5870]],[[0.2990]]]], dtype=torch.float32, device=src_target_guide_t.device)
                gray = (src_target_guide_t * gray).sum(1, keepdims=True)
                # mask from gray >= 0.5
                return (gray >= 0.5).type(torch.float32)
            return None
        
        @cached_method
        def get_src_target_mask_blur_t(self, res : int) -> torch.Tensor|None:
            if (src_target_mask_t := self.get_src_target_mask_t(res=res)) is not None:
                src_target_mask_t = torch.clamp(xF.gaussian_blur(src_target_mask_t, sigma=max(1, src_target_mask_t.shape[2] // 32) ), 0.0, 0.5) * 2.0
            return src_target_mask_t
        @cached_method
        def get_src_target_image_t_0(self) -> torch.Tensor|None: return self.image_seq_to_tensor(src_target_image, [3], [self.target_resolution], [self.target_resolution]) if (job_inputs := self.job.inputs) is not None and (src_target_image := job_inputs.src_target_image) is not None else None
        @cached_method
        def get_src_target_image_t(self, masked : bool) -> torch.Tensor|None:
            if (src_target_image_t := self.get_src_target_image_t_0()) is not None and masked:
                if (src_target_mask_blur_t := self.get_src_target_mask_blur_t(res=src_target_image_t.shape[2])) is not None:
                    src_target_image_t = src_target_image_t * src_target_mask_blur_t
            return src_target_image_t
        @cached_method
        def get_dst_input_image_t(self) -> torch.Tensor|None: return self.image_seq_to_tensor(dst_input_image, [3], [self.resolution], [self.resolution]) if (job_inputs := self.job.inputs) is not None and (dst_input_image := job_inputs.dst_input_image) is not None else None
        
        @cached_method
        def get_dst_target_guide_t_0(self) -> torch.Tensor|None: return self.image_seq_to_tensor(dst_target_guide, [3], [self.target_resolution], [self.target_resolution]) if (job_inputs := self.job.inputs) is not None and (dst_target_guide := job_inputs.dst_target_guide) is not None else None
        @cached_method
        def get_dst_target_guide_t(self, res : int = None) -> torch.Tensor|None:
            if (dst_target_guide_t := self.get_dst_target_guide_t_0()) is not None and res is not None and res != dst_target_guide_t.shape[2]:
                dst_target_guide_t = F.interpolate(dst_target_guide_t, size=res, mode='bilinear', align_corners=True)
            return dst_target_guide_t
        
        @cached_method
        def get_dst_target_mask_t(self, res : int = None) -> torch.Tensor|None:
            if (dst_target_guide_t := self.get_dst_target_guide_t(res=res)) is not None:
                # guide to gray
                gray = torch.tensor([[[[0.1140]],[[0.5870]],[[0.2990]]]], dtype=torch.float32, device=dst_target_guide_t.device)
                gray = (dst_target_guide_t * gray).sum(1, keepdims=True)
                # mask from gray >= 0.5
                return (gray >= 0.5).type(torch.float32)
            return None        
        
        @cached_method
        def get_dst_target_mask_blur_t(self, res : int = None) -> torch.Tensor|None:
            if (dst_target_mask_t := self.get_dst_target_mask_t(res=res)) is not None:
                dst_target_mask_t = torch.clamp(xF.gaussian_blur(dst_target_mask_t, sigma=max(1, dst_target_mask_t.shape[2] // 32) ), 0.0, 0.5) * 2.0
            return dst_target_mask_t
        @cached_method
        def get_dst_target_image_t_0(self) -> torch.Tensor|None: return self.image_seq_to_tensor(dst_target_image, [3], [self.target_resolution], [self.target_resolution]) if (job_inputs := self.job.inputs) is not None and (dst_target_image := job_inputs.dst_target_image) is not None else None
        @cached_method
        def get_dst_target_image_t(self, masked : bool) -> torch.Tensor|None:
            if (dst_target_image_t := self.get_dst_target_image_t_0()) is not None and masked:
                if (dst_target_mask_blur_t := self.get_dst_target_mask_blur_t(res=dst_target_image_t.shape[2])) is not None:
                    dst_target_image_t = dst_target_image_t * dst_target_mask_blur_t
            return dst_target_image_t
        @cached_method
        def get_src_enc_t(self) -> torch.Tensor|None: return self.encoder_forward(src_image_t) if (src_image_t := self.get_src_input_image_t()) is not None else None
        @cached_method
        def get_dst_enc_t(self) -> torch.Tensor|None: return self.encoder_forward(dst_image_t) if (dst_image_t := self.get_dst_input_image_t()) is not None else None
        @cached_method
        def get_src_src_code_t(self) -> torch.Tensor|None: return self.inter_src_forward(src_enc_t) if (src_enc_t := self.get_src_enc_t()) is not None else None
        @cached_method
        def get_src_dst_code_t(self) -> torch.Tensor|None: return self.inter_src_forward(dst_enc_t) if (dst_enc_t := self.get_dst_enc_t()) is not None else None
        @cached_method
        def get_dst_dst_code_t(self) -> torch.Tensor|None: return self.inter_dst_forward(dst_enc_t) if (dst_enc_t := self.get_dst_enc_t()) is not None else None
        @cached_method
        def get_src_code_t(self) -> torch.Tensor|None: return torch.cat([src_src_code_t, src_src_code_t], 1) if (src_src_code_t := self.get_src_src_code_t()) is not None else None
        @cached_method
        def get_dst_code_t(self) -> torch.Tensor|None:
            if (dst_dst_code_t := self.get_dst_dst_code_t()) is not None and \
               (src_dst_code_t := self.get_src_dst_code_t()) is not None: #(torch.zeros_like(dst_dst_code_t) if (self.job.training is not None and self.job.training.dst_pretrain_mode) else 
                return torch.cat([src_dst_code_t, dst_dst_code_t], 1)
            return None
        @cached_method
        def get_swap_code_t(self) -> torch.Tensor|None: return torch.cat([src_dst_code_t, src_dst_code_t], 1) if (src_dst_code_t := self.get_src_dst_code_t()) is not None else None
        @cached_method
        def get_pred_src_image_t_0(self) -> torch.Tensor|None: return self.decoder_forward(src_code_t) if (src_code_t := self.get_src_code_t()) is not None else None
        @cached_method
        def get_pred_src_image_t_1(self, detach=False) -> torch.Tensor|None:
            if (pred_src_image_t := self.get_pred_src_image_t_0()) is not None and detach:
                pred_src_image_t = pred_src_image_t.detach()
            return pred_src_image_t
        @cached_method
        def get_pred_src_image_t_3(self, masked : bool, detach=False) -> torch.Tensor|None:
            if (pred_src_image_t := self.get_pred_src_image_t_1(detach=detach)) is not None and masked:
                if (src_target_mask_blur_t := self.get_src_target_mask_blur_t(res=pred_src_image_t.shape[2])) is not None:
                    pred_src_image_t = pred_src_image_t * src_target_mask_blur_t
            return pred_src_image_t
        @cached_method
        def get_pred_src_image_t(self, masked : bool, res : int = None, detach=False) -> torch.Tensor|None:
            if (pred_src_image_t := self.get_pred_src_image_t_3(masked=masked, detach=detach)) is not None and res is not None and res != pred_src_image_t.shape[2]:
                pred_src_image_t = F.interpolate(pred_src_image_t, size=res, mode='bilinear', align_corners=True)
            return pred_src_image_t
        @cached_method
        def get_pred_src_enhanced_image_t(self) -> torch.Tensor|None:
            return self.enhancer_forward(pred_src_image_t) if (pred_src_image_t := self.get_pred_src_image_t(masked=False, res=self.enhancer_resolution)) is not None else None
        @cached_method
        def get_pred_src_guide_t(self) -> torch.Tensor|None: return self.decoder_guide_forward(src_code_t) if (src_code_t := self.get_src_code_t()) is not None else None
        
        @cached_method
        def get_pred_dst_image_t_0(self) -> torch.Tensor|None: return self.decoder_forward(dst_code_t) if (dst_code_t := self.get_dst_code_t()) is not None else None
        @cached_method
        def get_pred_dst_image_t_1(self, detach=False) -> torch.Tensor|None:
            if (pred_dst_image_t := self.get_pred_dst_image_t_0()) is not None and detach:
                pred_dst_image_t = pred_dst_image_t.detach()
            return pred_dst_image_t
        @cached_method
        def get_pred_dst_image_t_2(self, masked : bool, detach=False) -> torch.Tensor|None:
            if (pred_dst_image_t := self.get_pred_dst_image_t_1(detach=detach)) is not None and masked:
                if (dst_target_mask_blur_t := self.get_dst_target_mask_blur_t(res=pred_dst_image_t.shape[2])) is not None:
                    pred_dst_image_t = pred_dst_image_t * dst_target_mask_blur_t
            return pred_dst_image_t
        @cached_method
        def get_pred_dst_image_t(self, masked : bool, res : int = None, detach=False) -> torch.Tensor|None:
            if (pred_dst_image_t := self.get_pred_dst_image_t_2(masked=masked, detach=detach)) is not None and res is not None and res != pred_dst_image_t.shape[2]:
                pred_dst_image_t = F.interpolate(pred_dst_image_t, size=res, mode='bilinear', align_corners=True)
            return pred_dst_image_t
        
        @cached_method
        def get_pred_dst_guide_t(self) -> torch.Tensor|None: return self.decoder_guide_forward(dst_code_t) if (dst_code_t := self.get_dst_code_t()) is not None else None
        @cached_method
        def get_pred_swap_image_t_0(self) -> torch.Tensor|None: return self.decoder_forward(swap_code_t) if (swap_code_t := self.get_swap_code_t()) is not None else None
        @cached_method
        def get_pred_swap_image_t_1(self, detach=False) -> torch.Tensor|None:
            if (pred_swap_image_t := self.get_pred_swap_image_t_0()) is not None and detach:
                pred_swap_image_t = pred_swap_image_t.detach()
            return pred_swap_image_t
        @cached_method
        def get_pred_swap_image_t(self, res : int = None, detach=False) -> torch.Tensor|None:
            if (pred_swap_image_t := self.get_pred_swap_image_t_1(detach=detach)) is not None and res is not None and res != pred_swap_image_t.shape[2]:
                pred_swap_image_t = F.interpolate(pred_swap_image_t, size=res, mode='bilinear', align_corners=True)
            return pred_swap_image_t
        @cached_method
        def get_pred_swap_enhanced_image_t(self) -> torch.Tensor|None:
            return self.enhancer_forward(pred_swap_image_t) if (pred_swap_image_t := self.get_pred_swap_image_t(res=self.enhancer_resolution)) is not None else None
        @cached_method
        def get_pred_swap_guide_t(self) -> torch.Tensor|None:
            return self.decoder_guide_forward(swap_code_t) if (swap_code_t := self.get_swap_code_t()) is not None else None

