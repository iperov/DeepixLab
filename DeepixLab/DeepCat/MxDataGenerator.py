from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import numpy.random as nprnd

from common.FSIP import MxFSIPRefList
from core import ax, mx
from core.lib import mp as lib_mp
from core.lib import random as lib_rnd
from core.lib.collections import FDict, get_enum_id_by_name, shuffled
from core.lib.dataset.FSIP import FSIP
from core.lib.hash import Hash64
from core.lib.image import FImage
from core.lib.image import aug as lib_aug


class MxDataGenerator(mx.Disposable):

    class Mode(StrEnum):
        Fit = '@(Mode.Fit)'
        Patch = '@(Mode.Patch)'

    class ImageType(StrEnum):
        Image = '@(Image)'
        PairedImage = '@(Paired_image)'

    @dataclass
    class GenResult:
        image_paths : Sequence[Path]
        input_image : Sequence[FImage]
        target_image : Sequence[FImage]


    def __init__(self,  default_rnd_flip : bool = True,
                        state : FDict = None):
        super().__init__()

        self._main_thread = ax.get_current_thread()
        self._fg = ax.FutureGroup().dispose_with(self)
        self._reload_fg = ax.FutureGroup().dispose_with(self)
        self._bg_thread = ax.Thread('bg_thread').dispose_with(self)
        self._load_thread = ax.Thread('load_thread').dispose_with(self)
        #self._dcs_pool = ax.ThreadPool().dispose_with(self)

        self._s_pool = lib_mp.SubprocessPool(Worker, prio=lib_mp.SubprocessPool.Priority.IDLE).dispose_with(self)

        self._reloading = False
        self._mx_reloading_progress = mx.Progress().dispose_with(self)

        self._state = state = FDict(state)

        self._mx_error = mx.TextEmitter().dispose_with(self)

        self._mx_fsip_ref_list = MxFSIPRefList(pair_type_mode=MxFSIPRefList.PairTypeMode.SINGLE_CHOICE,
                                            allow_rate=True, state=state.get('fsip_ref_list', None)).dispose_with(self)


        self._mx_mode = mx.StateChoice[MxDataGenerator.Mode](availuator=lambda: MxDataGenerator.Mode).dispose_with(self)
        self._mx_mode.set( get_enum_id_by_name(MxDataGenerator.Mode, state.get('mode', None), MxDataGenerator.Mode.Fit) )

        self._mx_input_image_type = mx.StateChoice[MxDataGenerator.ImageType](availuator=lambda: MxDataGenerator.ImageType).dispose_with(self)
        self._mx_input_image_type.set( get_enum_id_by_name(MxDataGenerator.ImageType, state.get('input_image_type', None), MxDataGenerator.ImageType.Image) )
        self._mx_input_image_type.listen(lambda *_: self.apply_and_reload())

        self._mx_input_image_interp = mx.StateChoice[FImage.Interp](availuator=lambda: FImage.Interp).dispose_with(self)
        self._mx_input_image_interp.set( get_enum_id_by_name(FImage.Interp, state.get('input_image_interp', None), FImage.Interp.LANCZOS4) )
        self._mx_input_image_border_type = mx.StateChoice[FImage.Border](availuator=lambda: FImage.Border).dispose_with(self)
        self._mx_input_image_border_type.set( get_enum_id_by_name(FImage.Border, state.get('input_image_border_type', None), FImage.Border.REPLICATE) )

        self._mx_target_image_type = mx.StateChoice[MxDataGenerator.ImageType](availuator=lambda: MxDataGenerator.ImageType).dispose_with(self)
        self._mx_target_image_type.set( get_enum_id_by_name(MxDataGenerator.ImageType, state.get('target_image_type', None), MxDataGenerator.ImageType.PairedImage) )
        self._mx_target_image_type.listen(lambda *_: self.apply_and_reload())

        self._mx_target_image_interp = mx.StateChoice[FImage.Interp](availuator=lambda: FImage.Interp).dispose_with(self)
        self._mx_target_image_interp.set( get_enum_id_by_name(FImage.Interp, state.get('target_image_interp', None), FImage.Interp.LINEAR) )
        self._mx_target_image_border_type = mx.StateChoice[FImage.Border](availuator=lambda: FImage.Border).dispose_with(self)
        self._mx_target_image_border_type.set( get_enum_id_by_name(FImage.Border, state.get('target_image_border_type', None), FImage.Border.CONSTANT) )

        self._mx_offset_tx = mx.Number(state.get('offset_tx', 0.0), config=mx.Number.Config(min=-2.0, max=2.0, step=0.01)).dispose_with(self)
        self._mx_offset_ty = mx.Number(state.get('offset_ty', 0.0), config=mx.Number.Config(min=-2.0, max=2.0, step=0.01)).dispose_with(self)
        self._mx_offset_scale = mx.Number(state.get('offset_scale', 0.0), config=mx.Number.Config(min=-4.0, max=4.0, step=0.01)).dispose_with(self)
        self._mx_offset_rot_deg = mx.Number(state.get('offset_rot_deg', 0.0), config=mx.Number.Config(min=-180, max=180)).dispose_with(self)

        self._mx_rnd_tx_var = mx.Number(state.get('rnd_tx_var', 0.07), config=mx.Number.Config(min=0.0, max=2.0, step=0.01)).dispose_with(self)
        self._mx_rnd_ty_var = mx.Number(state.get('rnd_ty_var', 0.07), config=mx.Number.Config(min=0.0, max=2.0, step=0.01)).dispose_with(self)
        self._mx_rnd_scale_var = mx.Number(state.get('rnd_scale_var', 0.07), config=mx.Number.Config(min=0.0, max=4.0, step=0.01)).dispose_with(self)
        self._mx_rnd_rot_deg_var = mx.Number(state.get('rnd_rot_deg_var', 15), config=mx.Number.Config(min=0, max=180)).dispose_with(self)

        self._mx_input_blur_power = mx.Number(state.get('input_blur_power', 0.0), config=mx.Number.Config(min=0.0, max=16.0, step=0.01)).dispose_with(self)
        self._mx_target_blur_power = mx.Number(state.get('target_blur_power', 0.0), config=mx.Number.Config(min=0.0, max=16.0, step=0.01)).dispose_with(self)

        self._mx_rnd_flip = mx.Flag( state.get('rnd_flip', default_rnd_flip) ).dispose_with(self)
        self._mx_rnd_cut_edges = mx.Flag( state.get('rnd_cut_edges', True) ).dispose_with(self)
        self._mx_rnd_input_channels_deform  = mx.Flag( state.get('rnd_input_channels_deform', True) ).dispose_with(self)
        self._mx_rnd_target_channels_deform = mx.Flag( state.get('rnd_target_channels_deform', False) ).dispose_with(self)
        self._mx_rnd_input_levels_shift     = mx.Flag( state.get('rnd_input_levels_shift', True) ).dispose_with(self)
        self._mx_rnd_target_levels_shift    = mx.Flag( state.get('rnd_target_levels_shift', False) ).dispose_with(self)
        self._mx_rnd_input_blur             = mx.Flag( state.get('rnd_input_blur', True) ).dispose_with(self)
        self._mx_rnd_target_blur            = mx.Flag( state.get('rnd_target_blur', False) ).dispose_with(self)
        self._mx_rnd_input_sharpen          = mx.Flag( state.get('rnd_input_sharpen', True) ).dispose_with(self)
        self._mx_rnd_target_sharpen         = mx.Flag( state.get('rnd_target_sharpen', False) ).dispose_with(self)
        self._mx_rnd_input_resize           = mx.Flag( state.get('rnd_input_resize', True) ).dispose_with(self)
        self._mx_rnd_target_resize          = mx.Flag( state.get('rnd_target_resize', False) ).dispose_with(self)
        self._mx_rnd_input_jpeg_artifacts   = mx.Flag( state.get('rnd_input_jpeg_artifacts', True) ).dispose_with(self)
        self._mx_rnd_target_jpeg_artifacts  = mx.Flag( state.get('rnd_target_jpeg_artifacts', False) ).dispose_with(self)

        self._mx_transform_intensity = mx.Number(state.get('transform_intensity', 1.0), config=mx.Number.Config(min=0.0, max=1.0, step=0.01)).dispose_with(self)
        self._mx_image_deform_intensity = mx.Number(state.get('image_deform_intensity', 0.75), config=mx.Number.Config(min=0.0, max=1.0, step=0.01)).dispose_with(self)
        self._mx_image_deform_intensity.listen(lambda s: self._mx_target_image_deform_intensity.set(s) if self._mx_target_image_deform_intensity.get() > s else ...)
        self._mx_target_image_deform_intensity = mx.Number(state.get('target_image_deform_intensity', 0.75), config=mx.Number.Config(min=0.0, max=1.0, step=0.01)).dispose_with(self)
        self._mx_target_image_deform_intensity.listen(lambda s: self._mx_image_deform_intensity.set(s) if self._mx_image_deform_intensity.get() < s else ...)

        self._mx_dcs = mx.Flag( state.get('dcs', False) ).dispose_with(self)
        self._mx_dcs.listen(lambda b: self.apply_and_reload())
        self._mx_dcs_progress = mx.Progress().dispose_with(self)

        self.apply_and_reload()

    def get_state(self) -> FDict:
        return FDict({  'fsip_ref_list' : self._mx_fsip_ref_list.get_state(),
                        'mode' : self._mx_mode.get().name,

                        'input_image_type' : self._mx_input_image_type.get().name,
                        'input_image_interp' : self._mx_input_image_interp.get().name,
                        'input_image_border_type' : self._mx_input_image_border_type.get().name,

                        'target_image_type' : self._mx_target_image_type.get().name,
                        'target_image_interp' : self._mx_target_image_interp.get().name,
                        'target_image_border_type' : self._mx_target_image_border_type.get().name,

                        'offset_tx' : self._mx_offset_tx.get(),
                        'offset_ty' : self._mx_offset_ty.get(),
                        'offset_scale' : self._mx_offset_scale.get(),
                        'offset_rot_deg' : self._mx_offset_rot_deg.get(),

                        'rnd_tx_var' : self._mx_rnd_tx_var.get(),
                        'rnd_ty_var' : self._mx_rnd_ty_var.get(),
                        'rnd_scale_var' : self._mx_rnd_scale_var.get(),
                        'rnd_rot_deg_var' : self.mx_rnd_rot_deg_var.get(),

                        'input_blur_power' : self._mx_input_blur_power.get(),
                        'target_blur_power' : self._mx_target_blur_power.get(),

                        'rnd_flip' : self._mx_rnd_flip.get(),
                        'rnd_cut_edges' : self._mx_rnd_cut_edges.get(),
                        'rnd_input_channels_deform'  : self._mx_rnd_input_channels_deform.get(),
                        'rnd_target_channels_deform' : self._mx_rnd_target_channels_deform.get(),
                        'rnd_input_levels_shift'     : self._mx_rnd_input_levels_shift.get(),
                        'rnd_target_levels_shift'    : self._mx_rnd_target_levels_shift.get(),
                        'rnd_input_blur'             : self._mx_rnd_input_blur.get(),
                        'rnd_target_blur'            : self._mx_rnd_target_blur.get(),
                        'rnd_input_sharpen'          : self._mx_rnd_input_sharpen.get(),
                        'rnd_target_sharpen'         : self._mx_rnd_target_sharpen.get(),
                        'rnd_input_resize'           : self._mx_rnd_input_resize.get(),
                        'rnd_target_resize'          : self._mx_rnd_target_resize.get(),
                        'rnd_input_jpeg_artifacts'   : self._mx_rnd_input_jpeg_artifacts.get(),
                        'rnd_target_jpeg_artifacts'  : self._mx_rnd_target_jpeg_artifacts.get(),

                        'transform_intensity' : self._mx_transform_intensity.get(),
                        'image_deform_intensity' : self._mx_image_deform_intensity.get(),
                        'target_image_deform_intensity' : self._mx_target_image_deform_intensity.get(),


                        'dcs' : self._mx_dcs.get(), })

    @property
    def mx_error(self) -> mx.ITextEmitter_v: return self._mx_error
    @property
    def mx_fsip_ref_list(self) -> MxFSIPRefList: return self._mx_fsip_ref_list
    @property
    def mx_reloading_progress(self) -> mx.IProgress_rv: return self._mx_reloading_progress

    @property
    def mx_mode(self) -> mx.IStateChoice_v[MxDataGenerator.Mode]: return self._mx_mode
    @property
    def mx_input_image_type(self) -> mx.IStateChoice_v[MxDataGenerator.ImageType]: return self._mx_input_image_type
    @property
    def mx_input_image_interp(self) -> mx.IStateChoice_v[FImage.Interp]: return self._mx_input_image_interp
    @property
    def mx_input_image_border_type(self) -> mx.IStateChoice_v[FImage.Border]: return self._mx_input_image_border_type
    @property
    def mx_target_image_type(self) -> mx.IStateChoice_v[MxDataGenerator.ImageType]: return self._mx_target_image_type
    @property
    def mx_target_image_interp(self) -> mx.IStateChoice_v[FImage.Interp]: return self._mx_target_image_interp
    @property
    def mx_target_image_border_type(self) -> mx.IStateChoice_v[FImage.Border]: return self._mx_target_image_border_type

    @property
    def mx_offset_tx(self) -> mx.INumber_v: return self._mx_offset_tx
    @property
    def mx_offset_ty(self) -> mx.INumber_v: return self._mx_offset_ty
    @property
    def mx_offset_scale(self) -> mx.INumber_v: return self._mx_offset_scale
    @property
    def mx_offset_rot_deg(self) -> mx.INumber_v: return self._mx_offset_rot_deg

    @property
    def mx_input_blur_power(self) -> mx.INumber_v: return self._mx_input_blur_power
    @property
    def mx_target_blur_power(self) -> mx.INumber_v: return self._mx_target_blur_power

    @property
    def mx_rnd_tx_var(self) -> mx.INumber_v: return self._mx_rnd_tx_var
    @property
    def mx_rnd_ty_var(self) -> mx.INumber_v: return self._mx_rnd_ty_var
    @property
    def mx_rnd_scale_var(self) -> mx.INumber_v: return self._mx_rnd_scale_var
    @property
    def mx_rnd_rot_deg_var(self) -> mx.INumber_v: return self._mx_rnd_rot_deg_var
    @property
    def mx_rnd_flip(self) -> mx.IFlag_v: return self._mx_rnd_flip
    @property
    def mx_rnd_cut_edges(self) -> mx.IFlag_v: return self._mx_rnd_cut_edges
    @property
    def mx_rnd_input_channels_deform(self) -> mx.IFlag_v: return self._mx_rnd_input_channels_deform
    @property
    def mx_rnd_target_channels_deform(self) -> mx.IFlag_v: return self._mx_rnd_target_channels_deform
    @property
    def mx_rnd_input_levels_shift(self) -> mx.IFlag_v: return self._mx_rnd_input_levels_shift
    @property
    def mx_rnd_target_levels_shift(self) -> mx.IFlag_v: return self._mx_rnd_target_levels_shift
    @property
    def mx_rnd_input_blur(self) -> mx.IFlag_v: return self._mx_rnd_input_blur
    @property
    def mx_rnd_target_blur(self) -> mx.IFlag_v: return self._mx_rnd_target_blur
    @property
    def mx_rnd_input_sharpen(self) -> mx.IFlag_v: return self._mx_rnd_input_sharpen
    @property
    def mx_rnd_target_sharpen(self) -> mx.IFlag_v: return self._mx_rnd_target_sharpen
    @property
    def mx_rnd_input_resize(self) -> mx.IFlag_v: return self._mx_rnd_input_resize
    @property
    def mx_rnd_target_resize(self) -> mx.IFlag_v: return self._mx_rnd_target_resize
    @property
    def mx_rnd_input_jpeg_artifacts(self) -> mx.IFlag_v: return self._mx_rnd_input_jpeg_artifacts
    @property
    def mx_rnd_target_jpeg_artifacts(self) -> mx.IFlag_v: return self._mx_rnd_target_jpeg_artifacts


    @property
    def mx_transform_intensity(self) -> mx.INumber_v: return self._mx_transform_intensity
    @property
    def mx_image_deform_intensity(self) -> mx.INumber_v: return self._mx_image_deform_intensity
    @property
    def mx_target_image_deform_intensity(self) -> mx.INumber_v: return self._mx_target_image_deform_intensity

    @property
    def mx_dcs(self) -> mx.IFlag_v: return self._mx_dcs
    @property
    def mx_dcs_progress(self) -> mx.IProgress_rv: return self._mx_dcs_progress
    @property
    def workers_count(self) -> int: return self._s_pool._process_count



    @ax.task
    def apply_and_reload(self):
        """apply settings and reload files"""
        yield ax.switch_to(self._main_thread)
        yield ax.attach_to(self._fg)
        yield ax.attach_to(self._reload_fg, cancel_all=True)

        self._mx_dcs_progress.finish()

        self._reloading = True
        self._mx_reloading_progress.start().set_inf()

        input_image_type = self._mx_input_image_type.get()
        target_image_type = self._mx_target_image_type.get()

        yield ax.switch_to(self._load_thread)

        err = None
        try:
            # Collect paths from MxFSIPRef
            items_by_att_id = {}
            probs_by_att_id = {}

            for v in self._mx_fsip_ref_list.values:
                if v.mx_fsip_path.mx_opened.get():
                    fsip_path = v.mx_fsip_path.get()
                    pair_type = v.mx_pair_type.get()
                    rate      = v.mx_rate.get()/100.0

                    att_id    = 0#v.mx_att_id.get()

                    fsip = FSIP.open(fsip_path)

                    new_items = []
                    for item_id in range(fsip.item_count):
                        item_path = fsip.get_item_path(item_id)
                        pair_path = fsip.get_pair_path(item_id, pair_type) if pair_type not in [None, v.mx_pair_type.NO_PAIR] else None

                        if input_image_type == self.ImageType.Image:
                            input_path = item_path
                        elif input_image_type == self.ImageType.PairedImage:
                            input_path = pair_path

                        if target_image_type == self.ImageType.Image:
                            target_path = item_path
                        elif target_image_type == self.ImageType.PairedImage:
                            target_path = pair_path

                        if input_path is not None and target_path is not None:
                            new_items.append( (input_path, target_path) )

                    items = items_by_att_id.get(att_id, None)
                    if items is None:
                        items = items_by_att_id[att_id] = []
                    probs = probs_by_att_id.get(att_id, None)
                    if probs is None:
                        probs = probs_by_att_id[att_id] = []

                    items += new_items
                    probs += [ np.full( (len(new_items),), rate, dtype=np.float32) ]

            att_choicers = [ lib_rnd.Choicer(items, np.concatenate(probs_by_att_id[att_id], 0)) for att_id, items in items_by_att_id.items() ]

            choicer = lib_rnd.Choicer(att_choicers, [1.0]*len(att_choicers) )
        except Exception as e:
            err = e

        yield ax.switch_to(self._main_thread)

        self._reloading = False
        self._mx_reloading_progress.finish()

        if err is None:
            self._choicer = choicer

        if err is not None:
            self._mx_error.emit(str(err))
            yield ax.cancel(err)

    @ax.task
    def generate(self, shape : Tuple[int,int,int,int,int], asap=False) -> GenResult:
        """
        Start generation job.

            shape   (N,H,W,IC,OC)

        Depends on parent task.
        Cancelling task will interrupt the job ASAP.
        """
        N,H,W,IC,OC = shape

        if H < 4 or W < 4 or IC not in [1,3,4] or OC not in [1,3,4]:
            raise ValueError('Wrong shape')

        yield ax.attach_to(self._fg, detach_parent=False)
        yield ax.switch_to(self._main_thread)

        while self._reloading:
            yield ax.sleep(0.1)

        mode = self._mx_mode.get()

        choicer = self._choicer
        items = choicer.pick(N)
        if len(items) == 0:
            yield ax.cancel(Exception('No training data.'))

        yield ax.switch_to(self._bg_thread)

        yield ax.propagate(self._s_pool.process( Worker.Job(items = items,
                                                            shape = (H,W,IC,OC),
                                                            mode = mode,
                                                            input_image_interp=self._mx_input_image_interp.get(),
                                                            input_image_border_type = self._mx_input_image_border_type.get(),

                                                            target_image_interp=self._mx_target_image_interp.get(),
                                                            target_image_border_type = self._mx_target_image_border_type.get(),

                                                            offset_tx      = self._mx_offset_tx.get(),
                                                            offset_ty      = self._mx_offset_ty.get(),
                                                            offset_scale   = self._mx_offset_scale.get(),
                                                            offset_rot_deg = self._mx_offset_rot_deg.get(),

                                                            rnd_tx_var      = self._mx_rnd_tx_var.get(),
                                                            rnd_ty_var      = self._mx_rnd_ty_var.get(),
                                                            rnd_scale_var   = self._mx_rnd_scale_var.get(),
                                                            rnd_rot_deg_var = self._mx_rnd_rot_deg_var.get(),

                                                            transform_intensity  = self._mx_transform_intensity.get(),
                                                            image_deform_intensity = self._mx_image_deform_intensity.get(),
                                                            target_image_deform_intensity = self._mx_target_image_deform_intensity.get(),

                                                            input_blur_power = self._mx_input_blur_power.get(),
                                                            target_blur_power = self._mx_target_blur_power.get(),

                                                            rnd_flip            = self._mx_rnd_flip.get(),
                                                            rnd_cut_edges       = self._mx_rnd_cut_edges.get(),
                                                            rnd_input_channels_deform  = self._mx_rnd_input_channels_deform.get(),
                                                            rnd_target_channels_deform = self._mx_rnd_target_channels_deform.get(),
                                                            rnd_input_levels_shift     = self._mx_rnd_input_levels_shift.get(),
                                                            rnd_target_levels_shift    = self._mx_rnd_target_levels_shift.get(),
                                                            rnd_input_blur             = self._mx_rnd_input_blur.get(),
                                                            rnd_target_blur            = self._mx_rnd_target_blur.get(),
                                                            rnd_input_sharpen          = self._mx_rnd_input_sharpen.get(),
                                                            rnd_target_sharpen         = self._mx_rnd_target_sharpen.get(),
                                                            rnd_input_resize           = self._mx_rnd_input_resize.get(),
                                                            rnd_target_resize          = self._mx_rnd_target_resize.get(),
                                                            rnd_input_jpeg_artifacts   = self._mx_rnd_input_jpeg_artifacts.get(),
                                                            rnd_target_jpeg_artifacts  = self._mx_rnd_target_jpeg_artifacts.get(),

                                                            ),

                                                    asap=asap ))


class Worker(lib_mp.SubprocessClass):

    @dataclass
    class Job:
        items : List
        shape : Tuple[int, int, int, int]

        mode : MxDataGenerator.Mode

        input_image_interp : FImage.Interp
        input_image_border_type : FImage.Border

        target_image_interp  : FImage.Interp
        target_image_border_type : FImage.Border

        offset_tx : float
        offset_ty : float
        offset_scale : float
        offset_rot_deg : float

        rnd_tx_var : float
        rnd_ty_var : float
        rnd_scale_var : float
        rnd_rot_deg_var : float

        transform_intensity : float
        image_deform_intensity : float
        target_image_deform_intensity : float

        input_blur_power : float
        target_blur_power : float

        rnd_flip : bool
        rnd_cut_edges : bool
        rnd_input_channels_deform : bool
        rnd_target_channels_deform : bool
        rnd_input_levels_shift : bool
        rnd_target_levels_shift : bool
        rnd_input_blur : bool
        rnd_target_blur : bool
        rnd_input_sharpen : bool
        rnd_target_sharpen : bool
        rnd_input_resize : bool
        rnd_target_resize : bool
        rnd_input_jpeg_artifacts : bool
        rnd_target_jpeg_artifacts : bool


    def __init__(self):
        super().__init__()
        self._jobs_fg = ax.FutureGroup().dispose_with(self)
        import cv2
        cv2.setNumThreads(1)

    @ax.task
    def process(self, job : Worker.Job, asap = False):
        if not asap:
            while self._jobs_fg.count != 0:
                yield ax.sleep(0)

        yield ax.attach_to(self._jobs_fg)

        H, W, IC, OC = job.shape
        mode = job.mode

        out_image_paths : List[Path] = []

        out_input_image : List[FImage] = []
        out_target_image : List[FImage] = []

        rnd_state = np.random.RandomState()

        for i, (image_path, pair_path) in enumerate(job.items):
            try:
                input_image = FImage.from_file(image_path)
                target_image = input_image if pair_path == image_path else FImage.from_file(pair_path)

                input_image = input_image.ch(IC)
                target_image = target_image.ch(OC)

            except Exception as e:
                yield ax.cancel(error=e)


            if mode == MxDataGenerator.Mode.Fit:
                offset_transform_params = lib_aug.TransformParams(  tx=job.offset_tx,
                                                                    ty=job.offset_ty,
                                                                    scale=job.offset_scale,
                                                                    rot_deg=job.offset_rot_deg)
                transform_params = lib_aug.TransformParams( tx=nprnd.uniform(-job.rnd_tx_var, job.rnd_tx_var),
                                                            ty=nprnd.uniform(-job.rnd_ty_var, job.rnd_ty_var),
                                                            scale=nprnd.uniform(-job.rnd_scale_var, job.rnd_scale_var),
                                                            rot_deg=nprnd.uniform(-job.rnd_rot_deg_var, job.rnd_rot_deg_var))
                center_fit = True
            elif mode == MxDataGenerator.Mode.Patch:
                offset_transform_params = lib_aug.TransformParams(tx=job.offset_tx,
                                                                  ty=job.offset_ty,)
                transform_params = lib_aug.TransformParams( tx=nprnd.uniform(-0.50, 0.50),
                                                            ty=nprnd.uniform(-0.50, 0.50),
                                                            scale=nprnd.uniform(-job.rnd_scale_var, job.rnd_scale_var),
                                                            rot_deg=nprnd.uniform(-job.rnd_rot_deg_var, job.rnd_rot_deg_var))
                center_fit = False

            geo_aug = lib_aug.Geo(offset_transform_params=offset_transform_params, transform_params=transform_params)
            input_image  = geo_aug.transform(input_image, W, H, center_fit=center_fit, transform_intensity=job.transform_intensity, interp=job.input_image_interp, deform_intensity=job.image_deform_intensity, border=job.input_image_border_type)
            target_image = geo_aug.transform(target_image, W, H, center_fit=center_fit, transform_intensity=job.transform_intensity, interp=job.target_image_interp, deform_intensity=job.target_image_deform_intensity, border=job.target_image_border_type)

            # Static augmentations
            if (input_blur_power := job.input_blur_power) != 0:
                input_image = input_image.gaussian_blur(input_blur_power)

            if (target_blur_power := job.target_blur_power) != 0:
                target_image = target_image.gaussian_blur(target_blur_power)

            # Random augmentations
            if job.rnd_flip and nprnd.randint(2) == 0:
                input_image  = input_image.h_flip()
                target_image = target_image.h_flip()

            for aug in shuffled([0,1,2,3,4,5]):
                aug_input_image = None
                aug_target_image = None

                seed = rnd_state.randint(2**31)
                if aug == 0:
                    if job.rnd_input_channels_deform:
                        aug_input_image = lib_aug.channels_deform(input_image, seed=seed)
                    if job.rnd_target_channels_deform:
                        aug_target_image = lib_aug.channels_deform(target_image, seed=seed)
                elif aug == 1:
                    b = rnd_state.randint(2) == 0

                    if job.rnd_input_levels_shift:
                        aug_input_image = lib_aug.hsv_shift(input_image, seed=seed) if (IC == 3 and b == 0) else \
                                          lib_aug.levels(input_image, seed=seed)
                    if job.rnd_target_levels_shift:
                        aug_target_image = lib_aug.hsv_shift(target_image, seed=seed) if (OC == 3 and b == 0) else \
                                           lib_aug.levels(target_image, seed=seed)
                elif aug == 2:
                    blur_f = rnd_state.choice(_aug_blur_f)
                    if job.rnd_input_blur:
                        aug_input_image = blur_f(input_image, seed)
                    if job.rnd_target_blur:
                        aug_target_image = blur_f(target_image, seed)

                elif aug == 3:
                    sharpen_f = rnd_state.choice(_aug_sharpen_f)
                    if job.rnd_input_sharpen:
                        aug_input_image = sharpen_f(input_image, seed)
                    if job.rnd_target_sharpen:
                        aug_target_image = sharpen_f(target_image, seed)

                elif aug == 4:
                    resize_f = rnd_state.choice(_aug_resize_f)
                    if job.rnd_input_resize:
                        aug_input_image = resize_f(input_image, seed)
                    if job.rnd_target_resize:
                        aug_target_image = resize_f(target_image, seed)

                elif aug == 5:
                    if job.rnd_input_jpeg_artifacts:
                        aug_input_image = lib_aug.jpeg_artifacts(input_image, seed=seed)
                    if job.rnd_target_jpeg_artifacts:
                        aug_target_image = lib_aug.jpeg_artifacts(target_image, seed=seed)


                #seed = rnd_state.randint(2**31)
                if aug_input_image is not None or \
                   aug_target_image is not None:
                    aug_mask = lib_aug.circle_faded_mask(W, H)

                    if nprnd.randint(2) == 0:
                        clouds_mask = lib_aug.noise_clouds(W, H)
                        clouds_mask = lib_aug.motion_blur(clouds_mask)

                        aug_mask = aug_mask * clouds_mask

                    if nprnd.randint(4) == 0:
                        aug_mask = aug_mask * lib_aug.binary_stripes(W, H).gaussian_blur( nprnd.uniform(8, 32) ).satushift()

                if aug_input_image is not None:
                    input_image = input_image.blend(aug_input_image, aug_mask)
                if aug_target_image is not None:
                    target_image = target_image.blend(aug_target_image, aug_mask)

            # Random cut edges
            if job.rnd_cut_edges and nprnd.uniform() <= 0.25:
                mask = lib_aug.cut_edges_mask(H, W)

                input_image = input_image * mask.ch(input_image.shape[-1])
                target_image = target_image * mask.ch(target_image.shape[-1])

            out_image_paths.append(image_path)
            out_input_image.append(input_image)
            out_target_image.append(target_image)

            yield ax.sleep(0)

        return MxDataGenerator.GenResult(image_paths = out_image_paths,
                                         input_image = out_input_image,
                                         target_image = out_target_image,
                                        )

_aug_blur_f = [ lambda img, seed: lib_aug.motion_blur(img, seed=seed),
                lambda img, seed: lib_aug.gaussian_blur(img, seed=seed) ]

_aug_sharpen_f = [  lambda img, seed: lib_aug.box_sharpen(img, seed=seed),
                    lambda img, seed: lib_aug.gaussian_sharpen(img, seed=seed), ]

_aug_resize_f = [   lambda img, seed: lib_aug.resize(img, interp=FImage.Interp.NEAREST, seed=seed),
                    lambda img, seed: lib_aug.resize(img, interp=FImage.Interp.LINEAR, seed=seed),
                    #lambda img: lib_aug.resize(img, interp=FImage.Interp.CUBIC),
                    #lambda img: lib_aug.resize(img, interp=FImage.Interp.LANCZOS4),
                    ]