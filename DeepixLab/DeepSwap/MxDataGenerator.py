from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import numpy.random as nprnd

from common.FSIP import MxFSIPRefList
from core import ax, mx
from core.lib import mp as lib_mp
from core.lib import random as lib_rnd
from core.lib.collections import FDict
from core.lib.dataset.FSIP import FSIP
from core.lib.hash import Hash64, Hash64Similarity
from core.lib.image import FImage
from core.lib.image import aug as lib_aug


class MxDataGenerator(mx.Disposable):

    @dataclass
    class GenResult:
        image_paths : Sequence[Path]
        input_image : Sequence[FImage]
        target_image : Sequence[FImage]
        target_guide : Sequence[FImage]
        
    def __init__(self,  default_rnd_flip : bool = True,
                        process_frac : float = 1.0,
                        state : FDict = None):
        super().__init__()

        self._main_thread = ax.get_current_thread()
        self._fg = ax.FutureGroup().dispose_with(self)
        self._reload_fg = ax.FutureGroup().dispose_with(self)
        self._bg_thread = ax.Thread('bg_thread').dispose_with(self)
        self._ds_thread = ax.Thread('ds_thread').dispose_with(self)
        self._dcs_pool = ax.ThreadPool().dispose_with(self)

        self._s_pool = lib_mp.SubprocessPool(Worker, process_frac=process_frac, prio=lib_mp.SubprocessPool.Priority.IDLE).dispose_with(self)

        self._reloading = False
        self._mx_reloading_progress = mx.Progress().dispose_with(self)

        self._state = state = FDict(state)

        self._mx_error = mx.TextEmitter().dispose_with(self)

        self._mx_fsip_ref_list = MxFSIPRefList(pair_type_mode=MxFSIPRefList.PairTypeMode.SINGLE_CHOICE,
                                            allow_rate=True, state=state.get('fsip_ref_list', None)).dispose_with(self)

        self._mx_offset_tx = mx.Number(state.get('offset_tx', 0.0), config=mx.Number.Config(min=-2.0, max=2.0, step=0.01)).dispose_with(self)
        self._mx_offset_ty = mx.Number(state.get('offset_ty', 0.0), config=mx.Number.Config(min=-2.0, max=2.0, step=0.01)).dispose_with(self)
        self._mx_offset_scale = mx.Number(state.get('offset_scale', 0.0), config=mx.Number.Config(min=-4.0, max=4.0, step=0.01)).dispose_with(self)
        self._mx_offset_rot_deg = mx.Number(state.get('offset_rot_deg', 0.0), config=mx.Number.Config(min=-180, max=180)).dispose_with(self)

        self._mx_rnd_tx_var = mx.Number(state.get('rnd_tx_var', 0.07), config=mx.Number.Config(min=0.0, max=2.0, step=0.01)).dispose_with(self)
        self._mx_rnd_ty_var = mx.Number(state.get('rnd_ty_var', 0.07), config=mx.Number.Config(min=0.0, max=2.0, step=0.01)).dispose_with(self)
        self._mx_rnd_scale_var = mx.Number(state.get('rnd_scale_var', 0.07), config=mx.Number.Config(min=0.0, max=4.0, step=0.01)).dispose_with(self)
        self._mx_rnd_rot_deg_var = mx.Number(state.get('rnd_rot_deg_var', 15), config=mx.Number.Config(min=0, max=180)).dispose_with(self)

        self._mx_rnd_flip = mx.Flag( state.get('rnd_flip', default_rnd_flip) ).dispose_with(self)
        self._mx_rnd_r_exposure_var = mx.Number(state.get('rnd_r_exposure_var', 0.0), config=mx.Number.Config(min=0.0, max=2.0, step=0.1)).dispose_with(self)
        self._mx_rnd_g_exposure_var = mx.Number(state.get('rnd_g_exposure_var', 0.0), config=mx.Number.Config(min=0.0, max=2.0, step=0.1)).dispose_with(self)
        self._mx_rnd_b_exposure_var = mx.Number(state.get('rnd_b_exposure_var', 0.0), config=mx.Number.Config(min=0.0, max=2.0, step=0.1)).dispose_with(self)


        self._mx_transform_intensity = mx.Number(state.get('transform_intensity', 1.0), config=mx.Number.Config(min=0.0, max=1.0, step=0.01)).dispose_with(self)
        self._mx_image_deform_intensity = mx.Number(state.get('image_deform_intensity', 0.75), config=mx.Number.Config(min=0.0, max=1.0, step=0.01)).dispose_with(self)

        self._mx_dcs = mx.Flag( state.get('dcs', False) ).dispose_with(self)
        self._mx_dcs.listen(lambda b: self.apply_and_reload())
        self._mx_dcs_progress = mx.Progress().dispose_with(self)

        self.apply_and_reload()

    def get_state(self) -> FDict:
        return FDict({  'fsip_ref_list' : self._mx_fsip_ref_list.get_state(),

                        'offset_tx' : self._mx_offset_tx.get(),
                        'offset_ty' : self._mx_offset_ty.get(),
                        'offset_scale' : self._mx_offset_scale.get(),
                        'offset_rot_deg' : self._mx_offset_rot_deg.get(),

                        'rnd_tx_var' : self._mx_rnd_tx_var.get(),
                        'rnd_ty_var' : self._mx_rnd_ty_var.get(),
                        'rnd_scale_var' : self._mx_rnd_scale_var.get(),
                        'rnd_rot_deg_var' : self.mx_rnd_rot_deg_var.get(),

                        'rnd_flip' : self._mx_rnd_flip.get(),
                        'rnd_r_exposure_var' : self._mx_rnd_r_exposure_var.get(),
                        'rnd_g_exposure_var' : self._mx_rnd_g_exposure_var.get(),
                        'rnd_b_exposure_var' : self._mx_rnd_b_exposure_var.get(),


                        'transform_intensity' : self._mx_transform_intensity.get(),
                        'image_deform_intensity' : self._mx_image_deform_intensity.get(),

                        'dcs' : self._mx_dcs.get(), })

    @property
    def mx_error(self) -> mx.ITextEmitter_v: return self._mx_error
    @property
    def mx_fsip_ref_list(self) -> MxFSIPRefList: return self._mx_fsip_ref_list
    @property
    def mx_reloading_progress(self) -> mx.IProgress_rv: return self._mx_reloading_progress

    @property
    def mx_offset_tx(self) -> mx.INumber_v: return self._mx_offset_tx
    @property
    def mx_offset_ty(self) -> mx.INumber_v: return self._mx_offset_ty
    @property
    def mx_offset_scale(self) -> mx.INumber_v: return self._mx_offset_scale
    @property
    def mx_offset_rot_deg(self) -> mx.INumber_v: return self._mx_offset_rot_deg
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
    def mx_rnd_r_exposure_var(self) -> mx.INumber_v: return self._mx_rnd_r_exposure_var
    @property
    def mx_rnd_g_exposure_var(self) -> mx.INumber_v: return self._mx_rnd_g_exposure_var
    @property
    def mx_rnd_b_exposure_var(self) -> mx.INumber_v: return self._mx_rnd_b_exposure_var

    @property
    def mx_transform_intensity(self) -> mx.INumber_v: return self._mx_transform_intensity
    @property
    def mx_image_deform_intensity(self) -> mx.INumber_v: return self._mx_image_deform_intensity

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

        yield ax.switch_to(self._ds_thread)

        err = None
        try:
            # Collect paths from MxFSIPRef
            items = []
            items_rates = [ np.full( (0,), 0, dtype=np.float32) ]

            for v in self._mx_fsip_ref_list.values:
                if v.mx_fsip_path.mx_opened.get():
                    fsip_path = v.mx_fsip_path.get()
                    pair_type = v.mx_pair_type.get()
                    rate      = v.mx_rate.get()/100.0

                    fsip = FSIP.open(fsip_path)

                    new_items = []
                    for item_id in range(fsip.item_count):
                        item_path = fsip.get_item_path(item_id)
                        pair_path = fsip.get_pair_path(item_id, pair_type) if pair_type not in [None, v.mx_pair_type.NO_PAIR] else None

                        if item_path is not None and pair_path is not None:
                            new_items.append( (item_path, pair_path) )

                    items += new_items
                    items_rates += [ np.full( (len(new_items),), rate, dtype=np.float32) ]

            items_rates = np.concatenate(items_rates, 0)
        except Exception as e:
            err = e

        yield ax.switch_to(self._main_thread)

        self._reloading = False
        self._mx_reloading_progress.finish()

        if err is None:
            items_len = len(items)
            self._items = items
            
            self._choicer = lib_rnd.Choicer(items, items_rates)
            
            if self._mx_dcs.get() and items_len != 0:
                self._mx_dcs_progress.start().set(0, 100)

                @ax.task
                def _send_progress(progress : int):
                    yield ax.switch_to(self._main_thread)
                    self._mx_dcs_progress.set(progress)

                yield ax.switch_to(self._ds_thread)

                hash_sim = Hash64Similarity(items_len, similarity_factor=8)

                n_hashed = 0
                for value in ax.FutureGenerator(
                                (  (self._compute_hash(self._dcs_pool, item_path,), (item_id, item_path) )
                                    for item_id, (item_path, _) in enumerate(items) ),
                                max_parallel=self._dcs_pool.count*2 ):

                    if value is not None:
                        fut, (item_id, item_path) = value
                        n_hashed += 1

                        _send_progress( int( (n_hashed / items_len) * 100 ) )

                        if fut.succeeded:
                            hash_sim.add(item_id, fut.result )
                        else:
                            err = f'{item_path} : {fut.error}'
                            break
                    else:
                        yield ax.sleep(0)

                yield ax.switch_to(self._main_thread)

                if err is None:
                    similarities = hash_sim.get_similarities().astype(np.float32, copy=False)

                    # More similarities decrease rate
                    similarities_rates = similarities.min() / similarities
                    
                    # Replace choicer 
                    self._choicer = lib_rnd.Choicer(items, items_rates * similarities_rates)

                    self._mx_dcs_progress.finish()

        if err is not None:
            self._mx_error.emit(str(err))
            yield ax.cancel(err)


    @ax.task
    def _compute_hash(self, pool : ax.ThreadPool, path : Path) -> Hash64:
        yield ax.switch_to(pool)
        return FImage.from_file(path).get_perc_hash()

    @ax.task
    def generate(self, N : int, input_shape : Tuple[int,int,int], target_shape : Tuple[int,int,int], asap=False) -> GenResult:
        """
        Start generation job.

            shape   (N,H,W,C)

        Depends on parent task.
        Cancelling task will interrupt the job ASAP.
        """
        IH,IW,IC = input_shape
        OH,OW,OC = target_shape

        if IH < 4 or IW < 4 or IC not in [1,3] or \
           OH < 4 or OW < 4 or OC not in [1,3]:
            raise ValueError('Wrong shape')

        yield ax.attach_to(self._fg, detach_parent=False)
        yield ax.switch_to(self._main_thread)

        while self._reloading:
            yield ax.sleep(0.1)

        items = self._choicer.pick(N)
        if len(items) == 0:
            yield ax.cancel(Exception('No training data.'))

        yield ax.switch_to(self._bg_thread)

        yield ax.propagate(self._s_pool.process( Worker.Job(items = items,
                                                            input_shape = input_shape,
                                                            target_shape = target_shape,

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
                                                            
                                                            rnd_flip           = self._mx_rnd_flip.get(),
                                                            rnd_r_exposure_var = self._mx_rnd_r_exposure_var.get(),
                                                            rnd_g_exposure_var = self._mx_rnd_g_exposure_var.get(),
                                                            rnd_b_exposure_var = self._mx_rnd_b_exposure_var.get(),
                                                            ),

                                                    asap=asap ))


class Worker(lib_mp.SubprocessClass):

    @dataclass
    class Job:
        items : List
        input_shape : Tuple[int, int, int]
        target_shape : Tuple[int, int, int]

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

        rnd_flip : bool
        rnd_r_exposure_var : float
        rnd_g_exposure_var : float
        rnd_b_exposure_var : float


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

        IH, IW, IC = job.input_shape
        OH, OW, OC = job.target_shape
        
        rnd_r_exposure_var = job.rnd_r_exposure_var
        rnd_g_exposure_var = job.rnd_g_exposure_var
        rnd_b_exposure_var = job.rnd_b_exposure_var
        
        out_image_paths : List[Path] = []

        out_input_image : List[FImage] = []
        out_target_image : List[FImage] = []
        out_target_guide : List[FImage] = []

        for i, (image_path, pair_path) in enumerate(job.items):
            try:
                image = FImage.from_file(image_path)
                target_guide = FImage.from_file(pair_path)
            except Exception as e:
                yield ax.cancel(error=e)

            offset_transform_params = lib_aug.TransformParams(  tx=job.offset_tx,
                                                                ty=job.offset_ty,
                                                                scale=job.offset_scale,
                                                                rot_deg=job.offset_rot_deg)
            transform_params = lib_aug.TransformParams( tx=nprnd.uniform(-job.rnd_tx_var, job.rnd_tx_var),
                                                        ty=nprnd.uniform(-job.rnd_ty_var, job.rnd_ty_var),
                                                        scale=nprnd.uniform(-job.rnd_scale_var, job.rnd_scale_var),
                                                        rot_deg=nprnd.uniform(-job.rnd_rot_deg_var, job.rnd_rot_deg_var))
            
            
            geo_aug = lib_aug.Geo(offset_transform_params=offset_transform_params, transform_params=transform_params)

            input_image  = geo_aug.transform(image.ch(IC), IW, IH, center_fit=True, transform_intensity=job.transform_intensity, interp=FImage.Interp.LANCZOS4, deform_intensity=job.image_deform_intensity, border=FImage.Border.REPLICATE)
            target_image = geo_aug.transform(image.ch(OC), OW, OH, center_fit=True, transform_intensity=job.transform_intensity, interp=FImage.Interp.LANCZOS4, deform_intensity=0.0, border=FImage.Border.REPLICATE)
            target_guide = geo_aug.transform(target_guide.ch(OC), OW, OH, center_fit=True, transform_intensity=job.transform_intensity, interp=FImage.Interp.LINEAR,   deform_intensity=0.0, border=FImage.Border.CONSTANT)

            # Augmentations
            if job.rnd_flip and nprnd.rand() < 0.4:
                input_image  = input_image.h_flip()
                target_image = target_image.h_flip()
                target_guide = target_guide.h_flip()
                
            if rnd_r_exposure_var != 0.0 or rnd_g_exposure_var != 0.0 or rnd_b_exposure_var != 0.0:
                r = nprnd.uniform(-rnd_r_exposure_var, rnd_r_exposure_var)
                g = nprnd.uniform(-rnd_r_exposure_var, rnd_g_exposure_var)
                b = nprnd.uniform(-rnd_r_exposure_var, rnd_b_exposure_var)
                input_image = input_image.channel_exposure((b,g,r))
                target_image = target_image.channel_exposure((b,g,r))


            out_image_paths.append(image_path)
            out_input_image.append(input_image)
            out_target_image.append(target_image)
            out_target_guide.append(target_guide)

            yield ax.sleep(0)

        return MxDataGenerator.GenResult(image_paths = out_image_paths,
                                         input_image = out_input_image,
                                         target_image = out_target_image,
                                         target_guide = out_target_guide,
                                        )
