from __future__ import annotations

import math
import shutil
import sys
import uuid
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, Tuple

import numpy as np

from core import ax, mx
from core.lib import facedesc as fd
from core.lib.collections import FDict, FIndices, HFDict
from core.lib.dataset.FSIP import FSIP
from core.lib.hash import Hash64
from core.lib.image import FImage, ImageFormatType
from core.lib.image.compute import find_nearest_hist
from core.lib.math import FRectf, FVec2f


class MxDatasetEditor(mx.Disposable):

    class SortByMethod(StrEnum):
        Histogram_similarity = '@(Histogram_similarity)'
        Perceptual_dissimilarity = '@(Perceptual_dissimilarity)'
        Face_similarity_reference = '@(Face_similarity_reference)'
        Face_similarity_clustering = '@(Face_similarity_clustering)'
        Face_pitch = '@(Face_pitch)'
        Face_yaw = '@(Face_yaw)'
        Face_source_size = '@(Face_source_size)'
        Source_sequence_number = '@(Source_sequence_number)'

    def __init__(self, state : FDict|None = None, open_path_once : Path|None = None):
        super().__init__()
        self._state = HFDict(state)
        self._open_path_once = open_path_once

        self._main_thread = ax.get_current_thread()
        self._thread_pool = ax.ThreadPool().dispose_with(self)

        self._processing_fg = ax.FutureGroup().dispose_with(self)

        self._mx_info = mx.TextEmitter().dispose_with(self)

        self._mx_processing_progress = mx.Progress().dispose_with(self)
        self._mx_processing_progress.mx_active.reflect(lambda active, enter, bag=mx.Disposable().dispose_with(self):
                                                       self._ref_processing_progress_active(active, enter, bag))

    @property
    def mx_info(self) -> mx.ITextEmitter_v:
        """various info and errors posted during processes"""
        return self._mx_info

    @property
    def mx_processing_progress(self) -> mx.IProgress_rv: return self._mx_processing_progress
    @property
    def mx_dataset_path(self) -> mx.IPath_v:
        """avail if mx_processing_progress.mx_active==False"""
        return self._mx_dataset_path

    # Below avail if mx_processing_progress.mx_active==False and mx_dataset_path.mx_opened
    @property
    def fsip(self) -> FSIP:return self._fsip
    @property
    def mx_move_paths(self) -> Sequence[mx.IPath_v]: return self._mx_move_paths
    @property
    def mx_trash_path(self) -> mx.IPath_v: return self._mx_trash_path


    def _ref_processing_progress_active(self, active : bool, enter : bool, bag : mx.Disposable):
        if enter:
            if not active:
                sub_bag = mx.Disposable().dispose_with(bag)
                self._mx_dataset_path = mx.Path(mx.Path.Config( allow_open=True, dir=True, desc='Dataset directory'),
                                                        on_close=lambda bag=sub_bag: bag.dispose_items(),
                                                        on_open=lambda path, bag=sub_bag: self._mx_faceset_path_on_open(path, bag),
                                        ).dispose_with(self)

                if self._open_path_once is not None:
                    self._mx_dataset_path.open(self._open_path_once)
                    self._open_path_once = None
        else:
            bag.dispose_items()

    def _mx_faceset_path_on_open(self, path : Path, bag : mx.Disposable) -> bool:
        err = None
        try:
            fsip = FSIP.open(path)
        except Exception as e:
            err=e

        if err is not None:
            self._mx_info.emit(f'@(Error) {path}: {err}')
            return False

        self._fsip = fsip

        mx_move_paths = []
        for i in range(1,3+1):
            move_path = mx.Path( mx.Path.Config(dir=True, allow_open=True, allow_new=True, desc=f'Directory{i}') ).dispose_with(bag)
            move_path.new(path.parent / f'{path.name}_{i}')
            mx_move_paths.append(move_path)
        self._mx_move_paths = mx_move_paths

        self._mx_trash_path = mx.Path( mx.Path.Config(dir=True, allow_open=True, allow_new=True, desc=f'Trash directory') ).dispose_with(bag)
        self._mx_trash_path.new(path.parent / f'{path.name}_trash')

        return True

    def move_items(self, item_ids : FIndices|Iterable|int|None, path_id : int) -> FSIP.MoveResult:
        """
        avail if mx_dataset_path.mx_opened and mx_processing_progress.mx_active==False

        move `item_ids` to mx_move_path

        raise NO errors
        """
        mx_move_path = self._mx_move_paths[path_id]
        if (move_path := mx_move_path.get()) is not None:
            result = self._fsip.move_items(item_ids, move_path)
            for error in result.errors:
                self._mx_info.emit(f'@(Error): {error}')
            return result
        return FSIP.MoveResult(error=None, moved_ids=FIndices())

    @ax.task
    def process_cancel(self):
        """avail if mx_processing_progress.mx_active"""
        yield ax.switch_to(self._main_thread)
        self._processing_fg.cancel_all(error=self._ProcessCancelByUser())

    class _ProcessCancelByUser(Exception): ...

    @ax.task
    def process_sort_by_face_similarity(self, reference_item_id : int|None = None):
        @ax.task
        def _process(fsip : FSIP):
            yield ax.switch_to(self._main_thread)
            yield ax.wait(fut := self._process_load_metas(fsip))
            yield ax.wait(fut := self._process_sort_by_face_similarity(fut.result, reference_item_id))
            yield ax.wait(fut := self._process_fsip_reorder_items(fsip, fut.result))

        yield ax.propagate(self._process_fsip(_process))

    @ax.task
    def process_sort(self, method : SortByMethod):
        """avail if mx_processing_progress.mx_active==False and mx_dataset_path.mx_opened"""

        @ax.task
        def _process(fsip : FSIP):
            yield ax.switch_to(self._main_thread)

            # Calc indexes to be renamed
            if method == self.SortByMethod.Histogram_similarity:
                yield ax.wait(fut := self._process_sort_by_hist_sim(fsip))

            elif method == self.SortByMethod.Perceptual_dissimilarity:
                yield ax.wait(fut := self._process_sort_by_perc_dissim(fsip))

            elif method == self.SortByMethod.Face_pitch:
                yield ax.wait(fut := self._process_load_metas(fsip))
                yield ax.wait(fut := self._process_sort_by_meta(fut.result, lambda meta: self._get_face_pose_value(meta, 0)))

            elif method == self.SortByMethod.Face_yaw:
                yield ax.wait(fut := self._process_load_metas(fsip))
                yield ax.wait(fut := self._process_sort_by_meta(fut.result, lambda meta: self._get_face_pose_value(meta, 1)))

            elif method == self.SortByMethod.Face_source_size:
                yield ax.wait(fut := self._process_load_metas(fsip))
                yield ax.wait(fut := self._process_sort_by_meta(fut.result, lambda meta: self._get_face_source_size_value(meta) ))

            elif method == self.SortByMethod.Source_sequence_number:
                yield ax.wait(fut := self._process_load_metas(fsip))
                yield ax.wait(fut := self._process_sort_by_meta(fut.result, lambda meta: self._get_source_sequence_number_value(meta), reverse=False))

            yield ax.wait(fut := self._process_fsip_reorder_items(fsip, fut.result))

        yield ax.propagate(self._process_fsip(_process))


    @ax.task
    def _process_sort_by_hist_sim(self, fsip : FSIP) -> np.ndarray:
        """"""
        @ax.task
        def _send_inc_progress():
            yield ax.switch_to(self._main_thread)
            self._mx_processing_progress.inc()

        def _load_histogram(fsip : FSIP, item_id : int, item_path : Path) -> np.ndarray:
            return FImage.from_file(item_path).histogram(normalized=True)

        yield ax.switch_to(self._thread_pool)

        yield ax.wait(fut := self._process_load(fsip, _load_histogram))

        hist_ar = fut.result
        if len([1 for hist in hist_ar if hist is None]) != 0:
            yield ax.cancel()

        item_count = fsip.item_count
        yield ax.switch_to(self._main_thread)

        self._mx_processing_progress.set_caption('@(Sorting)...').set(0, item_count)

        yield ax.switch_to(self._thread_pool)

        hist_ar = np.stack(hist_ar)
        idxs = np.arange(item_count, dtype=np.int32)

        for i in range(item_count-1):
            n_idx = find_nearest_hist(hist_ar, i, i+1, item_count)

            hist_ar[[i+1, n_idx]] = hist_ar[[n_idx, i+1]]
            idxs[[i+1, n_idx]] = idxs[[n_idx, i+1]]

            _send_inc_progress()

        return idxs


    @ax.task
    def _process_sort_by_perc_dissim(self, fsip : FSIP) -> Sequence[int]:
        """"""
        def _load_perc_hash(fsip : FSIP, item_id : int, item_path : Path) -> Hash64:
            return FImage.from_file(item_path).get_perc_hash()

        yield ax.switch_to(self._thread_pool)

        yield ax.wait(fut := self._process_load(fsip, _load_perc_hash))

        hash_ar = fut.result
        if len([1 for hash in hash_ar if hash is None]) != 0:
            yield ax.cancel()

        yield ax.switch_to(self._main_thread)

        self._mx_processing_progress.set_caption('@(Sorting)...').set_inf()

        yield ax.switch_to(self._thread_pool)

        return Hash64.sorted_by_dissim(hash_ar)


    def _get_face_pose_value(self, meta : fd.FEmbedAlignedFaceInfo, pyr_mode : int) -> float|None:
        if (pose := meta.aligned_face.annotations.get_pose()) is not None:
            if pyr_mode == 0:
                return pose.pitch
            elif pyr_mode == 1:
                return pose.yaw
        return None


    def _get_face_source_size_value(self, meta : fd.FEmbedAlignedFaceInfo) -> float|None:
        aligned_face = meta.aligned_face
        a_rect = FRectf(FVec2f(0,0), aligned_face.image_size)
        g_rect = a_rect.transform(aligned_face.mat.inverted)
        return g_rect.area

    def _get_source_sequence_number_value(self, meta : fd.FEmbedAlignedFaceInfo) -> Any|None:
        source_frame = meta.source_frame
        if (frame_idx := source_frame.frame_idx) is not None:
            return f'{frame_idx:07}'
        if (media_url := source_frame.media_url) is not None:
            return media_url
        return None


    @ax.task
    def _process_sort_by_face_similarity(self, metas : Sequence[fd.FEmbedAlignedFaceInfo|None], ref_item_id : int|None) -> Sequence[int]:
        yield ax.switch_to(self._thread_pool)

        vectors = np.zeros( (len(metas), 256), np.float32)
        for item_id, meta in enumerate(metas):
            if meta is not None and \
              (anno_id := meta.aligned_face.annotations.get_first_by_class(fd.FAnnoID)) is not None:
                vectors[item_id] = anno_id.vector

        yield ax.switch_to(self._main_thread)

        count = vectors.shape[0]-1

        self._mx_processing_progress.set_caption('@(Sorting)...').set(0, count)

        @ax.task
        def send_inc_progress():
            yield ax.switch_to(self._main_thread)
            self._mx_processing_progress.inc()

        yield ax.switch_to(self._thread_pool)

        if ref_item_id is not None:
            # Sort by reference vector
            ref_vector = vectors[ref_item_id][:,None]
            return np.argsort( -(vectors @ ref_vector)[:,0] )
        else:
            # Sort by moving mean cluster

            ids = np.arange(vectors.shape[0])
            for id in range(count):
                # Average prev 32
                avg = vectors[ max(0,id-32):id+1]
                avg = avg.sum(0)

                avg = avg / max(1, np.sqrt( np.square(avg).sum(-1) ))
                ref_vector = avg[:,None]

                test_id = id+1
                sim = vectors[test_id:] @ ref_vector

                id_max = test_id+np.argmax(sim)

                ids[[test_id, id_max]] = ids[[id_max, test_id]]
                vectors[[test_id, id_max]] = vectors[[id_max, test_id]]

                send_inc_progress()

        return tuple(ids)


    @ax.task
    def import_dataset(self, path : Path):

        @ax.task
        def _process(fsip : FSIP):
            yield ax.switch_to(self._main_thread)

            self._mx_processing_progress.set_caption('@(Import)...').set_inf()

            yield ax.switch_to(self._thread_pool)

            other_fsip = None
            try:
                other_fsip = FSIP.open(path)
            except Exception as e:
                yield ax.switch_to(self._main_thread)
                self._mx_info.emit(f'@(Error) {path} : {e}')

            if other_fsip is not None:
                indices_w_names = { item_id : name for item_id, name in enumerate(fsip.generate_free_names(other_fsip.item_count)) }
                copy_result = other_fsip.copy_items(indices_w_names, fsip.root)

                yield ax.switch_to(self._main_thread)

                for err in copy_result.errors:
                    self._mx_info.emit(f'@(Error) : {err}')

        yield ax.propagate(self._process_fsip(_process))


    @ax.task
    def export_metadata(self, path : Path):

        @ax.task
        def _process(fsip : FSIP):
            yield ax.wait(fut := self._process_load_metas(fsip))
            metas = fut.result

            meta_dict = { fsip.get_item_name(item_id) : meta.get_state()
                          for item_id, meta in enumerate(metas) if meta is not None }

            # Dump to file
            self._mx_processing_progress.set_caption('@(Export_metadata)...').set_inf()

            yield ax.switch_to(self._thread_pool)

            try:
                FDict(meta_dict).dump_to_file(path)
            except Exception as e:
                yield ax.switch_to(self._main_thread)
                self._mx_info.emit(f'@(Error) {path} : {e}')

        yield ax.propagate(self._process_fsip(_process))



    @ax.task
    def import_metadata(self, path : Path):


        @ax.task
        def _embed_face_meta(image_path : Path, meta_state : FDict|None):
            yield ax.switch_to(self._thread_pool)

            if meta_state is not None and \
              (meta := fd.FEmbedAlignedFaceInfo.from_state(meta_state)) is not None:

                try:
                    img = FImage.from_file(image_path)
                except Exception as e:
                    yield ax.cancel(e)

                meta = meta.set_aligned_face(meta.aligned_face.resize(img.size))

                try:
                    meta.embed_to(image_path)
                except Exception as e:
                    yield ax.cancel(e)


        @ax.task
        def _process(fsip : FSIP):
            yield ax.switch_to(self._main_thread)

            self._mx_processing_progress.set_caption('@(Loading)...').set_inf()

            yield ax.switch_to(self._thread_pool)

            # Load meta_dict
            meta_dict = None
            try:
                meta_dict = FDict.from_file(path)
            except Exception as e:
                yield ax.switch_to(self._main_thread)
                self._mx_info.emit(f'@(Error) {path} : {e}')

            yield ax.switch_to(self._main_thread)

            self._mx_processing_progress.set_caption('@(Import_metadata)...').set(0, fsip.item_count)

            if meta_dict is not None:
                # Embed meta to all images
                for result in ax.FutureGenerator(
                            (  ( _embed_face_meta(fsip.get_item_path(item_id), meta_dict.get(fsip.get_item_name(item_id), None)),
                                item_id)
                                for item_id in range(fsip.item_count) ),
                            max_parallel=self._thread_pool.count*2 ):

                    if result is not None:
                        fut, item_id = result
                        self._mx_processing_progress.inc()

                        if not fut.succeeded:
                            self._mx_info.emit(f'@(Error) {fsip.get_item_path(item_id)} : {fut.error}')
                    else:
                        yield ax.sleep(0)

        yield ax.propagate(self._process_fsip(_process))


    @ax.task
    def delete_metadata(self):
        @ax.task
        def _delete_face_meta(image_path : Path) -> fd.FEmbedAlignedFaceInfo|None:
            yield ax.switch_to(self._thread_pool)
            try:
                fd.FEmbedAlignedFaceInfo.remove_from(image_path)
            except Exception as e:
                yield ax.cancel(e)

        @ax.task
        def _process(fsip : FSIP):
            yield ax.switch_to(self._main_thread)

            self._mx_processing_progress.set_caption('@(Deleting_metadata)...').set(0, fsip.item_count)

            for value in ax.FutureGenerator(
                        (  (_delete_face_meta(fsip.get_item_path(item_id)), item_id)
                            for item_id in range(fsip.item_count) ),
                        max_parallel=self._thread_pool.count*2 ):
                if value is not None:
                    fut, item_id = value
                    self._mx_processing_progress.inc()

                    if not fut.succeeded:
                        self._mx_info.emit(f'@(Error) {fsip.get_item_path(item_id)} : {fut.error}')
                else:
                    yield ax.sleep(0)

        yield ax.propagate(self._process_fsip(_process))



    @ax.task
    def filter_by_best(self, target_size : int):
        yield ax.switch_to(self._main_thread)

        if (trash_path := self._mx_trash_path.get()) is None:
            self._mx_info.emit('Trash path is not set.')
            yield ax.cancel()

        @dataclass
        class Data:
            perc : Hash64
            pitch : float
            yaw : float

        def _load(fsip : FSIP, item_id : int, item_path : Path) -> Data|None:
            if (meta := fd.FEmbedAlignedFaceInfo.from_embed(item_path)) is None:
                raise Exception('@(No_meta_data)')
            return Data(perc = FImage.from_file(item_path).get_perc_hash(),
                        pitch = self._get_face_pose_value(meta, 0),
                        yaw = self._get_face_pose_value(meta, 1),   )


        @ax.task
        def _process(fsip : FSIP):
            yield ax.switch_to(self._thread_pool)

            yield ax.wait(fut := self._process_load(fsip, _load))
            datas = fut.result

            if len([1 for data in datas if data is None]) != 0:
                yield ax.cancel()

            yield ax.switch_to(self._main_thread)

            self._mx_processing_progress.set_caption('@(Sorting)...').set_inf()

            yield ax.switch_to(self._thread_pool)

            # Gather
            pitch_cells = 16
            yaw_cells = 128
            min_pitch = min_yaw = sys.maxsize
            max_pitch = max_yaw = -sys.maxsize

            for item_id, data in enumerate(datas):
                pitch = data.pitch
                yaw = data.yaw
                min_pitch = min(min_pitch, pitch)
                max_pitch = max(max_pitch, pitch)
                min_yaw = min(min_yaw, yaw)
                max_yaw = max(max_yaw, yaw)

            pitch_per_cell = (max_pitch-min_pitch) / (pitch_cells-1)
            yaw_per_cell = (max_yaw-min_yaw) / (yaw_cells-1)

            # Gather items by yaw/pitch cells
            data_by_pitch_yaw = [ [ [] for _ in range(pitch_cells) ] for _ in range(yaw_cells) ]

            for item_id, data in enumerate(datas):
                yaw_id = int( (data.yaw-min_yaw) / yaw_per_cell )
                pitch_id = int( (data.pitch-min_pitch) / pitch_per_cell )
                data_by_pitch_yaw[yaw_id][pitch_id].append(item_id)

            # Sort items in every yaw/pitch cell by perceptual dissim
            for yaw_id in range(yaw_cells):
                for pitch_id in range(pitch_cells):
                    item_ids = data_by_pitch_yaw[yaw_id][pitch_id]

                    if len(item_ids) > 1:
                        hash_ar = [ datas[item_id].perc for item_id in item_ids ]
                        new_ids = Hash64.sorted_by_dissim(hash_ar)
                        data_by_pitch_yaw[yaw_id][pitch_id] = [item_ids[new_id] for new_id in new_ids]


            pitch_cells//2

            out_indices = []
            while True:
                n = 0

                for pitch_id in [8, 0, 15, 3, 12, 6, 10, 1, 14, 4, 11, 7, 9, 2, 13, 5]:
                    if len(out_indices) + yaw_cells >= target_size:
                        break

                    for yaw_id in range(yaw_cells):
                        ar = data_by_pitch_yaw[yaw_id][pitch_id]

                        if len(ar) != 0:
                            item_id = ar.pop()
                            n += 1
                            out_indices.append(item_id)

                if n == 0:
                    break



            all_indices = FIndices.from_range(fsip.item_count)

            # Trash indices
            trash_indices = all_indices.discard(out_indices)
            yield ax.wait(self._process_fsip_move_reorder_items(fsip, trash_indices, trash_path))

            # Retrieve out_datas for out_indices
            out_datas = [ datas[id] for id in all_indices.discard(trash_indices).to_list() ]
            out_indices = all_indices.discard(trash_indices, shift=True)

            # Sort out_indices by yaw
            out_indices = sorted([ (id, out_datas[id].yaw) for id in out_indices], key=lambda x: x[1], reverse=True)
            out_indices = [ id for id,_ in out_indices]

            yield ax.wait(self._process_fsip_reorder_items(fsip, out_indices))


        yield ax.propagate(self._process_fsip(_process))


    @ax.task
    def generate_mask_from_lmrks(self, pair_type : str):
        yield ax.switch_to(self._main_thread)

        def _process_item(fsip : FSIP, item_id : int, item_path : Path):
            if (meta := fd.FEmbedAlignedFaceInfo.from_embed(item_path)) is not None:
                aligned_face = meta.aligned_face
                if (lmrk := aligned_face.annotations.get_first_by_class_prio([fd.FAnnoLmrk2D106, fd.FAnnoLmrk2D68])) is not None:
                    W, H = aligned_face.image_size

                    pair_dir_path = fsip.info.get_pair_dir_path(pair_type)

                    mask = lmrk.generate_mask(W, H)
                    mask.save(pair_dir_path / f'{item_path.stem}.unk', ImageFormatType.PNG)

            else:
                raise Exception('@(No_meta_data)')

        @ax.task
        def _process(fsip : FSIP):
            try:
                fsip.delete_pair_type(pair_type)
            except:
                ...
            fsip.add_pair_type(pair_type)



            yield ax.wait(self._process_load(fsip, _process_item, caption='@(Processing)...'))

        yield ax.propagate(self._process_fsip(_process))

    @ax.task
    def realign(self, coverage : float = 1.0, y_offset : float = 0.0, y_axis_offset : float = 0.0, min_image_size : int = 128, max_image_size : int = 1024 ):
        yield ax.switch_to(self._main_thread)

        def _change_face_coverage(fsip : FSIP, item_id : int, item_path : Path):
            if (meta := fd.FEmbedAlignedFaceInfo.from_embed(item_path)) is not None:
                aligned_face = meta.aligned_face
                source_face = meta.source_face

                # realign in the space of aligned_face
                new_aligned_face = source_face.transform(aligned_face.mat)\
                                               .align(  coverage=coverage,
                                                        uy_offset=y_offset,
                                                        y_axis_offset=y_axis_offset,
                                                        min_image_size=min_image_size,
                                                        max_image_size=max_image_size)

                # warp and save new_aligned_image
                new_aligned_image = FImage.from_file(item_path)
                new_aligned_image = new_aligned_image.warp_affine(new_aligned_face.mat, new_aligned_face.image_size, interp=FImage.Interp.LANCZOS4)
                new_aligned_image.save(item_path)

                # Set correct mat to transform original source to realigned space
                new_aligned_face = new_aligned_face.set_mat( aligned_face.mat*new_aligned_face.mat )

                # renew meta
                meta = meta.set_aligned_face(new_aligned_face)
                meta.embed_to(item_path)

                # warp pairs
                for pair_type in fsip.pair_types:
                    if (pair_path := fsip.get_pair_path(item_id, pair_type)) is not None:
                        pair = FImage.from_file(pair_path)

                        pair_mat = aligned_face.mat.scale( FVec2f(pair.size)/FVec2f(aligned_face.image_size)).inverted * new_aligned_face.uni_mat.scale(pair.size)

                        pair = pair.warp_affine(pair_mat, pair.size)
                        pair.save(pair_path)

            else:
                raise Exception('@(No_meta_data)')

        @ax.task
        def _process(fsip : FSIP):
            yield ax.wait(self._process_load(fsip, _change_face_coverage, caption='@(Processing)...'))

        yield ax.propagate(self._process_fsip(_process))

    @ax.task
    def change_file_format(self, fmt_type : ImageFormatType, quality : int ):
        yield ax.switch_to(self._main_thread)

        def _change_face_coverage(fsip : FSIP, item_id : int, item_path : Path):
            img = FImage.from_file(item_path)
            meta = fd.FEmbedAlignedFaceInfo.from_embed(item_path)

            new_item_path = img.save(item_path, fmt_type=fmt_type, quality=quality)
            if meta is not None:
                meta.embed_to(new_item_path)

            if item_path != new_item_path:
                item_path.unlink(missing_ok=True)

        @ax.task
        def _process(fsip : FSIP):
            yield ax.wait(self._process_load(fsip, _change_face_coverage, caption='@(Processing)...'))

        yield ax.propagate(self._process_fsip(_process))

    @ax.task
    def create_tsv(self, file_path : Path, item_ids : FIndices|Iterable|int|None = None ):
        @ax.task
        def _process(fsip : FSIP):
            yield ax.switch_to(self._thread_pool)
            yield ax.wait(fut := self._process_load_metas(fsip, item_ids))
            metas = fut.result

            # Collect meta by frame_idx.
            meta_by_frame_idx = {}
            for meta in metas:
                # Non meta frames will be discarded
                if meta is not None:
                    if (source_frame := meta.source_frame) is not None:

                        # Meta without frame_idx or int'able media name will be discarded.
                        if (frame_idx := source_frame.frame_idx) is None:
                            # Try to determine frame_idx from media name
                            if (media_url := source_frame.media_url) is not None:
                                try:
                                    filename = media_url.split('/')[-1]
                                    name = filename.split('.')[0]
                                    frame_idx = int(name)
                                except:
                                    pass

                        if frame_idx is not None:
                            # Pick only first meta referring the same frame_idx.
                            if frame_idx not in meta_by_frame_idx:
                                meta_by_frame_idx[frame_idx] = meta

            # Sort meta by frame_idx
            frame_metas = sorted(tuple(meta_by_frame_idx.items()), key=lambda x: x[0])
            if len(frame_metas) == 0:
                yield ax.cancel()

            # Calc frame_dup for every frame
            frame_metas : Sequence[Tuple[int,int,fd.FEmbedAlignedFaceInfo]] = [ (frame_id, (frame_metas[i+1][0]-frame_id) if i < len(frame_metas)-1 else 1, meta) for i, (frame_id, meta) in enumerate(frame_metas) ]

            # Collect .tsv data
            lines = [ f'position.x\tposition.y\tscale\trotation' ]
            for frame_id, frame_dup, meta in frame_metas:
                aligned_face = meta.aligned_face

                g_rect = FRectf(aligned_face.image_size).transform(aligned_face.mat.inverted)

                p0 = g_rect.p0
                scale = g_rect.width
                rotation = (g_rect.p1-g_rect.p0).atan2() * 180.0 / math.pi

                for _ in range(frame_dup):
                    lines.append(f'{p0.x}\t{p0.y}\t{scale}\t{rotation}')

            file_path.write_text('\n'.join(lines))


        yield ax.propagate(self._process_fsip(_process))




    # COMMON PROCESSING METHODS BELOW
    @ax.task
    def _process_fsip_move_reorder_items(self, fsip : FSIP, indices : Sequence[int]|FIndices, other_fsip_root : Path):
        """move with reorder indices to other_fsip_root renaming if same name exist in other_fsip"""
        yield ax.switch_to(self._main_thread)
        self._mx_processing_progress.set_caption('@(Moving)...').set_inf()
        yield ax.switch_to(self._thread_pool)

        other_fsip_root.mkdir(parents=True, exist_ok=True)

        other_fsip = FSIP.open(other_fsip_root)
        for pair_type in fsip.pair_types:
            other_fsip.add_pair_type(pair_type)

        if isinstance(indices, FIndices):
            indices = indices.to_list()

        fsip.move_items( {item_id : new_item_name
                          for item_id, new_item_name in zip(indices, other_fsip.generate_free_names(len(indices)))}, other_fsip_root )

    @ax.task
    def _process_fsip_reorder_items(self, fsip : FSIP, item_ids : Sequence[int]):
        """keep only reordered item_ids in fsip"""
        tmp_fsip_root = fsip.info.root.parent / f'{fsip.info.root.name}_{uuid.uuid4().hex}'

        yield ax.wait(self._process_fsip_move_reorder_items(fsip, item_ids, tmp_fsip_root))

        try:
            shutil.rmtree(fsip.root)
            shutil.move(tmp_fsip_root, fsip.root)
        except Exception as e:
            yield ax.switch_to(self._main_thread)
            self._mx_info.emit(f'@(Error) {fsip.root} {tmp_fsip_root} : {e}')


    @ax.task
    def _process_sort_by_meta(self, metas : Sequence[fd.FEmbedAlignedFaceInfo|None],
                            func : Callable[ [fd.FEmbedAlignedFaceInfo], Any|None],
                            reverse = True) -> Sequence[int]:
        yield ax.switch_to(self._main_thread)

        self._mx_processing_progress.set_caption('@(Sorting)...').set(0, len(metas))

        meta_idxs = []
        non_meta_idxs = []

        yield ax.switch_to(self._thread_pool)

        @ax.task
        def send_inc_progress():
            yield ax.switch_to(self._main_thread)
            self._mx_processing_progress.inc()

        for idx, meta in enumerate(metas):
            if meta is not None and \
              (value := func(meta)) is not None:
                meta_idxs.append( (idx, value) )
            else:
                non_meta_idxs.append(idx)
            send_inc_progress()

        yield ax.switch_to(self._main_thread)

        self._mx_processing_progress.set_caption('@(Sorting)...').set_inf()

        yield ax.switch_to(self._thread_pool)

        return [ item_id for item_id, value in sorted(meta_idxs, key=lambda v: v[1], reverse=reverse) ] + non_meta_idxs


    @ax.task
    def _process_load_metas(self, fsip : FSIP, indices : FIndices|Iterable|int|None = None) -> Sequence[fd.FEmbedAlignedFaceInfo|None]:

        def _load_face_meta(fsip : FSIP, item_id : int, item_path : Path) -> fd.FEmbedAlignedFaceInfo|None:
            meta = fd.FEmbedAlignedFaceInfo.from_embed(item_path)
            if meta is None:
                raise Exception('@(No_meta_data)')
            return meta

        yield ax.wait(fut := self._process_load(fsip, _load_face_meta, caption='@(Loading_meta_data)...', indices=indices))

        return fut.result

    @ax.task
    def _process_load[T](self, fsip : FSIP, load_func : Callable[ [FSIP, int, Path], T|None ], caption = '@(Loading)...', indices : FIndices|Iterable|int|None = None, max_parallel=None) -> Sequence[T|None]:
        """
            load_func errors will be posted to mx_info
        """
        @ax.task
        def _load_task(item_id : int, item_path : Path) -> fd.FEmbedAlignedFaceInfo|None:
            yield ax.switch_to(self._thread_pool)
            try:
                return load_func(fsip, item_id, item_path)
            except Exception as e:
                yield ax.cancel(e)

        yield ax.switch_to(self._main_thread)

        if indices is None:
            indices = FIndices.from_range(fsip.item_count)
        else:
            indices = FIndices(indices)

        self._mx_processing_progress.set_caption(caption).set(0, indices.count)

        datas = []
        infos = []

        if max_parallel is None:
            max_parallel = self._thread_pool.count*2
        for value in ax.FutureGenerator(
                            (  ( _load_task(item_id, fsip.get_item_path(item_id)), item_id)
                                 for item_id in indices ),
                            max_parallel=max_parallel ):
            if value is not None:
                fut, item_id = value
                if fut.succeeded:
                    data = fut.result
                else:
                    data = None
                    infos.append(f"{fsip.get_item_path(item_id)} : {fut.error}")
                datas.append(data)
                self._mx_processing_progress.inc()
            else:
                yield ax.sleep(0)

        for info in infos:
            self._mx_info.emit(info)

        yield ax.sleep(0.2)

        return datas

    @ax.task
    def _process_fsip(self, task : Callable[ [FSIP], ax.Future]):
        """
        base task for fsip processing

        avail if mx_processing_progress.mx_active==False and mx_dataset_path.mx_opened
        """
        yield ax.switch_to(self._main_thread)

        if not (self._mx_processing_progress.mx_active.get() == False and \
                self._mx_dataset_path.mx_opened.get()):
            yield ax.cancel()

        yield ax.attach_to(self._processing_fg)

        fsip = self._fsip
        self._mx_dataset_path.close()
        self._mx_processing_progress.start()

        try:
            yield ax.wait(fut := task(fsip))
        except ax.TaskFinishException as e:
            if type(e.error) != self._ProcessCancelByUser:
                return

        self._mx_processing_progress.finish()
        self._mx_dataset_path.open(fsip.root)