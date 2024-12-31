from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from fnmatch import fnmatch

from common.FSIP import MxPairType, QxPairTypeAddDel
from common.ImageFormat import MxImageFormat, QxImageFormat
from core import ax, lx, mx, qt, qx
from core.lib import facedesc as fd
from core.lib.collections import FIndices, get_enum_id_by_name
from core.lib.dataset.FSIP import IFSIP_v
from core.lib.image import FImage
from core.lib.math import FVec2i

from .MxDatasetEditor import MxDatasetEditor
from .QxMaskEditor import QxMaskEditor
from .QxTransformEditor import QxTransformEditor


class QxDatasetEditor(qx.QVBox):
    class ViewMode(StrEnum):
        Image = '@(Image)'
        PairedImage = '@(Paired_image)'

    def __init__(self, de : MxDatasetEditor):
        super().__init__()
        self._de = de
        self._main_thread = ax.get_current_thread()
        self._thread_pool = ax.ThreadPool(count=ax.CPU_COUNT*2).dispose_with(self)

        self._mx_settings.reflect(lambda settings, enter, bag=mx.Disposable().dispose_with(self):
                                  self.__ref_settings(settings, enter, bag))


    def __ref_settings(self, settings : qx.QSettings, enter : bool, bag : mx.Disposable):
        if enter:
            self.__settings = settings

            if (wnd := self.get_first_parent_by_class(qx.QMainWIndow)) is not None:
                self._f_post_info = lambda info: wnd.q_info_bar.add_text(info)
            else:
                self._f_post_info = lambda info: ...

            self._mx_info = mx.TextEmitter().dispose_with(self)
            self._mx_info.listen(lambda text: self._f_post_info(text)).dispose_with(bag)
            self._de.mx_info.listen(lambda text: self._f_post_info(text)).dispose_with(bag)

            self._de.mx_processing_progress.mx_active.reflect(lambda active, enter, bag=mx.Disposable().dispose_with(bag):
                                                               self._ref_progress_active(active, enter, bag)).dispose_with(bag)
        else:
            bag.dispose_items()

    def _ref_progress_active(self, active : bool, enter : bool, bag : mx.Disposable):

        if enter:
            if active:

                self.add(qx.QHBox().dispose_with(bag)
                        .add(qx.QMxProgress(self._de.mx_processing_progress).set_show_it_s(True))
                        .add(qx.QPushButton().set_text('@(Cancel)').v_compact().h_compact()
                                .inline(lambda btn: btn.mx_clicked.listen(lambda: self._de.process_cancel()))))
            else:

                self._q_central_panel_vbox = qx.QVBox()

                self.add(qx.QVBox().dispose_with(bag)
                            .add(qx.QMxPathState(self._de.mx_dataset_path).v_compact())
                            .add_spacer(8)
                            .add(self._q_central_panel_vbox) )

                self._de.mx_dataset_path.mx_opened.reflect(lambda opened, enter, bag=mx.Disposable().dispose_with(bag):
                                                          self._ref_mx_dataset_path(opened, enter, bag)).dispose_with(bag)
        else:
            bag.dispose_items()


    def _ref_mx_dataset_path(self, opened : bool, enter : bool, bag : mx.Disposable):
        if enter:
            if opened:
                self._mx_mode = mx.State[DatasetMode|TransformMode|None]().dispose_with(bag)
                self._mx_mode.set( DatasetMode() )
                self._mx_mode.reflect(lambda mode, enter, bag=mx.Disposable().dispose_with(bag):
                                      self._ref_mode(mode, enter, bag) )

            else:
                self._q_central_panel_vbox.add(
                    qx.QVBox().dispose_with(bag)
                        .add(qx.QLabel().set_text('- @(open_the_dataset) -'), align=qx.Align.CenterF))

        else:
            bag.dispose_items()


    def _ref_mode(self, mode : DatasetMode|TransformMode|MaskEditorMode|None, enter : bool, bag : mx.Disposable):
        if enter:
            de = self._de
            fsip = de.fsip
            state = self.__settings.state

            if isinstance(mode, DatasetMode):
                self._mx_sort_by_method = mx.StateChoice[MxDatasetEditor.SortByMethod](availuator=lambda: MxDatasetEditor.SortByMethod).dispose_with(bag)
                self._mx_sort_by_method.set( get_enum_id_by_name(MxDatasetEditor.SortByMethod, state.get('_mx_sort_by_method', None), MxDatasetEditor.SortByMethod.Histogram_similarity) )

                self._mx_pair_type = MxPairType(fsip.root, state=state.get('_mx_pair_type', None) ).dispose_with(bag)
                self._mx_pair_type.listen(lambda *_: self._update_filtered_fsip())

                self._mx_filter_by_pair_type = mx.Flag(state.get('_mx_filter_by_pair_type', False)).dispose_with(bag)
                self._mx_filter_by_pair_type.listen(lambda *_: self._update_filtered_fsip())

                self._mx_filter_name = mx.Text( state.get('_mx_filter_name', '*') ).dispose_with(bag)
                self._mx_filter_name.listen(lambda *_: self._update_filtered_fsip())

                self._mx_show_file_name = mx.Flag(state.get('_mx_show_file_name', True)).dispose_with(bag)
                self._mx_show_file_name.listen(lambda b: self._update_item_view_size())

                self._mx_show_file_ext = mx.Flag(state.get('_mx_show_file_ext', False)).dispose_with(bag)
                self._mx_show_file_ext.listen(lambda b: self._update_item_view_size())

                self._mx_draw_annotations = mx.Flag(state.get('_mx_draw_annotations', False)).dispose_with(bag)
                self._mx_draw_annotations.listen(lambda b: q_item_view.update_items())



                self._mx_thumbnail_size = mx.Number(state.get('_mx_thumbnail_size', 96), mx.Number.Config(min=64, max=1024, step=32)).dispose_with(bag)
                self._mx_thumbnail_size.listen(lambda b: self._update_item_view_size())

                self._mx_view_mode = mx.StateChoice[self.ViewMode](lambda: self.ViewMode).dispose_with(bag)
                self._mx_view_mode.set( get_enum_id_by_name(self.ViewMode, state.get('_mx_view_mode', None), self.ViewMode.Image) )

                self._mx_filter_by_best_target = mx.Number(state.get('_mx_filter_by_best_target', 5120), mx.Number.Config(min=384, max=1024*1024, step=128)).dispose_with(bag)

                self._mx_realign_coverage = mx.Number(state.get('_mx_realign_coverage', 1.80), mx.Number.Config(min=1.0, max=4.0, step=0.05)).dispose_with(bag)

                self._mx_realign_y_offset = mx.Number(state.get('_mx_realign_y_offset', -0.14), config=mx.Number.Config(min=-0.5, max=0.5, step=0.01)).dispose_with(bag)
                self._mx_realign_y_axis_offset = mx.Number(state.get('face_y_axis_offset', 0.0), config=mx.Number.Config(min=-1.0, max=1.0, step=0.01)).dispose_with(bag)

                self._mx_realign_min_image_size = mx.Number(state.get('_mx_realign_min_image_size', 128), mx.Number.Config(min=128, max=2048, step=64)).dispose_with(bag)
                self._mx_realign_min_image_size.listen(lambda s: self._mx_realign_max_image_size.set(s) if self._mx_realign_max_image_size.get() < s else ...)
                self._mx_realign_max_image_size = mx.Number(state.get('_mx_realign_max_image_size', 1024), mx.Number.Config(min=128, max=2048, step=64)).dispose_with(bag)
                self._mx_realign_max_image_size.listen(lambda s: self._mx_realign_min_image_size.set(s) if self._mx_realign_min_image_size.get() > s else ...)

                self._mx_image_format = MxImageFormat(state=state.get('_mx_image_format',None)).dispose_with(bag)

                state_upd = lambda: self.__settings.state.update({
                            '_mx_sort_by_method'    :   self._mx_sort_by_method.get().name,
                            '_mx_pair_type'         :   self._mx_pair_type.get_state(),
                            '_mx_filter_by_pair_type' :   self._mx_filter_by_pair_type.get(),
                            '_mx_filter_name'       :   self._mx_filter_name.get(),
                            '_mx_show_file_name'    :   self._mx_show_file_name.get(),
                            '_mx_show_file_ext'     :   self._mx_show_file_ext.get(),
                            '_mx_draw_annotations'  :   self._mx_draw_annotations.get(),
                            '_mx_thumbnail_size'    :   self._mx_thumbnail_size.get(),
                            '_mx_view_mode'         :   self._mx_view_mode.get().name,
                            '_mx_filter_by_best_target' :   self._mx_filter_by_best_target.get(),
                            '_mx_realign_coverage'      :   self._mx_realign_coverage.get(),
                            '_mx_realign_y_offset'      :   self._mx_realign_y_offset.get(),
                            '_mx_realign_y_axis_offset' :   self._mx_realign_y_axis_offset.get(),
                            '_mx_realign_min_image_size' :  self._mx_realign_min_image_size.get(),
                            '_mx_realign_max_image_size' :  self._mx_realign_max_image_size.get(),
                            '_mx_image_format' :  self._mx_image_format.get_state(), })

                self.__settings.ev_update.listen(state_upd).dispose_with(bag)
                mx.CallOnDispose(state_upd).dispose_with(bag)

                def start_transform_item():
                    marked_items = q_item_view.model.marked_items
                    selected_items = q_item_view.model.selected_items
                    if selected_items.count == 1:
                        self._mx_mode.set( TransformMode(fsip=self._filtered_fsip,
                                                         item_id=selected_items.to_list()[0],
                                                         marked_items=marked_items,
                                                         selected_items=selected_items,
                                                         ) )

                def start_mask_editor():
                    marked_items = q_item_view.model.marked_items
                    selected_items = q_item_view.model.selected_items
                    if selected_items.count == 1:
                        pair_type = self._mx_pair_type.get()
                        if pair_type == MxPairType.NO_PAIR:
                            pair_type = None

                        self._mx_mode.set( MaskEditorMode(fsip=self._filtered_fsip,
                                                          item_id=selected_items.to_list()[0],
                                                          pair_type=pair_type,
                                                          marked_items=marked_items,
                                                          selected_items=selected_items,
                                                         ) )

                q_splitter = qx.QSplitter().dispose_with(bag)

                shortcut_prev_pair_type = qx.QShortcut(qt.QKeyCombination(qt.Qt.KeyboardModifier.ControlModifier, qt.Qt.Key.Key_Tab)).set_parent(q_splitter).inline(lambda shortcut: shortcut.mx_press.listen(lambda: self._mx_pair_type.set_prev() ))
                shortcut_next_pair_type = qx.QShortcut(qt.QKeyCombination(qt.Qt.Key.Key_Tab)).set_parent(q_splitter).inline(lambda shortcut: shortcut.mx_press.listen(lambda: self._mx_pair_type.set_next() ))
                shortcut_toggle_view_mode = qx.QShortcut(qt.QKeyCombination(qt.Qt.Key.Key_F3)).set_parent(q_splitter).inline(lambda shortcut: shortcut.mx_press.listen(lambda: self._mx_view_mode.set_next() ))
                shortcut_toggle_draw_annotations = qx.QShortcut( qt.QKeyCombination(qt.Qt.Key.Key_Agrave)).set_parent(q_splitter).inline(lambda shortcut: shortcut.mx_press.listen( lambda: self._mx_draw_annotations.toggle()  ))
                shortcut_transform_item = qx.QShortcut(qt.QKeyCombination(qt.Qt.Key.Key_F5)).set_parent(q_splitter).inline(lambda shortcut: shortcut.mx_press.listen(lambda: start_transform_item()))
                shortcut_mask_editor    = qx.QShortcut(qt.QKeyCombination(qt.Qt.Key.Key_F6)).set_parent(q_splitter).inline(lambda shortcut: shortcut.mx_press.listen(lambda: start_mask_editor()))



                q_item_view = self._q_item_view = qx.QCachedGridItemView(self._on_get_item_pixmap).dispose_with(bag)
                q_item_view_shortcuts = qx.QGridItemViewShortcuts(q_item_view).set_parent(q_item_view)

                self._mx_view_mode.listen(lambda view_mode, enter: q_item_view.update_items() if enter else ...)

                delete_meta_dlg = self._wipe_dir_dlg = qx.QInplaceDialog()
                delete_meta_dlg.q_label_title.set_text('@(Are_you_sure)')
                delete_meta_dlg.q_btn_ok.mx_clicked.listen(lambda: self._de.delete_metadata())

                qx.QApplication.instance().mx_language.reflect(lambda lang: (setattr(self, '_lang', lang),
                                                                             q_item_view.update_items())).dispose_with(bag)

                q_total_items_label = qx.QLabel()
                q_selected_items_label = qx.QLabel()
                q_marked_items_label = qx.QLabel()
                q_sort_params_vbox = qx.QVBox()
                q_move_paths_vbox = qx.QVBox()

                q_item_view.model.selected_items

                self._q_central_panel_vbox.add(
                    q_splitter
                        .add(qx.QVScrollArea().h_compact().set_widget(
                            qx.QVBox().set_spacing(4).v_compact()


                                .add(qx.QCollapsibleVBox().open().set_text('@(View)').inline(lambda collapsible: collapsible.content_vbox
                                    .add(qx.QGrid().set_spacing(1)
                                        .row(0)
                                            .add(qx.QLabel().set_text('@(Pair_type)'), align=qx.Align.RightF)
                                            .add(QxPairTypeAddDel(self._mx_pair_type, fsip)
                                                    .inline(lambda q_pair_type: q_pair_type.mx_error.listen(lambda err: self._mx_info.emit(f'@(Error): {err}'))))

                                        .next_row()
                                            .add(None)
                                            .add(qx.QHBox()
                                                .add(qx.QPushButton().set_text(f"@(Previous) {qx.hfmt.colored_shortcut_keycomb(shortcut_prev_pair_type)}")
                                                                .inline(lambda btn: btn.mx_clicked.listen(lambda: shortcut_prev_pair_type.press())))
                                                .add(qx.QPushButton().set_text(f"@(Next) {qx.hfmt.colored_shortcut_keycomb(shortcut_next_pair_type)}")
                                                            .inline(lambda btn: btn.mx_clicked.listen(lambda: shortcut_next_pair_type.press()))))

                                        .next_row()
                                            .add(qx.QLabel().set_text('@(Mode)'), align=qx.Align.RightF)
                                            .add(qx.QComboBoxMxStateChoice(self._mx_view_mode))
                                        .next_row()
                                            .add(None)
                                            .add(qx.QPushButton().set_text(f"@(Toggle) {qx.hfmt.colored_shortcut_keycomb(shortcut_toggle_view_mode)}")
                                                        .inline(lambda btn: btn.mx_clicked.listen(lambda: shortcut_toggle_view_mode.press())))
                                        .next_row()
                                            .add(qx.QGrid().set_spacing(1)
                                                    .row(0)
                                                        .add(qx.QLabel().set_text('@(Thumbnail_size)'), align=qx.Align.RightF)
                                                        .add(qx.QDoubleSpinBoxMxNumber(self._mx_thumbnail_size), align=qx.Align.LeftF)

                                                    .next_row()
                                                        .add(qx.QLabel().set_text('@(Show_file_name)'), align=qx.Align.RightF)
                                                        .add(qx.QCheckBoxMxFlag(self._mx_show_file_name))
                                                    .next_row()
                                                        .add(qx.QLabel().set_text('@(Show_file_extension)'), align=qx.Align.RightF)
                                                        .add(qx.QCheckBoxMxFlag(self._mx_show_file_ext))
                                                    .next_row()
                                                        .add(qx.QLabel().set_text('@(Draw_annotations)'), align=qx.Align.RightF)
                                                        .add( qx.QHBox()
                                                                .add(qx.QCheckBoxMxFlag(self._mx_draw_annotations))
                                                                .add(qx.QLabel().set_text(f'{qx.hfmt.colored_shortcut_keycomb(shortcut_toggle_draw_annotations)}'))
                                                            , align=qx.Align.LeftF)

                                                    .grid()
                                                 , col_span=2)

                                        .grid())))

                                .add(qx.QCollapsibleVBox().open().set_text('@(Filter)').inline(lambda collapsible: collapsible.content_vbox
                                    .add(qx.QGrid().set_spacing(1)
                                        .row(0)
                                            .add(qx.QLabel().set_text('@(by_pair_type)'), align=qx.Align.RightF)
                                            .add(qx.QCheckBoxMxFlag(self._mx_filter_by_pair_type))
                                        .next_row()
                                            .add(qx.QLabel().set_text('@(by_name)'), align=qx.Align.RightF)
                                            .add(qx.QLineEditMxText(self._mx_filter_name, on_editing_finished=True))
                                        .grid())))

                                .add(qx.QCollapsibleVBox().set_text('@(Edit)').inline(lambda collapsible: collapsible.content_vbox
                                    .add(qx.QGrid().set_spacing(1)
                                        .row(0)
                                            .add(qx.QHBox()
                                                .add(qx.QPushButton().set_text(f"@(Transform) {qx.hfmt.colored_shortcut_keycomb(shortcut_transform_item)}")
                                                            .inline(lambda btn: btn.mx_clicked.listen(lambda: shortcut_transform_item.press())))
                                                .add(qx.QPushButton().set_text(f"@(Mask) {qx.hfmt.colored_shortcut_keycomb(shortcut_mask_editor)}")
                                                            .inline(lambda btn: btn.mx_clicked.listen(lambda: shortcut_mask_editor.press()))))
                                        .grid())))



                                .add(qx.QCollapsibleVBox().set_text('@(Items)').inline(lambda collapsible: collapsible.content_vbox
                                    .add(qx.QGrid().set_spacing(1)
                                        .row(0)

                                            .add(qx.QHBox()
                                                    .add( qx.QLabel().set_text('@(Total)'), align=qx.Align.RightF)
                                                    .add( q_total_items_label.set_font(qx.FontDB.Digital), align=qx.Align.LeftF)
                                                    , align=qx.Align.CenterF
                                                )

                                            .add( qx.QVBox()
                                                    .add(qx.QHBox()
                                                            .add( qx.QLabel().set_text('@(Selected)'), align=qx.Align.RightF)
                                                            .add( q_selected_items_label.set_font(qx.FontDB.Digital), align=qx.Align.LeftF)
                                                         )
                                                    .add(qx.QHBox()
                                                        .add( qx.QLabel().set_text('@(Marked)'), align=qx.Align.RightF)
                                                        .add( q_marked_items_label.set_font(qx.FontDB.Digital), align=qx.Align.LeftF)
                                                    ), align=qx.Align.LeftF
                                                 )


                                        .next_row()
                                            .add(qx.QPushButton().set_text(f"@(Select_unselect_all) {qx.hfmt.colored_shortcut_keycomb(q_item_view_shortcuts.shortcut_select_unselect_all)}")
                                                        .inline(lambda btn: btn.mx_clicked.listen(lambda: q_item_view_shortcuts.shortcut_select_unselect_all.press())), col_span=2)
                                        .next_row()
                                            .add(qx.QPushButton().set_text(f"@(Mark_unmark_selected) {qx.hfmt.colored_shortcut_keycomb(q_item_view_shortcuts.shortcut_mark_unmark_selected)}")
                                                        .inline(lambda btn: btn.mx_clicked.listen(lambda: q_item_view_shortcuts.shortcut_mark_unmark_selected.press())), col_span=2)
                                        .next_row()
                                            .add(qx.QPushButton().set_text(f"@(Select_marked) {qx.hfmt.colored_shortcut_keycomb(q_item_view_shortcuts.shortcut_select_marked)}")
                                                        .inline(lambda btn: btn.mx_clicked.listen(lambda: q_item_view_shortcuts.shortcut_select_marked.press())), col_span=2)
                                        .next_row()
                                            .add(qx.QPushButton().set_text(f"@(Invert_selection) {qx.hfmt.colored_shortcut_keycomb(q_item_view_shortcuts.shortcut_invert_selection)}")
                                                        .inline(lambda btn: btn.mx_clicked.listen(lambda: q_item_view_shortcuts.shortcut_invert_selection.press())), col_span=2)
                                        .next_row()
                                            .add( qx.QLabel().set_text('@(Move_selected_to_directory)'), col_span=2)
                                        .next_row()
                                            .add( q_move_paths_vbox, col_span=2)

                                        .grid())))




                                    .add(qx.QCollapsibleVBox().set_text('@(Dataset)').inline(lambda collapsible: collapsible.content_vbox.set_spacing(4)

                                        .add( qx.QPushButton().set_text('@(Import)...').inline(lambda btn:
                                            btn.mx_clicked.listen(lambda:

                                                    qx.QFileDialog().set_parent(q_splitter)
                                                        .set_accept_mode(qx.QFileDialog.AcceptMode.AcceptOpen)
                                                        .set_file_mode(qx.QFileDialog.FileMode.Directory)
                                                        .inline(lambda dlg: dlg.mx_accepted.listen(lambda: ((dataset_path := dlg.selected_files[0]),
                                                                                                                dlg.dispose(),
                                                                                                                self._de.import_dataset(dataset_path),
                                                                                                            ) ))
                                                        .open() )
                                            ))

                                        .add(qx.QCollapsibleVBox().open().set_text('@(Sort)').inline(lambda collapsible: collapsible.content_vbox
                                            .add(qx.QGrid().set_spacing(1)
                                                .row(0)
                                                    .add(qx.QLabel().set_text('@(Method)'), align=qx.Align.RightF)
                                                    .add(qx.QComboBoxMxStateChoice(self._mx_sort_by_method))
                                                .next_row()
                                                    .add(q_sort_params_vbox.hide(), col_span=2)

                                                .next_row()
                                                    .add(qx.QPushButton().set_text('@(Start)')
                                                            #.inline(lambda btn: btn.mx_clicked.listen(lambda: self._de.process_sort(self._mx_sort_by_method.get())))
                                                            .inline(lambda btn: btn.mx_clicked.listen(self._on_start_sort_btn_clicked))
                                                        , col_span=2)
                                                .grid())))


                                        .add(qx.QCollapsibleVBox().open().set_text('@(Filter_by_uniform_distribution)').inline(lambda collapsible: collapsible.content_vbox
                                            .add(qx.QGrid().set_spacing(1)
                                                .row(0)
                                                    .add(qx.QVBox()
                                                            .add(qx.QLabel().set_text('@(Trash_directory)'))
                                                            .add(qx.QMxPathState(self._de.mx_trash_path)) )
                                                .next_row()
                                                    .add(qx.QGrid()
                                                            .row(0)
                                                                .add(qx.QLabel().set_text('@(Target)'), align=qx.Align.RightF)
                                                                .add(qx.QDoubleSpinBoxMxNumber(self._mx_filter_by_best_target), align=qx.Align.LeftF)
                                                            .grid())

                                                .next_row()
                                                    .add(qx.QPushButton().set_text('@(Start)')
                                                            .inline(lambda btn: btn.mx_clicked.listen(lambda: self._de.filter_by_best(self._mx_filter_by_best_target.get())))
                                                        )
                                                .grid())))

                                        .add(qx.QCollapsibleVBox().open().set_text('@(Face_realign)').inline(lambda collapsible: collapsible.content_vbox
                                            .add(qx.QGrid().set_spacing(1)
                                                .row(0 )
                                                    .add(qx.QGrid()
                                                            .row(0)
                                                                .add(qx.QLabel().set_text('@(Face_coverage)'), align=qx.Align.RightF)
                                                                .add(qx.QDoubleSpinBoxMxNumber(self._mx_realign_coverage), align=qx.Align.LeftF)
                                                            .next_row()
                                                                .add(qx.QLabel().set_text('@(Face_Y_offset)'), align=qx.Align.RightF)
                                                                .add(qx.QDoubleSpinBoxMxNumber(self._mx_realign_y_offset), align=qx.Align.LeftF)
                                                            .next_row()
                                                                .add(qx.QLabel().set_text('@(Face_Y_axis_offset)'), align=qx.Align.RightF)
                                                                .add(qx.QDoubleSpinBoxMxNumber(self._mx_realign_y_axis_offset), align=qx.Align.LeftF)
                                                            .next_row()
                                                                .add(qx.QLabel().set_text('@(Image_size)'), align=qx.Align.RightF)
                                                                .add(qx.QHBox()
                                                                        .add(qx.QDoubleSpinBoxMxNumber(self._mx_realign_min_image_size))
                                                                        .add(qx.QLabel().set_text('-'))
                                                                        .add(qx.QDoubleSpinBoxMxNumber(self._mx_realign_max_image_size))

                                                                        , align=qx.Align.LeftF)

                                                            .grid())

                                                .next_row()
                                                    .add(qx.QPushButton().set_text('@(Start)')
                                                            .inline(lambda btn: btn.mx_clicked.listen(lambda: self._de.realign(coverage=self._mx_realign_coverage.get(),
                                                                                                                               y_offset=self._mx_realign_y_offset.get(),
                                                                                                                               y_axis_offset=self._mx_realign_y_axis_offset.get(),
                                                                                                                               min_image_size=self._mx_realign_min_image_size.get(),
                                                                                                                               max_image_size=self._mx_realign_max_image_size.get())))
                                                        )
                                                .grid())))

                                        .add(qx.QCollapsibleVBox().open().set_text('@(Change_file_format)').inline(lambda collapsible: collapsible.content_vbox
                                            .add(qx.QGrid().set_spacing(1)
                                                .row(0)
                                                    .add( QxImageFormat(self._mx_image_format))

                                                .next_row()
                                                    .add(qx.QPushButton().set_text('@(Start)')
                                                            .inline(lambda btn: btn.mx_clicked.listen(lambda: self._de.change_file_format(self._mx_image_format.mx_image_format_type.get(), self._mx_image_format.mx_quality.get())))
                                                        )

                                                .grid())))

                                        .add(qx.QCollapsibleVBox().open().set_text('Generate mask from landmarks').inline(lambda collapsible: collapsible.content_vbox
                                            .add(qx.QGrid().set_spacing(1)
                                                .row(0)
                                                    .add(qx.QPushButton().set_text('@(Start)')
                                                            .inline(lambda btn: btn.mx_clicked.listen(lambda: self._de.generate_mask_from_lmrks(pair_type) if (pair_type := self._mx_pair_type.get()) is not None else ...  ))
                                                        )
                                                .grid())))

                                        .add(qx.QCollapsibleVBox().set_text('@(Metadata)').inline(lambda collapsible: collapsible.content_vbox
                                            .add(qx.QHBox()
                                                .add( qx.QPushButton().set_text('@(Import)...').inline(lambda btn:
                                                        btn.mx_clicked.listen(lambda:

                                                                qx.QFileDialog().set_parent(q_splitter)
                                                                    .set_accept_mode(qx.QFileDialog.AcceptMode.AcceptOpen)
                                                                    .set_file_mode(qx.QFileDialog.FileMode.ExistingFile)
                                                                    .set_filter('(*.meta)')
                                                                    .set_directory(fsip.root)
                                                                    .inline(lambda dlg: dlg.mx_accepted.listen(lambda: ((filepath := dlg.selected_files[0]),
                                                                                                                        dlg.dispose(),
                                                                                                                        self._de.import_metadata(filepath),
                                                                                                                            ) ))
                                                                    .open() )))

                                                .add( qx.QPushButton().set_text('@(Export)...').inline(lambda btn:
                                                        btn.mx_clicked.listen(lambda:

                                                                qx.QFileDialog().set_parent(q_splitter)
                                                                    .set_accept_mode(qx.QFileDialog.AcceptMode.AcceptSave)
                                                                    .set_file_mode(qx.QFileDialog.FileMode.AnyFile)
                                                                    .set_filter('(*.meta)')
                                                                    .set_directory(fsip.root)
                                                                    .inline(lambda dlg: dlg.mx_accepted.listen(lambda: ((filepath := dlg.selected_files[0]),
                                                                                                                        dlg.dispose(),
                                                                                                                        self._de.export_metadata(filepath),
                                                                                                                            ) ))
                                                                    .open() )))

                                                .add( qx.QPushButton().set_text('@(Delete)...').inline(lambda btn:
                                                                btn.mx_clicked.listen(lambda: delete_meta_dlg.popup()))) )

                                        .add(delete_meta_dlg) ))

                                    ))
                                ))
                        .add(q_item_view) )

                q_item_view.mx_model.reflect(lambda m: q_total_items_label.set_text(f'{m.item_count}') ).dispose_with(bag)
                q_item_view.mx_selected_items.reflect(lambda indices: q_selected_items_label.set_text(f'{indices.count}')).dispose_with(bag)
                q_item_view.mx_marked_items.reflect(lambda indices: q_marked_items_label.set_text(f'{indices.count}')).dispose_with(bag)

                for path_id, move_path in enumerate(de.mx_move_paths):
                    q_move_vbox = qx.QVBox()
                    q_move_paths_vbox.add( qx.QHBox().add(qx.QMxPathState(move_path)).add(q_move_vbox) )

                    move_path.mx_opened.reflect(lambda opened, enter, bag=mx.Disposable().dispose_with(bag):
                                                self._ref_move_path(opened, enter, q_move_vbox, path_id, bag)).dispose_with(bag)

                self._filtered_fsip = None
                self._update_item_view_size()
                self._update_filtered_fsip()

                if (marked_items := mode.marked_items) is not None:
                    q_item_view.apply_model( q_item_view.model.mark(marked_items) )

                if (selected_items := mode.selected_items) is not None:
                    model = q_item_view.model.select(selected_items)
                    if selected_items.count != 0:
                        model = model.ensure_visible_item(selected_items.max)
                    q_item_view.apply_model(model)

                if q_item_view.model.selected_items.count == 0 and \
                   q_item_view.model.item_count != 0:
                    q_item_view.apply_model( q_item_view.model.select(0) )


            elif isinstance(mode, TransformMode):
                def transform_mode_exit(save : bool):
                    if save:
                        try:
                            # Warp image and pairs according model result_uni_mat
                            uni_mat = q_te.model.result_uni_mat

                            itempath = fsip.get_item_path(mode.item_id)
                            meta = fd.FEmbedAlignedFaceInfo.from_embed(itempath)

                            image = FImage.from_file(itempath)
                            image = image.warp_affine(mat := uni_mat.scale_space(image.size),
                                                      image.size, interp=FImage.Interp.LANCZOS4)
                            image.save(itempath)

                            if meta is not None:
                                # Restore meta if exists
                                aligned_face = meta.aligned_face.transform(mat)
                                meta = meta.set_aligned_face(aligned_face)
                                meta.embed_to(itempath)

                            for pair_type in fsip.pair_types:
                                if (pair_path := fsip.get_pair_path(mode.item_id, pair_type)) is not None:
                                    pair = FImage.from_file(pair_path)
                                    pair = pair.warp_affine(uni_mat.scale_space(pair.size), pair.size)
                                    pair.save(pair_path)

                        except Exception as e:
                            self._mx_info.emit(f'@(Error): {itempath} {e}')

                    self._mx_mode.set( DatasetMode(marked_items=mode.marked_items,
                                                   selected_items=mode.selected_items,
                                                   ) )


                fsip = mode.fsip
                err = None
                try:
                    image = FImage.from_file(fsip.get_item_path(mode.item_id))
                except Exception as e:
                    err = e

                if err is None:
                    q_te = QxTransformEditor(image).dispose_with(bag)
                    q_te.mx_quit_ev.listen(lambda save: transform_mode_exit(save))

                    self._q_central_panel_vbox.add(q_te)

                else:
                    self._main_thread.call_soon(lambda: (transform_mode_exit(False),
                                                         self._mx_info.emit(f'@(Error): {err}')
                                                         )).dispose_with(bag)


            elif isinstance(mode, MaskEditorMode):
                def mask_editor_exit():
                    self._mx_mode.set( DatasetMode(marked_items=mode.marked_items,
                                                   selected_items=FIndices([me.current_item_id])) )
                me = QxMaskEditor(mode.fsip, mode.item_id, pair_type=mode.pair_type).dispose_with(bag)
                me.mx_quit_ev.listen(mask_editor_exit)

                self._q_central_panel_vbox.add(me)

        else:
            bag.dispose_items()

    def _ref_move_path(self, opened : bool, enter : bool, q_trash_vbox : qx.QVBox, path_id : int, bag : mx.Disposable):
        if enter:
            if opened:
                shortcut = qx.QShortcut(qt.QKeyCombination(  qt.Qt.Key(qt.Qt.Key.Key_1.value+ path_id) )).set_parent(q_trash_vbox).dispose_with(bag) \
                                            .inline(lambda shortcut: shortcut.mx_press.listen(lambda: self._on_shortcut_move_selected(path_id)))
                q_trash_vbox.add(qx.QPushButton().dispose_with(bag).set_text(f"{qx.hfmt.colored_shortcut_keycomb(shortcut)}")
                                            .inline(lambda btn: (btn.mx_pressed.listen( lambda: shortcut.press()), btn.mx_released.listen(lambda: shortcut.release()))))
        else:
            bag.dispose_items()

    def _update_filtered_fsip(self):
        fsip = self._de.fsip

        pair_type = self._mx_pair_type.get()

        filter_by_pair_type = self._mx_filter_by_pair_type.get()
        filter_name    = self._mx_filter_name.get()

        filtered_ids = []
        for item_id in range(fsip.item_count):
            b = True

            if filter_by_pair_type and pair_type != MxPairType.NO_PAIR:
                b = fsip.has_pair(item_id, pair_type)

            if b and filter_name != '*':
                b = fnmatch(fsip.get_item_path(item_id).name, filter_name)

            if b:
                filtered_ids.append(item_id)

        q_item_view = self._q_item_view
        model = q_item_view.model

        filtered_fsip = fsip.filtered_view(filtered_ids)

        old_filtered_fsip, self._filtered_fsip = self._filtered_fsip, filtered_fsip

        if old_filtered_fsip is not None:
            orig_selected_items = old_filtered_fsip.to_orig_indices(model.selected_items)
            orig_marked_items = old_filtered_fsip.to_orig_indices(model.marked_items)

        model = model.set_item_count(filtered_fsip.item_count).unselect_all().unmark_all()

        if old_filtered_fsip is not None:
            model = model.select( filtered_fsip.from_orig_indices(orig_selected_items) )
            model = model.mark( filtered_fsip.from_orig_indices(orig_marked_items) )

        q_item_view.apply_model(model)
        q_item_view.update_items()



    def _update_item_view_size(self):
        W = H = self._mx_thumbnail_size.get()

        show_caption = self._mx_show_file_name.get() or self._mx_show_file_ext.get()
        if show_caption:
            H += 16

        q_item_view = self._q_item_view
        q_item_view.apply_model( q_item_view.model.set_item_size(W, H) )
        q_item_view.update_items()

    def _on_start_sort_btn_clicked(self):
        method = self._mx_sort_by_method.get()

        if method == self._de.SortByMethod.Face_similarity_reference:
            q_item_view = self._q_item_view
            orig_indices = self._filtered_fsip.to_orig_indices(q_item_view.model.selected_items)
            reference_item_id = orig_indices.min if orig_indices.count != 0 else None
            self._de.process_sort_by_face_similarity(reference_item_id)
        elif method == self._de.SortByMethod.Face_similarity_clustering:
            self._de.process_sort_by_face_similarity()
        else:
            self._de.process_sort(method)

    def _on_shortcut_move_selected(self, dir_id : int):
        q_item_view = self._q_item_view

        # Do trash original indices
        result = self._de.move_items( self._filtered_fsip.to_orig_indices(q_item_view.model.selected_items), dir_id )

        # Back to filtered indices
        moved_ids = self._filtered_fsip.from_orig_indices(result.moved_ids)


        selected_items = q_item_view.model.selected_items
        marked_items = q_item_view.model.marked_items

        self._update_filtered_fsip()

        # Recover selected & marked items after trash
        model = q_item_view.model

        selected_items = selected_items.discard(moved_ids, shift=True)
        marked_items = marked_items.discard(moved_ids, shift=True)

        if len(selected_items) == 0 and len(moved_ids) != 0:
            selected_items = selected_items | min(moved_ids.max - moved_ids.count + 1, model.item_count-1)

        model = model.unselect_all().unmark_all().select(selected_items).mark(marked_items)

        if selected_items.count != 0:
            model = model.ensure_visible_item(selected_items.max)

        q_item_view.apply_model( model )


    @ax.task
    def _on_get_item_pixmap(self, item_id : int, size : qt.QSize):
        yield ax.switch_to(self._thread_pool)

        pair_type = self._mx_pair_type.get()
        view_mode = self._mx_view_mode.get()

        filtered_fsip = self._filtered_fsip
        show_file_name = self._mx_show_file_name.get()
        show_file_ext = self._mx_show_file_ext.get()
        show_caption = show_file_name or show_file_ext
        draw_annotations = self._mx_draw_annotations.get()
        draw_no_pair = False

        err = None
        image = None

        if pair_type == MxPairType.NO_PAIR and \
           view_mode == self.ViewMode.PairedImage:
            view_mode = self.ViewMode.Image
            draw_no_pair = True

        item_path = filtered_fsip.get_item_path(item_id)

        if view_mode == self.ViewMode.PairedImage:
            if (pair_path := filtered_fsip.get_pair_path(item_id, pair_type)) is not None:
                try:
                    image = FImage.from_file(pair_path)
                except Exception as e:
                    err = e
            else:
                draw_no_pair = True
                view_mode = self.ViewMode.Image

        if view_mode == self.ViewMode.Image:
            try:
                image = FImage.from_file(item_path)
            except Exception as e:
                err = e

        cap_height = 16

        w = size.width()
        h = size.height()

        pixmap = qt.QPixmap(size)
        pixmap.fill( qt.QColor(0,0,0,0) )

        qp = qt.QPainter(pixmap)

        if show_caption:
            caption = ''
            if show_file_name:
                caption += f'{item_path.stem}'
            if show_file_ext:
                caption += f'{item_path.suffix}'


            image_rect = qt.QRect(0,0, w, h-cap_height)
            cap_rect = qt.QRect(0, h-cap_height, w, cap_height)

            caption_bg_color = qx.StyleColor.Midlight
            qp.fillRect(cap_rect, caption_bg_color)

            font = qx.QFontDB.instance().fixed_width()
            fm = qt.QFontMetrics(font)
            qp.setFont(font)
            qp.setPen(qx.StyleColor.Text)
            caption_text = fm.elidedText(caption, qt.Qt.TextElideMode.ElideLeft, cap_rect.width())
            qp.drawText(cap_rect, qt.Qt.AlignmentFlag.AlignCenter, caption_text)

        else:
            image_rect = qt.QRect(0,0, w, h)

        qp.fillRect(image_rect, qx.StyleColor.Midlight)

        if err is None:
            W, H = image.width, image.height
            fitted_image_rect = qt.QRect_fit_in(qt.QRect(0,0, W, H), image_rect)

            TW = fitted_image_rect.width()
            TH = fitted_image_rect.height()

            image = image.resize(TW, TH).u8().bgr()

            if draw_annotations:
                try:
                    meta = fd.FEmbedAlignedFaceInfo.from_embed(item_path)
                except Exception as e:
                    meta = None

                if meta is not None:
                    aligned_face = meta.aligned_face.resize(FVec2i(TW, TH))

                    lmrks = aligned_face.annotations.get_first_by_class_prio([fd.FAnnoLmrk2D106, fd.FAnnoLmrk2D68, fd.FAnnoLmrk2D])

                    if isinstance(lmrks, fd.FAnnoLmrk2D):
                        image = lmrks.draw(image, [0,255,0])

            qp.drawPixmap(fitted_image_rect, qt.QPixmap_from_FImage(image))

            if draw_no_pair:

                qp.fillRect(image_rect, qt.QColor(0,0,0,170))
                qp.setPen(qt.QColor(255,255,255,255))
                qp.drawText(image_rect, qt.Qt.AlignmentFlag.AlignCenter, lx.L(f'- @(no_pair) -', self._lang))

        else:
            qp.fillRect(image_rect, qt.QColor(0,0,0,170))
            qp.drawText(image_rect, qt.Qt.AlignmentFlag.AlignCenter, lx.L(f'- @(error) -', self._lang))

        qp.end()

        return pixmap



@dataclass
class DatasetMode:
    marked_items : FIndices|None = None
    selected_items : FIndices|None = None

@dataclass
class TransformMode:
    fsip : IFSIP_v
    item_id : int
    marked_items : FIndices|None = None
    selected_items : FIndices|None = None

@dataclass
class MaskEditorMode:
    fsip : IFSIP_v
    item_id : int
    pair_type : str|None = None
    marked_items : FIndices|None = None
    selected_items : FIndices|None = None

