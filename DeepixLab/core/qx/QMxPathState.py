from __future__ import annotations

from enum import Enum, auto
from pathlib import Path

from .. import mx
from .QBox import QHBox
from .QFileDialog import QFileDialog
from .QFontDB import FontDB
from .QIconDB import IconDB
from .QLineEdit import QLineEdit
from .QPushButton import QPushButton
from .QRevealInExplorerButton import QRevealInExplorerButton


class QMxPathState(QHBox):

    def __init__(self, path : mx.IPath_v):
        """ViewController widget composition of model mx.Path"""
        super().__init__()
        self._path = path

        self._dlg_mode = None

        config = path.config


        line_edit = self._line_edit = QLineEdit()
        line_edit.set_font(FontDB.FixedWidth).set_read_only(True)
        if (desc := config.desc) is not None:
            line_edit.set_placeholder_text(desc)

        (self
            .add(QPushButton()  .h_compact().set_icon(IconDB.close_outline).set_tooltip('@(Close)')
                                .inline( lambda btn: btn.mx_clicked.listen(lambda: path.close()).dispose_with(self) ) )

            .add(QPushButton()  .h_compact().set_icon(IconDB.open_outline).set_tooltip('@(Open)')
                                .inline(lambda btn: btn.mx_clicked.listen(lambda: self._open_dlg(QMxPathState._DlgMode.Open)).dispose_with(self) )
                                if config.allow_open else None )

            .add(QPushButton()  .h_compact().set_icon(IconDB.add_circle_outline).set_tooltip('@(New)')
                                .inline(lambda btn: btn.mx_clicked.listen(lambda: self._open_dlg(QMxPathState._DlgMode.New)).dispose_with(self) )
                                if config.allow_new else None )

            .add( rename_btn := (QPushButton().hide().h_compact().set_icon(IconDB.pencil_outline).set_tooltip('@(Change)')
                                .inline(lambda btn: btn.mx_clicked.listen(lambda: self._open_dlg(QMxPathState._DlgMode.Rename)).dispose_with(self) )
                                ) if config.allow_rename else None )

            .add( reveal_btn := QRevealInExplorerButton(lambda: (p if p.is_dir() else p.parent) if (p := path.get()) is not None else None ) )

            .add(self._line_edit.h_expand()))

        self._rename_btn = rename_btn
        self._reveal_btn = reveal_btn

        path.mx_opened.reflect(lambda opened, enter, bag=mx.Disposable().dispose_with(self): 
                                self._ref_opened(opened, enter, bag)).dispose_with(self)

        self._show_bag = mx.Disposable().dispose_with(self)

    def _ref_opened(self, opened : bool, enter : bool, bag : mx.Disposable):
        
        if enter:
            if opened:
                if self._rename_btn is not None:
                    self._rename_btn.show()
                self._reveal_btn.show()
                self._path.reflect(lambda path: self._line_edit.set_text(str(path))).dispose_with(bag)
                
            else:
                if self._rename_btn is not None:
                    self._rename_btn.hide()
                self._reveal_btn.hide()
        else:
            bag.dispose_items()
            self._line_edit.set_text(None)

    def _open_dlg(self, mode : QMxPathState._DlgMode):
        bag = self._show_bag.dispose_items()

        file_dlg = QFileDialog().set_parent(self).dispose_with(bag)
        file_dlg.mx_accepted.listen(lambda: self._on_file_dlg_mx_accepted(file_dlg))

        if (path := self._path.get()) is not None:
            file_dlg.set_directory( path if path.is_dir() else path.parent )

        config = self._path.config
        if config.dir:
            file_dlg.set_file_mode( QFileDialog.FileMode.Directory)
            file_dlg.set_option( QFileDialog.Option.ShowDirsOnly, True)
        else:
            if mode in [QMxPathState._DlgMode.New, QMxPathState._DlgMode.Rename]:
                file_dlg.set_file_mode( QFileDialog.FileMode.AnyFile)
            else:
                file_dlg.set_file_mode( QFileDialog.FileMode.ExistingFile)

            if (extensions := config.extensions) is not None:

                if (desc := config.desc) is None:
                    desc = '@(Acceptable_files)'

                file_dlg.set_filter(f"{desc} ({' '.join([f'*{ext}' for ext in extensions])})")

        if mode in [QMxPathState._DlgMode.New, QMxPathState._DlgMode.Rename] and not config.dir:
            file_dlg.set_accept_mode(QFileDialog.AcceptMode.AcceptSave)
        else:
            file_dlg.set_accept_mode(QFileDialog.AcceptMode.AcceptOpen)

        self._dlg_mode = mode
        file_dlg.open()

    def _on_file_dlg_mx_accepted(self, file_dlg : QFileDialog):
        path = file_dlg.selected_files[0]

        if self._dlg_mode == QMxPathState._DlgMode.Open:
            self._path.open(path)
        elif self._dlg_mode == QMxPathState._DlgMode.New:
            self._path.new(path)
        elif self._dlg_mode == QMxPathState._DlgMode.Rename:
            self._path.rename(path)

    class _DlgMode(Enum):
        Open = auto()
        New = auto()
        Rename = auto()