from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .. import lx, mx, qt
from ._helpers import q_init
from .QApplication import QApplication
from .QEvent import QEvent0
from .QObject import QObject
from .QSettings import QSettings


class QFileDialog(QObject):
    Option = qt.QFileDialog.Option
    AcceptMode = qt.QFileDialog.AcceptMode
    FileMode = qt.QFileDialog.FileMode

    def __init__(self, **kwargs):
        super().__init__(q_object=q_init('q_file_dialog', qt.QFileDialog, **kwargs), **kwargs)

        q_file_dialog = self.q_file_dialog
        self.__mx_accepted = QEvent0(q_file_dialog.accepted).dispose_with(self)

        self._mx_settings.reflect(lambda setting, enter, bag=mx.Disposable().dispose_with(self): self.__ref_settings(setting, enter, bag))

    @property
    def q_file_dialog(self) -> qt.QFileDialog: return self.q_object

    @property
    def mx_accepted(self) -> mx.IEvent0_rv: return self.__mx_accepted

    @property
    def selected_files(self) -> Sequence[Path]: return tuple(Path(x) for x in self.q_file_dialog.selectedFiles())
    @property
    def directory(self) -> Path: return Path(self.q_file_dialog.directory().path())

    def __ref_settings(self, settings : QSettings, enter : bool, bag : mx.Disposable):
        if enter:
            if (dir := settings.state.get('dir', None)) is not None:
                self.set_directory(dir)

            self.__mx_accepted.listen(lambda: settings.state.set('dir', self.directory)).dispose_with(bag)
        else:
            bag.dispose_items()


    def set_directory(self, dir : Path|None):
        if dir is not None:
            self.q_file_dialog.setDirectory(str(dir))
        return self

    def set_filter(self, filter : str):
        if (disp := getattr(self, '_QLineEdit_text_disp', None)) is not None:
            disp.dispose()
        self._QLineEdit_text_disp = QApplication.instance().mx_language.reflect(lambda lang: self.q_file_dialog.setNameFilter(lx.L(filter, lang))).dispose_with(self)
        return self

    def set_option(self, option : Option, on : bool = None):
        self.q_file_dialog.setOption(option, on)
        return self

    def set_accept_mode(self, accept_mode : AcceptMode):
        self.q_file_dialog.setAcceptMode(accept_mode)
        return self

    def set_file_mode(self, file_mode : FileMode):
        self.q_file_dialog.setFileMode(file_mode)
        return self

    def reject(self):
        self.q_file_dialog.reject()
        return self

    def open(self):
        dir = self.directory
        while True:
            if dir.exists():
                self.set_directory(dir)
                break

            if dir == dir.parent:
                self.set_directory(None)
                break

            dir = dir.parent

        self.q_file_dialog.open()
        return self
