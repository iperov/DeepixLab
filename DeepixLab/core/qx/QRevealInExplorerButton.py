from pathlib import Path
from typing import Callable

from .. import qt
from .QIconDB import IconDB
from .QPushButton import QPushButton

class QRevealInExplorerButton(QPushButton):
    """generic reveal in explorer icon button"""
    def __init__(self, path_evaluator : Callable[[], Path|None]):
        """
            path_evaluator      when clicked, evaluates  Path
                                or None if do nothing
        """
        super().__init__()
        self.h_compact().set_tooltip('@(Reveal_in_explorer)').set_icon(IconDB.eye_outline)
        self.mx_clicked.listen(lambda: qt.QDesktopServices.openUrl(qt.QUrl.fromLocalFile(str(path))) if (path := path_evaluator()) is not None else ...)
