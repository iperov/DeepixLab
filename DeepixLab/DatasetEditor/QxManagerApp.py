from pathlib import Path

from core import qt, qx

from .MxManager import MxManager
from .QxManager import QxManager


class QxManagerApp(qx.QApplication):
    def __init__(self, mgr : MxManager, state_path = None):
        super().__init__(app_name='Dataset editor app', state_path=state_path)
        app_wnd = qx.QAppWindow().set_parent(self).set_title('Dataset editor')
        app_wnd.set_window_icon(qt.QIcon(str(Path(__file__).parent / 'assets' / 'icons' / 'app_icon.png')))
        app_wnd.q_central_vbox.add(QxManager(mgr))
        app_wnd.show()

