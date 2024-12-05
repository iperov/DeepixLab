from __future__ import annotations

from .. import mx
from ..lib import os as lib_os
from ..lx import allowed_langs
from ._constants import ProcessPriority
from .QAction import QAction
from .QApplication import QApplication
from .QAxMonitorWindow import QAxMonitorWindow
from .QMenu import QMenu
from .QSettings import QSettings
from .QTextBrowser import QTextBrowser
from .QMainWIndow import QMainWIndow
from .QWindow import QWindow


class QAppWindow(QMainWIndow): 
    def __init__(self):
        """
        Main application window.

        Provides base functionality and menus. Allowed only single instance. 
        """
        if QAppWindow._instance is not None:
            raise Exception('Only one QAppWindow can exist.')
        QAppWindow._instance = self
        super().__init__()

        app = QApplication.instance()
        
        (self.q_menu_bar
            .add(QMenu().set_title('@(Application)')
                    .add(QMenu().set_title('@(Process_priority)')
                         .inline(lambda menu: menu.mx_about_to_show.listen(lambda me=menu:
                                    me.dispose_actions()
                                        .add( QAction()
                                                .set_text(f"{'-> ' if app.mx_process_priority.get() == ProcessPriority.NORMAL else ''}@(Process_priority.Normal)")
                                                .inline(lambda act: act.mx_triggered.listen(lambda: app.mx_process_priority.set(ProcessPriority.NORMAL))))

                                        .add( QAction()
                                                .set_text(f"{'-> ' if app.mx_process_priority.get() == ProcessPriority.BELOW_NORMAL else ''}@(Process_priority.Low)")
                                                .inline(lambda act: act.mx_triggered.listen(lambda: app.mx_process_priority.set(ProcessPriority.BELOW_NORMAL)))))))

                    .add(QAction().set_text('@(Reset_UI)').inline(lambda act: act.mx_triggered.listen(lambda: app.reset_state())))
                    .add(QAction().set_text('@(AsyncX_monitor)').inline(lambda act: act.mx_triggered.listen(lambda: self._on_open_ax_monitor())))
                    .add(QAction().set_text('@(Show_hide_console)').inline(lambda act: act.mx_triggered.listen(lambda: QApplication.instance().switch_console_window() )))
                    .add(QAction().set_text('@(Quit)').inline(lambda act: act.mx_triggered.listen(lambda: app.quit()))))

            .add(QMenu().set_title('@(Language)')
                    .inline(lambda menu:
                                [ menu.add(QAction().set_text(name).inline(lambda act: act.mx_triggered.listen(lambda lang=lang: app.mx_language.set(lang))))
                                    for lang, name in allowed_langs.items() ]))

            .add(QMenu().set_title('@(Help)')
                .add(QAction().set_text('@(About)').inline(lambda act: act.mx_triggered.listen(lambda: self._on_open_about())))))
        
        self.set_parent(app)
        
        self.mx_close.listen(lambda _: app.quit())
    #     self._mx_settings.reflect(lambda settings, enter, bag=mx.Disposable().dispose_with(self): self.__ref_settings(settings, enter, bag))
     
    # def __ref_settings(self, settings : QSettings, enter : bool, bag : mx.Disposable):
    #     if enter:
    #         self.__settings = settings
    #     else:
    #         bag.dispose_items()

    def __dispose__(self):
        QAppWindow._instance = None
        super().__dispose__()

    def _on_open_ax_monitor(self):
        if (wnd := getattr(self, '_ax_monitor_wnd', None)) is not None:
            wnd.activate()
        else:
            wnd = self._ax_monitor_wnd = QAxMonitorWindow().dispose_with(self)
            wnd.set_window_icon(self.q_window_icon)
            mx.CallOnDispose(lambda: setattr(self, '_ax_monitor_wnd', None)).dispose_with(wnd)
            wnd.mx_close.listen(lambda ev: wnd.dispose())
            wnd.show()
            

    def _on_open_about(self):
        if (wnd := getattr(self, '_about_wnd', None)) is not None:
            wnd.activate()
        else:
            wnd = self._about_wnd = QAboutWindow().dispose_with(self)
            wnd.set_parent(self)
            wnd.set_window_icon(self.q_window_icon)
            mx.CallOnDispose(lambda: setattr(self, '_about_wnd', None)).dispose_with(wnd)
            wnd.mx_close.listen(lambda ev: wnd.dispose())
            wnd.show()

    _instance : QAppWindow = None


class QAboutWindow(QWindow):
    def __init__(self):
        super().__init__()

        app = QApplication.instance()

        self.set_window_size(320,200).set_title('@(About)')

        te = QTextBrowser().set_open_external_links(True)
        te.set_html("""
<html><body>

<table width="100%" height="100%">
<tr>
<td valign='middle' align='center'>

<span style='font-size:14.0pt'>DeepixLab</span>
<br>
<span style='font-size:10.0pt'><a href="https://iperov.github.io/DeepixLab">https://iperov.github.io/DeepixLab</a></span>
<br><br>
<span style='font-size:8.0pt'>Free open source software under GPL-3 license.
<br>
Designed and developed from scratch by <a href="https://github.com/iperov">iperov</a><br>
</span>
<br>
<br>

</td>
</tr>
</table>

</body></html>
""")
        self.add(te)