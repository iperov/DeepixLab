import re

from core import qx

from .MxMultiStateManager import MxMultiStateManager


class QxMultiStateManager(qx.QVBox):

    def __init__(self, mgr : MxMultiStateManager):
        super().__init__()
        self._mgr = mgr

        menu_save = qx.QMenu().set_parent(self)
        menu_save.mx_about_to_show.listen(lambda:
            (   menu_save.dispose_actions(),
                [   menu_save.add( qx.QAction().set_text(state_name).inline(lambda act, state_name=state_name:
                                     act.mx_triggered.listen(lambda: mgr.save(state_name) )))
                    for state_name in mgr.mx_state.mx_avail.get() ],
                menu_save.add( qx.QAction().set_text('@(New)...').inline(lambda act:
                                act.mx_triggered.listen( lambda: save_new_dlg.popup()))) # (save_btn.hide(), holder_save_as_new.show()))) )
                ))

        menu_delete = qx.QMenu().set_parent(self)
        menu_delete.mx_about_to_show.listen(lambda:
            (   menu_delete.dispose_actions(),
                [   menu_delete.add( qx.QAction().set_text(state_name).inline(lambda act, state_name=state_name: act.mx_triggered.listen(lambda: mgr.remove_state(state_name) )))
                    for state_name in mgr.mx_state.mx_avail.get() ], ))

        save_new_dlg = qx.QInplaceInputDialog()
        save_new_dlg.q_label_title.hide()
        save_new_dlg.q_lineedit.set_filter(lambda s: re.sub(r'\W', '', s)).set_placeholder_text('@(State_name)')
        save_new_dlg.q_btn_ok.mx_clicked.listen(lambda: (   mgr.add_state(state_name),
                                                            mgr.save(state_name) )
                                                            if len(state_name := save_new_dlg.q_lineedit.get_text()) != 0 else ... )

        (self
            .add(qx.QHBox().v_compact()
                .add(qx.QLabel().set_text('@(State)'))
                .add(qx.QComboBoxMxStateChoice(mgr.mx_state))
                .add(qx.QPushButton().set_tooltip('@(Save)').set_icon(qx.IconDB.save_outline).inline(lambda btn: btn.mx_clicked.listen(lambda: menu_save.popup() )))
                .add(qx.QPushButton().set_tooltip('@(Delete)').set_icon(qx.IconDB.trash_outline).inline(lambda btn: btn.mx_clicked.listen(lambda: menu_delete.popup() )))
                .add(save_new_dlg)

                , align=qx.Align.LeftF))

