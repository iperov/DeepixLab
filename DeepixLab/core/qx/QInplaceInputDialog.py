from .QBox import QHBox, QVBox
from .QLabel import QLabel
from .QLineEdit import QLineEdit
from .QPushButton import QPushButton


class QInplaceInputDialog(QVBox):
    """
    a set of inplace preconfigured widgets: title, input, ok cancel buttons.

    hidden by default
    """

    def __init__(self):
        super().__init__()

        self._q_label_title = QLabel().set_text('- no title set -')
        self._q_lineedit = QLineEdit()
        self._q_btn_ok = QPushButton()
        self._q_btn_ok.mx_clicked.listen(lambda: self.hide())

        self._q_btn_cancel = QPushButton()
        self._q_btn_cancel.mx_clicked.listen(lambda: self.hide())

        (self   .add(self._q_label_title)

                .add(QHBox()
                        .add(self._q_lineedit)
                        .add(self._q_btn_ok.h_compact().set_text('@(Ok)'))
                        .add(self._q_btn_cancel.h_compact().set_text('@(Cancel)'))
                    )
        )

        self.hide()
        
    @property
    def q_label_title(self) -> QLabel: return self._q_label_title
    @property
    def q_lineedit(self) -> QLineEdit: return self._q_lineedit
    @property
    def q_btn_ok(self) -> QPushButton: return self._q_btn_ok
    @property
    def q_btn_cancel(self) -> QPushButton: return self._q_btn_cancel

    def popup(self):
        """.show(), clear lineedit, focus on lineedit"""
        self.show()
        self._q_lineedit.set_text('').set_focus()
