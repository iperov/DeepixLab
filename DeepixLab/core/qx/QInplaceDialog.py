from .QBox import QHBox, QVBox
from .QLabel import QLabel
from .QPushButton import QPushButton


class QInplaceDialog(QVBox):
    """
    a set of inplace preconfigured widgets: title, ok cancel buttons.

    hidden by default
    """

    def __init__(self):
        super().__init__()

        self._q_label_title = QLabel().set_text('- no title set -')
        self._q_btn_ok = QPushButton()
        self._q_btn_ok.mx_clicked.listen(lambda: self.hide())

        self._q_btn_cancel = QPushButton()
        self._q_btn_cancel.mx_clicked.listen(lambda: self.hide())

        (self   .add(self._q_label_title)

                .add(QHBox()
                        .add(self._q_btn_ok.set_text('@(Ok)'))
                        .add(self._q_btn_cancel.set_text('@(Cancel)'))
                    ))
        self.hide()

    @property
    def q_label_title(self) -> QLabel: return self._q_label_title
    @property
    def q_btn_ok(self) -> QPushButton: return self._q_btn_ok
    @property
    def q_btn_cancel(self) -> QPushButton: return self._q_btn_cancel
    
    def popup(self):
        """.show()"""
        self.show()