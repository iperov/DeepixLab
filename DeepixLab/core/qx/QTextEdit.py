from .. import qt
from ._helpers import q_init
from .QWidget import QWidget
from .QEvent import QEvent0

class QTextEdit(QWidget):
    def __init__(self, **kwargs):
        super().__init__(q_widget=q_init('q_text_edit', qt.QTextEdit, **kwargs), **kwargs)
        
        q_text_edit = self.q_text_edit
        
        self._mx_text_changed = QEvent0(q_text_edit.textChanged).dispose_with(self)
        
        #q_text_edit.setAcceptRichText(True)
    @property
    def q_text_edit(self) -> qt.QTextEdit: return self.q_widget
    
    @property
    def mx_text_changed(self) -> QEvent0:
        return self._mx_text_changed
    
    def get_plain_text(self) -> str:
        return self.q_text_edit.toPlainText()
    
    def clear(self):
        self.q_text_edit.clear()
        return self

    def set_read_only(self, read_only : bool):
        self.q_text_edit.setReadOnly(read_only)
        return self
    
    def set_html(self, html : str):
        self.q_text_edit.setHtml(html)
        return self
    
    def set_plain_text(self, text : str):
        self.q_text_edit.setPlainText(text)
        return self

    # def _size_hint(self) -> qt.QSize:
    #     size = super()._size_hint()
    #     return qt.QSize(60,60)
