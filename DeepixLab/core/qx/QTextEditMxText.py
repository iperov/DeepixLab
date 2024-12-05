from .QTextEdit import QTextEdit
from .. import mx



class QTextEditMxText(QTextEdit):
    def __init__(self, text : mx.IText_v, **kwargs):
        super().__init__(**kwargs)
        self._text = text
        
        q_text_edit = self.q_text_edit
        
        self._conn = self.mx_text_changed.listen(self._text_changed)
        
        text.reflect(self._ref_text).dispose_with(self)
    
    def _ref_text(self, text : str):
        if self.get_plain_text() != text:
            with self._conn.disabled_scope():
                self.set_plain_text(text)
        
    def _text_changed(self):
        self._text.set(self.get_plain_text())

    # def _size_hint(self) -> qt.QSize:
    #     size = super()._size_hint()
    #     return qt.QSize(60,60)
