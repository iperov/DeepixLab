from .. import mx
from .QLineEdit import QLineEdit


class QLineEditMxText(QLineEdit):
    def __init__(self, text : mx.IText_v, on_editing_finished=False, **kwargs):
        """
            on_editing_finished(False)   
        """
        super().__init__(**kwargs)
        self._text = text

        if on_editing_finished:
            self._conn = self.mx_editing_finished.listen(lambda: text.set(self.mx_text.get()) )
        else:
            self._conn = self.mx_text.listen(lambda s: text.set(s))
        
        text.reflect(self._ref_text).dispose_with(self)

    def _ref_text(self, text):
        with self._conn.disabled_scope():
            self.set_text(text)
