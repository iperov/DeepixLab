from .. import lx, qt
from ._constants import Align
from .QApplication import QApplication
from .QBox import QHBox, QVBox
from .QFontDB import FontDB
from .QFrame import QVFrame
from .QIconDB import IconDB
from .QLabel import QLabel
from .QPushButton import QPushButton
from .QTextEdit import QTextEdit
from .QTimer import QTimer


class QInfoBar(QHBox):
    """
    hidden by default.
    """
    def __init__(self):
        super().__init__()
        self._title = None
        
        self._upd_timer = QTimer(self._on_timer).set_parent(self).set_interval(100).start()
        
        self._te = QTextEdit().set_font(FontDB.FixedWidth).set_read_only(True)

        self._title_label = QLabel()
        
        self._text_strings = []
        self._pending_text_strings = []

        (self.hide().set_spacing(1)
            .add(QVBox()
                    .add(QVFrame().add(self._title_label.hide(), align=Align.CenterH).v_compact())
                    .add(self._te)
                    )
            .add(QPushButton().h_compact()
                    .set_icon(IconDB.checkmark_done, qt.QColor(100,200,0))
                    .inline(lambda btn: btn.mx_clicked.listen(self._on_btn_done)))) #lambda: (self._te.clear(), self.hide())))))
    
    def _on_timer(self, timer : QTimer):
        if len(self._pending_text_strings) != 0:
            self._text_strings.extend(self._pending_text_strings)
            self._pending_text_strings = []
            
            self.show()
            q_text_edit = self._te.q_text_edit
            scroll_bar = q_text_edit.verticalScrollBar()
            scroll_value = scroll_bar.value()
            scroll_at_max = scroll_value == scroll_bar.maximum()
            self._update_text_edit()
            scroll_bar.setValue(scroll_bar.maximum() if scroll_at_max else scroll_value)
        
    def _on_btn_done(self):
        self._text_strings = []
        self._update_text_edit()
        self.hide()
    
    def _update_text_edit(self):
        text ='<br>'.join(self._text_strings)
        
        self._te.set_html(f'<html><body>{text}</body></html>')
    
    def set_title(self, title : str|None):
        self._title_label.show().set_text(title).set_visible(title is not None)
        return self

    def add_text(self, text : str):
        """add text and show the bar"""
        self._pending_text_strings.append(lx.L(text, QApplication.instance().mx_language.get()))
                
        
        # te = self._te.q_text_edit

        # cursor = te.textCursor()
        # cursor.clearSelection()

        # cursor.movePosition(qt.QTextCursor.MoveOperation.End)

        # if cursor.position() != 0:
        #     cursor.insertText('\n')

        # cursor.insertText(text)

        # te.verticalScrollBar().setValue(te.verticalScrollBar().maximum())

        # self.show()