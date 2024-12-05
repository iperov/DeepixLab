from .. import lx, qt
from .QApplication import QApplication
from .QBox import QVBox
from .QWidget import QWidget
from .StyleColor import StyleColor


class QHeader(QWidget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__hover = False
    
    
    def set_text(self, text : str):
        if (disp := getattr(self, '_QHeader__text_disp', None)) is not None:
            disp.dispose()
        self._QHeader__text_disp = QApplication.instance().mx_language.reflect(lambda lang: (setattr(self, '_QHeader__text', lx.L(text, lang)), self.update())                                                                              
                                                                              ).dispose_with(self)
        return self
    
    def _minimum_size_hint(self) -> qt.QSize:
        fm = self.q_widget.fontMetrics()
        
        text = self.__text
        q_widget = self.q_widget
        fm = q_widget.fontMetrics()
        
        text_rect = fm.boundingRect(qt.QRect(), 0, text).adjusted(0,0,0,8)
        text_rect = text_rect.adjusted(0,0,text_rect.height(),0)
        return text_rect.size()

    def _enter_event(self, ev: qt.QEnterEvent):
        super()._enter_event(ev)
        self.__hover = True
        self.update()

    def _leave_event(self, ev: qt.QEvent):
        super()._leave_event(ev)
        self.__hover = False
        self.update()

    def _paint_event(self, ev: qt.QPaintEvent):
        rect = self.rect
        q_widget = self.q_widget
        font = self.q_font
        text_rect = qt.QRect(0, 0, rect.width(), rect.height())

        qp = qt.QPainter(q_widget)
        qp.fillRect(rect, StyleColor.Midlight if self.__hover else StyleColor.Mid)
        qp.setFont(font)
        qp.drawText(text_rect, qt.Qt.AlignmentFlag.AlignCenter, self.__text, )
        qp.end()


class QHeaderVBox(QVBox):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._default_opened = None
        
        content_vbox = self.__content_vbox = QVBox()

        header = self.__header = QHeader()
        
        (self.set_spacing(1)
            .add(header.v_compact())
            .add(content_vbox))
    
    @property
    def content_vbox(self) -> QVBox: return self.__content_vbox

    def set_text(self, text : str|None):
        self.__header.set_text(text)
        return self
    