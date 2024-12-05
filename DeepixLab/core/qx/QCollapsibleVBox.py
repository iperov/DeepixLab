from .. import mx, qt
from .QAbstractButton import QAbstractButton
from .QBox import QVBox
from .QSettings import QSettings
from .StyleColor import StyleColor


class QCollapsibleBarButton(QAbstractButton):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__hover = False
        self.set_checkable(True).set_checked(True)

    def _minimum_size_hint(self) -> qt.QSize:
        fm = self.q_widget.fontMetrics()
        
        text = self.text
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
        icon_size = rect.height()
        q_widget = self.q_widget
        font = self.q_font
        style = self.q_style
        text_rect = qt.QRect(icon_size, 0, rect.width()-icon_size, rect.height())

        opt = qt.QStyleOption()
        opt.initFrom(q_widget)
        opt.rect = qt.QRect(0,0, icon_size,icon_size)

        qp = qt.QPainter(q_widget)

        qp.fillRect(rect, StyleColor.Midlight if self.__hover else StyleColor.Mid)

        style.drawPrimitive(qt.QStyle.PrimitiveElement.PE_IndicatorArrowDown if self.is_checked() else qt.QStyle.PrimitiveElement.PE_IndicatorArrowRight, opt, qp)

        qp.setFont(font)
        qp.drawText(text_rect, qt.Qt.AlignmentFlag.AlignLeft | qt.Qt.AlignmentFlag.AlignVCenter,
                    self.text,
                    #fm.elidedText(self.text, qt.Qt.TextElideMode.ElideRight, text_rect.width())
                    )

        qp.end()


class QCollapsibleVBox(QVBox):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._default_opened = None
        
        content_vbox = self.__content_vbox = QVBox()

        bar_btn = self.__bar_btn = QCollapsibleBarButton()
        bar_btn.mx_toggled.listen(lambda checked: content_vbox.set_visible(checked))
        
        (self.set_spacing(1)
            .add(bar_btn.v_compact())
            .add(content_vbox))

        self._mx_settings.reflect(lambda settings, enter, bag=mx.Disposable().dispose_with(self): 
                                  self.__ref_settings(settings, enter, bag))
    
    @property
    def content_vbox(self) -> QVBox: return self.__content_vbox

    def is_opened(self) -> bool: return self.__bar_btn.is_checked()
    
    def set_opened(self, opened : bool):
        if self.__bar_btn.is_checked() != opened:
            self.__bar_btn.set_checked(opened)
            
        return self
    
    def open(self): return self.set_opened(True)
    def close(self): return self.set_opened(False)

    def toggle(self):
        self.open() if not self.is_opened() else self.close()
        return self

    def set_text(self, text : str|None):
        self.__bar_btn.set_text(text)
        return self
    
    def __ref_settings(self, settings : QSettings, enter : bool, bag : mx.Disposable):
        if enter:
            if self._default_opened is None:
                self._default_opened = self.is_opened()

            self.set_opened( settings.state.get('opened', self._default_opened) )
            
            self.__bar_btn.mx_toggled.listen(lambda checked: settings.state.set('opened', checked)).dispose_with(bag)
        else:
            bag.dispose_items()
        
    