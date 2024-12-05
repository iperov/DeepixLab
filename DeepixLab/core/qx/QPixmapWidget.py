from .. import qt
from .QWidget import QWidget

class QPixmapScaled:
    """holds cached scaled version of pixmap"""
    def __init__(self):
        ...
        
    @property
    def pixmap(self) -> qt.QPixmap:
        """original pixmap"""
        return self._pixmap
    
    def scaled(self, ):
        ...
        
        #qt.QImage().scaled()
    
class QPixmapWidget(QWidget):
    """draws QPixmap centered in provided area"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._qp = qt.QPainter()
        self._pixmap : qt.QPixmap = None

    def set_pixmap(self, pixmap : qt.QPixmap|None):
        self._pixmap = pixmap
        self._update_fitted_rect()
        self.update()
        return self

    def _update_fitted_rect(self):
        rect = self.rect
        pixmap = self._pixmap
        if pixmap is not None:
            self._fitted_rect = qt.QRect_fit_in(pixmap.rect(), rect)
        else:
            self._fitted_rect = rect

    def _resize_event(self, ev: qt.QResizeEvent):
        super()._resize_event(ev)
        self._update_fitted_rect()

    def _paint_event(self, ev: qt.QPaintEvent):
        if (pixmap := self._pixmap) is not None:

            qp = self._qp
            qp.begin(self.q_widget)
            qp.setRenderHint( qt.QPainter.RenderHint.SmoothPixmapTransform)

            qp.drawPixmap(self._fitted_rect, pixmap, pixmap.rect())
            qp.end()
