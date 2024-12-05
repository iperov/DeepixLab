from typing import Tuple, overload

from .. import qt
from ._constants import Size, icon_Size_to_int
from ._helpers import q_init
from .QIconDB import IconDB, QIconDB
from .QWidget import QWidget
from .StyleColor import StyleColor


class _QIconWidget(qt.QWidget):
    """draws QIcon centered in provided area"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._qp = qt.QPainter()
        self._icon : qt.QIcon = None

        icon_size = self.style().pixelMetric(qt.QStyle.PixelMetric.PM_ButtonIconSize)
        self._icon_size = qt.QSize(icon_size, icon_size)

    def get_icon(self) -> qt.QIcon|None:
        return self._icon

    def set_icon(self, icon : qt.QIcon):
        self._icon = icon
        self._update()
        self.update()
        return self

    def set_icon_size(self, size : qt.QSize):
        self._icon_size = size
        self._update()
        self.updateGeometry()
        self.update()
        return self

    def _update(self):
        rect = self.rect()
        if (icon := self._icon) is not None:
            size = icon.actualSize(self._icon_size)
            self._pixmap_rect = qt.QRect_center_in(qt.QRect(0,0, size.width(), size.height()), rect)
        else:
            self._pixmap_rect = rect

    def minimumSizeHint(self) -> qt.QSize:
        return self._icon_size.grownBy(qt.QMargins(0,0,8,8))
    
    def sizeHint(self) -> qt.QSize:
        return self._icon_size.grownBy(qt.QMargins(0,0,8,8))
    
    def resizeEvent(self, ev: qt.QResizeEvent):
        super().resizeEvent(ev)
        self._update()

    def paintEvent(self, ev: qt.QPaintEvent):
        # qp = self._qp
        # qp.begin(self)
        # qp.fillRect(self.rect(), qt.QColor(255,255,255))
        # qp.end()


        if (icon := self._icon) is not None:
            qp = self._qp
            qp.begin(self)
            pixmap = icon.pixmap(self._icon_size)
            qp.drawPixmap(self._pixmap_rect, pixmap, pixmap.rect())
            qp.end()

class QIconWidget(QWidget):
    """draws QIcon centered in provided area"""
    def __init__(self, **kwargs):
        super().__init__(q_widget=q_init('q_icon_widget', _QIconWidget, **kwargs), **kwargs)

    @property
    def q_icon_widget(self) -> _QIconWidget: return self.q_widget
    
    @property
    def icon(self) -> qt.QIcon: return self.q_icon_widget.get_icon()
    
    @overload
    def set_icon(self, icon : qt.QIcon): ...
    @overload
    def set_icon(self, icon : IconDB, color : qt.QColor = StyleColor.Text): ...
    def set_icon(self, *args, **kwargs):
        if len(args) == 1:
            arg0 = args[0]
            if isinstance(arg0, qt.QIcon):
                self.q_icon_widget.set_icon(arg0)
            else:
                args = (arg0, kwargs.get('color', StyleColor.Text))

        if len(args) == 2:
            self.set_icon(QIconDB.instance().icon(args[0], args[1]))

        return self

    def set_icon_size(self, size : Tuple[int, int] | Size):
        if isinstance(size, Size):
            size = (icon_Size_to_int[size],)*2

        self.q_icon_widget.set_icon_size(qt.QSize(*size))
        return self
