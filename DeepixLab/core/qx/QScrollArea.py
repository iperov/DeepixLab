from .. import qt
from ._helpers import q_init
from .QTimer import QTimer
from .QWidget import QWidget


class QVScrollArea(QWidget):
    def __init__(self, min_width_from_widget=True, **kwargs):
        super().__init__(q_widget=q_init('q_scroll_area', qt.QScrollArea, **kwargs),  **kwargs)

        q_scroll_area = self.q_scroll_area

        self._min_width_from_widget = min_width_from_widget
        self._widget_min_size = qt.QSize(0,0)

        q_scroll_area.setWidgetResizable(True)
        q_scroll_area.setHorizontalScrollBarPolicy(qt.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        q_scroll_area.setVerticalScrollBarPolicy(qt.Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        QTimer(on_timeout=self._on_timer).set_interval(200).start().dispose_with(self)
    
    @property
    def q_scroll_area(self) -> qt.QScrollArea: return self.q_widget

    def set_widget(self, widget : QWidget):
        widget.set_parent(self)
        self.q_scroll_area.setWidget(widget.q_widget)
        return self

    def _on_timer(self, *_):
        if (widget := self.q_scroll_area.widget()) is not None:
            if self._widget_min_size != widget.minimumSizeHint():
                self.update_geometry()

    def _minimum_size_hint(self) -> qt.QSize:
        min_size = super()._minimum_size_hint()

        if self._min_width_from_widget:
            if (widget := self.q_scroll_area.widget()) is not None:

                widget_min_size = self._widget_min_size = widget.minimumSizeHint()
                min_size.setWidth(widget_min_size.width() + self.q_style.pixelMetric(qt.QStyle.PixelMetric.PM_ScrollBarExtent)+2 )
                
        return min_size

    def _size_hint(self) -> qt.QSize:
        return self._minimum_size_hint()


