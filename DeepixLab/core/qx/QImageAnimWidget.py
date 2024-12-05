from datetime import datetime
from typing import overload

from .. import qt
from ..lib.time import timeit
from .QAnimDB import AnimDB, QAnimDB
from .QImageAnim import QImageAnim
from .QTimer import QTimer
from .QWidget import QWidget
from .StyleColor import StyleColor


class QImageAnimWidget(QWidget):
    """plays QImageAnim centered in provided area"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._qp = qt.QPainter()
        self._image_anim : QImageAnim = None
        self._anim_start_time = None
        self._timer = QTimer(on_timeout=self._on_timer).set_interval(16).dispose_with(self)
        self._last_frame_id = None
    
    
    @overload
    def set_image_anim(self, anim : QImageAnim|None): ...
    @overload
    def set_image_anim(self, anim : AnimDB, color : qt.QColor = StyleColor.Text): ...
    def set_image_anim(self, *args, **kwargs):
        if len(args) == 1:
            arg0 = args[0]
            if arg0 is None or isinstance(arg0, QImageAnim):
                self._image_anim = arg0
                self._last_frame_id = None
                self.update()
            else:
                args = (arg0, kwargs.get('color', StyleColor.Text))
            
        if len(args) == 2:
            self.set_image_anim(QAnimDB.instance().get(args[0], args[1]))
            
        return self
    
    def set_playing(self, playing : bool):
        if playing:
            self.start()
        else:
            self.stop()
            
        return self
    def start(self):
        """start animation"""
        self._anim_start_time = datetime.now().timestamp()
        self._last_frame_id = None
        self._timer.start()
        
        self.update()
        return self
    
    def stop(self):
        """start animation"""
        self._anim_start_time = None
        self._timer.stop()
        self.update()
        return self
    
    def _get_current_frame_id(self):
        if (image_anim := self._image_anim) is not None:
            if (anim_start_time := self._anim_start_time) is not None:
                return int( (datetime.now().timestamp() - anim_start_time) * image_anim.fps ) % image_anim.frame_count
        return 0
    
    def _on_timer(self, *_):
        if self.visible:
            frame_id = self._get_current_frame_id()
            if self._last_frame_id != frame_id:
                self._last_frame_id = frame_id
                self.update()
            

    def _paint_event(self, ev: qt.QPaintEvent):
        if (image_anim := self._image_anim) is not None:
            rect = self.rect
            
            frame_id = self._get_current_frame_id()
            #icon = image_anim.get_icon(frame_id)
            
            
            #icon.paint
            
            qp = self._qp
            qp.begin(self.q_widget)
            qp.setRenderHint( qt.QPainter.RenderHint.SmoothPixmapTransform)
            
            pixmap = image_anim.get_pixmap(frame_id)
            
            fitted_rect = qt.QRect_fit_in(pixmap.rect(), rect)
            
            pixmap = image_anim.get_pixmap(frame_id, size=fitted_rect.size())
            
            qp.drawPixmap(fitted_rect, pixmap)#, pixmap.rect()
            qp.end()
