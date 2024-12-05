from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from ... import mx, qt
from .IconDB import IconDB
from .._constants import Size
from ...lib.image import FImage

T = TypeVar('T')

Size_to_icon_size = {   Size.XXL : 64,
                        Size.XL : 48,
                        Size.L : 32,
                        Size.M : 24,
                        Size.S : 16}


class QIconDB(mx.Disposable):

    @staticmethod
    def instance() -> QIconDB:
        if QIconDB._instance is None:
            raise Exception('No QIconDB instance.')
        return QIconDB._instance

    def __init__(self):
        super().__init__()
        if QIconDB._instance is not None:
            raise Exception('QIconDB instance already exists.')
        QIconDB._instance = self
        self._cached_np_image = {}
        self._cached_pixmap = {}
        self._cached_image = {}
        self._cached_icon = {}
        self._cached_cursor = {}


    def __dispose__(self):
        QIconDB._instance = None
        self._cached_cursor = None
        self._cached_icon = None
        self._cached_image = None
        self._cached_pixmap = None
        self._cached_np_image = None
        super().__dispose__()
    
    def _get_image(self, icon : IconDB) -> FImage:
        key = (icon, )
        result = self._cached_np_image.get(key, None)
        if result is None:
            result = self._cached_np_image[key] = FImage.from_file(Path(__file__).parent / 'assets' / (icon.name+'.png')).f32()
        return result


    def pixmap(self, icon : IconDB, color : qt.QColor) -> qt.QPixmap:
        key = (icon, color.getRgb(),)
        result = self._cached_pixmap.get(key, None)
        if result is None:
            pixmap = qt.QPixmap_from_FImage(self._get_image(icon))
            result = self._cached_pixmap[key] = qt.QPixmap_colorized(pixmap, color)
        return result

    def image(self, icon : IconDB, color : qt.QColor) -> qt.QImage:
        key = (icon, color.getRgb(),)
        result = self._cached_image.get(key, None)
        if result is None:
            result = self._cached_image[key] = self.pixmap(icon, color).toImage()
        return result

    def icon(self, icon : IconDB, color : qt.QColor) -> qt.QIcon:
        key = (icon, color.getRgb(),)
        result = self._cached_icon.get(key, None)
        if result is None:
            result = self._cached_icon[key] = qt.QIcon(self.pixmap(icon, color))
        return result
    
    def cursor(self, icon : IconDB,  size : Size = Size.M,
                        color : qt.QColor = qt.QColor(255,255,255),
                        stroke_width : int = 1,
                        stroke_color : qt.QColor|None = qt.QColor(0,0,0),                        
                    ) -> qt.QCursor:
        key = (icon, size, color.getRgb(), stroke_width, stroke_color.getRgb())
        result = self._cached_cursor.get(key, None)
        if result is None:
            img = self._get_image(icon)
            
            cursor_size = Size_to_icon_size[size]
            
            base_mask = (img.ch1_from_a().resize(cursor_size,cursor_size, smooth=True).pad(stroke_width))
            base_mask_wider = base_mask.dilate(stroke_width)

            stroke_mask = base_mask_wider-base_mask
            colored_stroke = stroke_mask.ch(3)*FImage.full_f32_like(stroke_mask, stroke_color.getRgbF()[:3])

            colored_base = base_mask.ch(3)*FImage.full_f32_like(base_mask, color.getRgbF()[:3])
            
            np_cursor = FImage.from_bgr_a(colored_base+colored_stroke, base_mask_wider)
            
            cursor = qt.QCursor(qt.QPixmap_from_FImage(np_cursor))
            
            result = self._cached_cursor[key] = cursor
        return result


    _instance : QIconDB = None

