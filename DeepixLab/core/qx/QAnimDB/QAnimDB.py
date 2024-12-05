from __future__ import annotations

from ... import mx, qt
from ...lib.image import gen as lib_gen
from ..QImageAnim import QImageAnim
from .AnimDB import AnimDB


class QAnimDB(mx.Disposable):
    
    @staticmethod
    def instance() -> QAnimDB:
        if QAnimDB._instance is None:
            raise Exception('No QAnimDB instance.')
        return QAnimDB._instance

    def __init__(self):
        super().__init__()
        if QAnimDB._instance is not None:
            raise Exception('QAnimDB instance already exists.')
        QAnimDB._instance = self
        self._cached = {}

    def __dispose__(self):
        QAnimDB._instance = None
        self._cached = None
        super().__dispose__()

    def get(self, anim : AnimDB, color : qt.QColor) -> QImageAnim:
        return self._get(anim, color)

    def _get(self, anim : AnimDB, color : qt.QColor):
        key = (anim, color.getRgb())

        result = self._cached.get(key, None)
        if result is None:
            result = self._cached[key] = gen_func[anim](color)
            
        return result

    _instance : QAnimDB = None



def generate_icon_loading(color : qt.QColor, size=1024, frame_count=10) -> QImageAnim:
    bg_color = color.getRgbF()[2::-1]+(0,)
    fg_color = color.getRgbF()[2::-1]+(1,)
    
    return QImageAnim(  frames = [  qt.QImage_from_np(lib_gen.icon_loading(size, size*0.5*0.75, size*0.5, 4, bg_color, fg_color, u_time=frame_id/frame_count).HWC())
                                    for frame_id in range(frame_count) ],
                        fps=frame_count)

gen_func = {
    AnimDB.icon_loading : generate_icon_loading,
}