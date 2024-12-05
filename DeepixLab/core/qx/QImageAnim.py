from typing import Sequence

from .. import qt


class QImageAnim:
    """holds image animation"""

    def __init__(self, frames : Sequence[qt.QImage], fps : float):
        super().__init__()
        self._frames = frames
        self._cached = {}
        
        self._fps = fps
        self._frame_count = len(frames)
    
    @property
    def fps(self) -> float: return self._fps
    @property
    def frame_count(self) -> int: return self._frame_count
    @property
    def duration(self) -> float:
        """duration in sec float"""
        return self._frame_count / self._fps
    
    def get_image(self, frame_id : int, size : qt.QSize = None) -> qt.QImage: 
        if size is not None:      
            key = (qt.QImage, frame_id, size)
            if (image := self._cached.get(key, None)) is None:
                image = self._cached[key] = self.get_image(frame_id).scaled(size, mode=qt.Qt.TransformationMode.SmoothTransformation)
            return image
        else:
            return self._frames[frame_id]
    
    def get_pixmap(self, frame_id : int, size : qt.QSize = None) -> qt.QPixmap: 
        key = (qt.QPixmap, frame_id, size)
        if (pixmap := self._cached.get(key, None)) is None:
            pixmap = self._cached[key] = qt.QPixmap(self.get_image(frame_id, size=size))
        return pixmap
    
 