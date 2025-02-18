from typing import Callable, Dict

from .. import ax, qt
from .QGridItemView import QGridItemView
from .FGridItemView import FGridItemView
from ..lib.collections import shuffled

class QCachedGridItemView(QGridItemView):
    """
    Cached version of QGridItemView.
    """

    def __init__(self, fut_get_item_pixmap : Callable[ [int, qt.QSize], ax.Future[qt.QPixmap] ] ):
        """
            fut_get_item_pixmap (item_id, QSize) -> Future[QPixmap]
        """
        super().__init__()
        self._fut_get_item_pixmap = fut_get_item_pixmap

        self._fg = ax.FutureGroup().dispose_with(self)
        self._caching_fg = ax.FutureGroup().dispose_with(self)
        
        self._cached : Dict[int, qt.QPixmap ] = {}
        self._caching_futs : Dict[int, ax.Future[qt.QPixmap] ] = {}
        
        self._bg_cacher()
        
    def apply_model(self, new_m: FGridItemView):
        m = self._m
        super().apply_model(new_m)
        new_m = self._m
        
        if new_m.item_size != m.item_size or \
           new_m.item_count != m.item_count:
            # Recache all items
            self.update_items()
                
    def update_items(self, clear : bool = False):
        """Recache and repaint."""
        if clear:
            self._cached = {}
            
        for fut in self._caching_futs.values():
            fut.cancel()
        self._caching_futs = {}
        
        self.update()
    
    @ax.task
    def _bg_cacher(self):
        yield ax.attach_to(self._fg)
        
        # counter = 0
        while True:
            cached = self._cached
            caching_futs = self._caching_futs
            
            m = self._m
            
            item_size = qt.QSize_from_FVec2(m.item_size)
            
            for item_id in range( max(0, m.v_item_start-m.v_col_count),
                                  min(m.v_item_start + m.v_item_count+m.v_col_count, m.item_count) ):
                if self._caching_fg.count >= ax.CPU_COUNT*2:
                    break
                if item_id not in caching_futs:
                    caching_futs[item_id] = self._caching_task(item_id, item_size)
                                
            # Lazy removing out of scope 
            item_scope_start = max(0, m.v_item_start - m.v_item_count)
            item_scope_end   = min(m.v_item_start + m.v_item_count*2, m.item_count)
            
            for cached_item_id in cached:
                if cached_item_id < item_scope_start or cached_item_id >= item_scope_end:
                    cached.pop(cached_item_id)
                    break
            for cached_item_id in caching_futs:
                if cached_item_id < item_scope_start or cached_item_id >= item_scope_end:
                    caching_futs.pop(cached_item_id).cancel()
                    break        
                  
            yield ax.sleep(0)
            
            
    @ax.task
    def _caching_task(self, item_id, item_size : qt.QSize):
        yield ax.attach_to(self._caching_fg, detach_parent=False)
        
        yield ax.wait( fut := self._fut_get_item_pixmap(item_id, item_size) )
        
        if not fut.succeeded:   
            yield ax.cancel(fut.error)
        
        self._cached[item_id] = fut.result
        
        self.update()
        

    def _on_paint_item(self, item_id: int, qp: qt.QPainter, rect: qt.QRect):
        pixmap = self._cached.get(item_id, None)
            
        if pixmap is not None:
            qp.drawPixmap(rect, pixmap)
        else:
            qp.drawText(rect, qt.Qt.AlignmentFlag.AlignCenter, '...')



