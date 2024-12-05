from __future__ import annotations

import queue
from typing import Any, List

from .. import ax, qt
from ._constants import Align
from .QAbstractItemModel import QAbstractItemModel
from .QApplication import QApplication
from .QLabel import QLabel
from .QTableView import QTableView
from .QTimer import QTimer
from .QWindow import QWindow


class QAxMonitorWindow(QWindow):
    def __init__(self):
        super().__init__()
        
        app = QApplication.instance()
        self.set_parent(app)
        self.set_title('AsyncX monitor')
        
        test_model = QTaskTableModel().set_parent(self)
        
        table_view = QTableView().set_model(test_model)
        table_view.set_column_width(0, 300)
        
        (self
            .add( QLabel().set_text('Active tasks').set_align(Align.CenterF).v_compact())
            .add( table_view ))
        
        
class QTaskTableModel(QAbstractItemModel):
    def __init__(self, **kwargs):
        super().__init__()
        self._tasks : List[ax.Task] = []
        self._layout_upd_q = queue.Queue()
        
        ax.g_debug.attach(self._on_task_created, self._on_task_finished)
        
        self._layout_upd_timer = QTimer(on_timeout=self._on_layout_upd_timer).dispose_with(self).set_interval(100).start()
        self._upd_timer = QTimer(on_timeout=self._on_upd_timer).dispose_with(self).set_interval(1000).start()
        
    def __dispose__(self):
        ax.g_debug.detach(self._on_task_created, self._on_task_finished)
        super().__dispose__()
            
    def _on_layout_upd_timer(self, *_):
        q = self._layout_upd_q
        upd = False
        while not q.empty():
            upd = True
            t, v = q.get()
            if t == 0:
                self._tasks.append(v)
            else:
                self._tasks.remove(v)
        
        if upd:      
            self._ev_layout_changed.emit()
        
    def _on_upd_timer(self, *_):
        if len(self._tasks) != 0:
            self._ev_data_changed.emit(self.index(0, 0, qt.QModelIndex()), self.index(len(self._tasks)-1, 1, qt.QModelIndex()))
        
    def _on_task_created(self, task : ax.Task):
        # Called from undetermined thread
        self._layout_upd_q.put((0,task))
        
    def _on_task_finished(self, task : ax.Task):
        # Called from undetermined thread
        self._layout_upd_q.put((1,task))
        
    def flags(self, index : qt.QModelIndex) -> qt.Qt.ItemFlag:
        return qt.Qt.ItemFlag.NoItemFlags
        
    def row_count(self, index : qt.QModelIndex) -> int:
        if not index.isValid():
            return len(self._tasks)
        return 0    
    
    def column_count(self, index : qt.QModelIndex ) -> int:
        return 2
    
    def index(self, row : int, col : int, parent : qt.QModelIndex) -> qt.QModelIndex:
        if not parent.isValid():
            return self.create_index(row, col, None)
        return qt.QModelIndex()
    
    def header_data(self, section: int, orientation: qt.Qt.Orientation, role: int = ...) -> Any:
        if orientation == qt.Qt.Orientation.Horizontal:
            if role == qt.Qt.ItemDataRole.DisplayRole:
                if section == 0:
                    return 'Task name'
                elif section == 1:
                    return 'Time alive (sec)'
        else:
            ...
            #if role == qt.Qt.ItemDataRole.DisplayRole:
            #    return str(section+1)
            
        return None
    
    def data(self, index : qt.QModelIndex, role: int = ...) -> Any: 
        if role == qt.Qt.ItemDataRole.DisplayRole:
            row = index.row()
            
            #if row < len(self._tasks):
            task = self._tasks[row]
            
            col = index.column()
            if col == 0:
                return task.name
            elif col == 1:
                return f'{task.alive_time:.2f}'
            
        return None
    
    def parent(self, child : qt.QModelIndex) -> qt.QModelIndex:
        if not child.isValid():
            return qt.QModelIndex()
        return qt.QModelIndex()
    
    
    
