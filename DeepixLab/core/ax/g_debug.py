from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Callable

from .Thread import Thread

if TYPE_CHECKING:
    from .Task import Task


class _GDebug:
    def __init__(self):
        self._lock = threading.RLock()
        self._active_tasks = set()
        self._task_created_funcs = set()
        self._task_finished_funcs = set()
        
    def attach(self, on_task_created : Callable[ [Task], None ],
                     on_task_finished : Callable[ [Task], None ],
                     emit_current = True,
                    ):
        """ 
            on_task_created     called from undetermined thread
            
            on_task_finished    called from undetermined thread
        """
        with self._lock:
            self._task_created_funcs.add(on_task_created)
            self._task_finished_funcs.add(on_task_finished)
                
            if emit_current:
                for task in tuple(self._active_tasks):
                    on_task_created(task)
    
    def detach(self, on_task_created : Callable[ [Task], None ],
                     on_task_finished : Callable[ [Task], None ],):
        """"""
        with self._lock:
            self._task_created_funcs.remove(on_task_created)
            self._task_finished_funcs.remove(on_task_finished)
     
               
    def _on_task_created(self, task : Task):
        with self._lock:
            self._active_tasks.add(task)
            funcs = tuple(self._task_created_funcs)
        
        for func in funcs:
            func(task)
    
    def _on_task_finished(self, task : Task):
        with self._lock:
            self._active_tasks.remove(task)
            funcs = tuple(self._task_finished_funcs)
        
        for func in funcs:
            func(task)


    def print_info(self):
        """
        Prints active Threads and Tasks
        """
        s = ''

        active_tasks = set(self._active_tasks)

        if len(Thread._by_ident) != 0:
            s += '\nUnfinalized threads: '

            for i, thread in enumerate(Thread._by_ident.values()):

                for task in thread.get_active_tasks():
                    if task in active_tasks:
                        active_tasks.remove(task)

                s += f'\n[{i}]: {thread.get_printable_info(include_tasks=True)}'

        if len(active_tasks) != 0:
            s += '\nTasks not attached to threads: '
            for i, task in enumerate(active_tasks):
                s += f'\n[{i}]: {task}'

        if len(s) != 0:
            s = '\nAsyncX debug info:' + s + '\n'
            print(s)
            
g_debug = _GDebug()