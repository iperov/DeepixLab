from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Deque, Dict

if TYPE_CHECKING:
    from .Task import Task


class ThreadLocalStorage:
    _by_ident : Dict[int, 'ThreadLocalStorage'] = {}

    def __init__(self):
        self._task_exec_stack : Deque[Task] = deque() # Stack of Tasks execution
