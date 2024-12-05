"""
AsyncX library. Designed and developed from scratch by github.com/iperov

Work with tasks in multiple threads easily.
```
Future  .finished  True:   .succeeded   True:   .result:      task result
                    |                       |
                    |                       |
                    |                       False:  mean cancelled
                    |                               .exception():   None        - no error provided
                    False:  mean active                             |
                                                                    Exception   - class of Exception error

Future -> Task
```


"""
import sys as _sys

if not (_sys.version_info.major >= 4 or
       (_sys.version_info.major >= 3 and _sys.version_info.minor >= 12)):
    raise Exception("AsyncX requires Python 3.12+")

from .clear import clear
from .Future import Future
from .FutureGroup import FutureGroup
from .FutureSet import FutureSet
from .FutureGenerator import FutureGenerator
from .g_debug import g_debug
from .g_log import g_log
from .Task import (Task, TaskFinishException, attach_to, cancel, detach,
                   get_current_task, propagate, sleep, success, switch_to,
                   task, wait)
from .Thread import Thread, get_current_thread
from .ThreadPool import CPU_COUNT, ThreadPool
