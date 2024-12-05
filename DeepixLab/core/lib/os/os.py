import multiprocessing
import os
import platform
import time
import traceback
from enum import IntEnum

is_win = False
is_linux = False
is_darwin = False

if platform.system() == 'Windows':
    is_win = True
    from ..api.win32 import kernel32, ntdll, user32, winmm, wintypes
elif platform.system() == 'Linux':
    is_linux = True
elif platform.system() == 'Darwin':
    is_darwin = True

_niceness = 0

class ProcessPriority(IntEnum):
    HIGH = 4,
    ABOVE_NORMAL = 3,
    NORMAL = 2,
    BELOW_NORMAL = 1,
    IDLE = 0

class ThreadPriority(IntEnum):
    HIGH = 4,
    ABOVE_NORMAL = 3,
    NORMAL = 2,
    BELOW_NORMAL = 1,
    IDLE = 0

if is_win:
    kernel32_PriorityClass_2_ProcessPriority = {
            kernel32.PriorityClass.HIGH_PRIORITY_CLASS         : ProcessPriority.HIGH        ,
            kernel32.PriorityClass.ABOVE_NORMAL_PRIORITY_CLASS : ProcessPriority.ABOVE_NORMAL,
            kernel32.PriorityClass.NORMAL_PRIORITY_CLASS       : ProcessPriority.NORMAL      ,
            kernel32.PriorityClass.BELOW_NORMAL_PRIORITY_CLASS : ProcessPriority.BELOW_NORMAL,
            kernel32.PriorityClass.IDLE_PRIORITY_CLASS         : ProcessPriority.IDLE        , }

    PriorityClass_2_kernel32_ProcessPriority = { val : key for key, val in kernel32_PriorityClass_2_ProcessPriority.items() }

    kernel32_ThreadPriority_2_ThreadPriority = {
            kernel32.ThreadPriority.THREAD_PRIORITY_HIGHEST      : ThreadPriority.HIGH        ,
            kernel32.ThreadPriority.THREAD_PRIORITY_ABOVE_NORMAL : ThreadPriority.ABOVE_NORMAL,
            kernel32.ThreadPriority.THREAD_PRIORITY_NORMAL       : ThreadPriority.NORMAL      ,
            kernel32.ThreadPriority.THREAD_PRIORITY_BELOW_NORMAL : ThreadPriority.BELOW_NORMAL,
            kernel32.ThreadPriority.THREAD_PRIORITY_IDLE         : ThreadPriority.IDLE        , }

    ThreadPriority_2_kernel32_ThreadPriority = { val : key for key, val in kernel32_ThreadPriority_2_ThreadPriority.items() }


def get_cpu_count() -> int:
    return multiprocessing.cpu_count()

def get_process_priority() -> ProcessPriority:
    """
    """
    global _niceness

    if is_win:
        return kernel32_PriorityClass_2_ProcessPriority[ kernel32.GetPriorityClass (kernel32.GetCurrentProcess()) ]
    elif is_linux:
        prio = {-20 : ProcessPriority.HIGH        ,
                -10 : ProcessPriority.ABOVE_NORMAL,
                0   : ProcessPriority.NORMAL      ,
                10  : ProcessPriority.BELOW_NORMAL,
                20  : ProcessPriority.IDLE        ,
                }[_niceness]
    elif is_darwin:
        prio = {-10 : ProcessPriority.HIGH        ,
                -5  : ProcessPriority.ABOVE_NORMAL,
                0   : ProcessPriority.NORMAL      ,
                5   : ProcessPriority.BELOW_NORMAL,
                10  : ProcessPriority.IDLE        ,
                }[_niceness]

        return prio

def set_process_priority(prio : ProcessPriority):
    """
    """
    global _niceness
    try:
        if is_win:
            hProcess = kernel32.GetCurrentProcess()

            kernel32.SetPriorityClass (hProcess, PriorityClass_2_kernel32_ProcessPriority[prio])
        elif is_linux:
            val = {ProcessPriority.HIGH         : -20,
                   ProcessPriority.ABOVE_NORMAL : -10,
                   ProcessPriority.NORMAL       : 0  ,
                   ProcessPriority.BELOW_NORMAL : 10 ,
                   ProcessPriority.IDLE         : 20 ,
                   }[prio]

            _niceness = os.nice(val)
        elif is_darwin:
            val = {ProcessPriority.HIGH         : -10,
                   ProcessPriority.ABOVE_NORMAL : -5 ,
                   ProcessPriority.NORMAL       : 0  ,
                   ProcessPriority.BELOW_NORMAL : 5  ,
                   ProcessPriority.IDLE         : 10 ,
                   }[prio]

            _niceness = os.nice(val)
    except:
        print(f'set_process_priority error: {traceback.format_exc()}')

def get_thread_priority() -> ThreadPriority:
    """
    """
    if is_win:
        return kernel32_ThreadPriority_2_ThreadPriority [ kernel32.GetThreadPriority (kernel32.GetCurrentThread()) ]

    return ThreadPriority.NORMAL

def set_thread_priority(prio : ThreadPriority):
    """
    """
    global _niceness
    try:
        if is_win:
            kernel32.SetThreadPriority (kernel32.GetCurrentThread(), ThreadPriority_2_kernel32_ThreadPriority[prio])
    except:
        print(f'set_thread_priority error: {traceback.format_exc()}')


prec_set = False

def sleep_precise(sec : float):
    """from 0.001 if supported by OS"""
    if is_win:
        global prec_set
        if not prec_set:
            prec_set = True
            rmin = wintypes.ULONG(0)
            rmax = wintypes.ULONG(0)
            rcur = wintypes.ULONG(0)
            ntdll.NtQueryTimerResolution(rmin, rmax, rcur)

            if rcur != rmax:
                actual = wintypes.ULONG(0)
                ntdll.ZwSetTimerResolution(rmax, True, actual)

        n = -int(sec * 10000000)
        interval = wintypes.LARGE_INTEGER(n)
        ntdll.NtDelayExecution(False, interval)
    else:
        # in *nix already precise according https://docs.python.org/3.0/library/time.html
        time.sleep(sec)

def is_visible_console_window() -> bool:
    if is_win:
        return user32.IsWindowVisible(kernel32.GetConsoleWindow())
    return False

def set_console_window(show : bool):
    if is_win:
        wnd = kernel32.GetConsoleWindow()
        if wnd.value != 0:
            user32.ShowWindow(wnd, user32.SW_SHOW if show else user32.SW_HIDE)

def hide_console_window():
    if is_win:
        wnd = kernel32.GetConsoleWindow()
        if wnd.value != 0:
            user32.ShowWindow(wnd, user32.SW_HIDE)

def show_console_window():
    if is_win:
        wnd = kernel32.GetConsoleWindow()
        if wnd.value != 0:
            user32.ShowWindow(wnd, user32.SW_SHOW)