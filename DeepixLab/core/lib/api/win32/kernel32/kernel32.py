from ctypes import POINTER, c_void_p, c_wchar_p

from ..wintypes import (BOOL, DWORD, ERROR, HANDLE, HRESULT, HWND, INT,
                        LARGE_INTEGER, dll_import)


class PriorityClass(DWORD):
    HIGH_PRIORITY_CLASS         = DWORD(0x00000080)
    ABOVE_NORMAL_PRIORITY_CLASS = DWORD(0x00008000)
    NORMAL_PRIORITY_CLASS       = DWORD(0x00000020)
    BELOW_NORMAL_PRIORITY_CLASS = DWORD(0x00004000)
    IDLE_PRIORITY_CLASS         = DWORD(0x00000040)

class ThreadPriority(INT):
    THREAD_PRIORITY_HIGHEST      = INT(2)
    THREAD_PRIORITY_ABOVE_NORMAL = INT(1)
    THREAD_PRIORITY_NORMAL       = INT(0)
    THREAD_PRIORITY_BELOW_NORMAL = INT(-1)
    THREAD_PRIORITY_LOWEST       = INT(-2)
    THREAD_PRIORITY_IDLE         = INT(-15)

@dll_import('kernel32')
def CreateEventW(lpEventAttributes : c_void_p, bManualReset : BOOL, bInitialState : BOOL, lpName : c_wchar_p) -> HANDLE: ...

@dll_import('kernel32')
def GetConsoleWindow() -> HWND: ...

@dll_import('kernel32')
def GetCurrentProcess() -> HANDLE: ...

@dll_import('kernel32')
def GetCurrentThread() -> HANDLE: ...

@dll_import('kernel32')
def GetPriorityClass(hProcess : HANDLE) -> DWORD: ...

@dll_import('kernel32')
def GetThreadPriority(hThread : HANDLE ) -> INT: ...

@dll_import('kernel32')
def QueryPerformanceCounter(lpPerformanceCount : POINTER(LARGE_INTEGER) ) -> BOOL: ...

@dll_import('kernel32')
def QueryPerformanceFrequency(lpFrequency : POINTER(LARGE_INTEGER) ) -> BOOL: ...

@dll_import('kernel32')
def SetPriorityClass(hProcess : HANDLE, priority_class : DWORD ) -> BOOL: ...

@dll_import('kernel32')
def SetThreadPriority(hThread : HANDLE, priority : INT ) -> BOOL: ...

@dll_import('kernel32')
def WaitForSingleObject(hHandle : HANDLE, dwMilliseconds : DWORD) -> DWORD:...

@dll_import('kernel32')
def Sleep( dwMilliseconds : DWORD) -> None:...


