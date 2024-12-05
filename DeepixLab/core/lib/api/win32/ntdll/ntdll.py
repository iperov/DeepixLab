from ctypes import POINTER

from ..wintypes import BOOL, LARGE_INTEGER, NTSTATUS, ULONG, dll_import


@dll_import('ntdll')
def ZwSetTimerResolution(RequestedResolution : ULONG, Set : BOOL, ActualResolution : POINTER(ULONG)) -> NTSTATUS: ...

@dll_import('ntdll')
def NtQueryTimerResolution(MinimumResolution : POINTER(ULONG), MaximumResolution : POINTER(ULONG), CurrentResolution : POINTER(ULONG)) -> None: ...

@dll_import('ntdll')
def NtDelayExecution(Alertable : BOOL, DelayInterval : POINTER(LARGE_INTEGER) ) -> NTSTATUS: ...
