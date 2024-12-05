from ..wintypes import dll_import, HWND, INT, BOOL

SW_HIDE = 0
SW_SHOW = 5

@dll_import('user32')
def ShowWindow( wnd : HWND, nCmdShow : INT ) -> None: ...

@dll_import('user32')
def IsWindowVisible( wnd : HWND ) -> BOOL: ...
