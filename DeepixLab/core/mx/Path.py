from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path as Path_
from typing import Callable, Sequence

from .Property import IProperty_rv, Property
from .State import IState_rv, State


class IPath_v(IProperty_rv[Path_|None]):
    """View Interface for mx.Path control"""

    @dataclass
    class Config:
        """```
            allow_open(True)  Allows opening existing file/dir.

            allow_new(False)  Allows new file/dir

            allow_rename(False) Allows to rename opened path


            ^ if both are false, will work only as closeable control.

            dir(False)      Accepted directories only, otherwise only files.

            extensions      Accepted file extensions if not directory.
                            example ['jpg','png']

            desc            Description
                            example 'Video File'
                                    'Sequence directory'
        ```"""
        allow_open : bool = True
        allow_new : bool = False
        allow_rename : bool = False
        dir : bool = False
        extensions : Sequence[str]|None = None
        desc : str|None = None

        def acceptable_extensions(self, path : Path_) -> bool:
            return self.dir or self.extensions is None or path.suffix in self.extensions

        def openable(self, path : Path_) -> bool:
            return self.allow_open and (not self.dir or path.is_dir()) and self.acceptable_extensions(path) and path.exists()

        def newable(self, path : Path_) -> bool:
            return self.allow_new and (self.dir or self.acceptable_extensions(path))

        def renameable(self, path : Path_) -> bool:
            return self.allow_rename and (self.dir or self.acceptable_extensions(path))

    @property
    def config(self) -> IPath_v.Config:  ...
    @property
    def mx_opened(self) -> IState_rv[bool]: ...

    def close(self): ...
    def open(self, path : Path_): ...
    def new(self, path : Path_): ...
    def rename(self, path : Path_): ...

class Path(Property[Path_|None], IPath_v):
    """
    mx.Path is Property[Path|None] with mx_opened in order to handle open/close logic.
    """

    def __init__(self,  config : Path.Config = None,
                        on_close : Callable[ [], None] = None,
                        on_open : Callable[ [Path_], bool] = None,
                        on_new : Callable[ [Path_], bool] = None,
                        on_rename : Callable[ [Path_], bool] = None, ):
        """```
        Operate file/dir open/new/close

            allow_open(True)  Allows opening existing file/dir.

            allow_new(False)  Allows new file/dir

            ^ if both are false, will work only as closeable control.

        Auto closes on dispose.
        ```"""
        super().__init__(None)

        self.__mx_opened = State[bool]().set(False).dispose_with(self)

        self.__config = config if config is not None else Path.Config()

        self.__on_close = on_close if on_close is not None else lambda: ...
        self.__on_open = on_open if on_open is not None else lambda p: True
        self.__on_new = on_new if on_new is not None else lambda p: True
        self.__on_rename = on_rename if on_rename is not None else lambda p: True

    @property
    def config(self) -> Path.Config: return self.__config
    @property
    def mx_opened(self) -> IState_rv[bool]: return self.__mx_opened

    def __dispose__(self):
        self.close()
        super().__dispose__()

    def close(self):
        """Close path"""
        if self.__mx_opened.get():
            self.__on_close()
            self.__mx_opened.set(False)
            self.set(None)

    def open(self, path : Path_):
        """Open path if applicable by configuration."""
        if self.__config.openable(path):
            self.close()
            if self.__on_open(path):
                self.set(path)
                self.__mx_opened.set(True)

    def new(self, path : Path_):
        """New path if applicable by configuration."""
        if self.__config.newable(path):
            self.close()

            if self.__on_new(path):
                self.set(path)
                self.__mx_opened.set(True)

    def rename(self, path : Path_):
        """
        avail if opened path
        rename opened path if applicable by configuration.
        """
        if self.__mx_opened.get():
            if self.__config.renameable(path):
                if self.__on_rename(path):
                    self.set(path)
        