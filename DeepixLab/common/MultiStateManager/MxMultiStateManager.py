from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from core import ax, mx
from core.lib import path as lib_path
from core.lib.collections import FDict


class MxMultiStateManager(mx.Disposable):
    """
    Manages multiple state dict selectable by user from predefined filepath.
    """

    def __init__(self,  state_or_path : Path|FDict,
                        default_state_name,
                        on_close : Callable[ [str], None ] = None,
                        on_load : Callable[ [FDict, str], None ] = None,
                        get_state : Callable[ [str], ax.Future[FDict] ] = None ):
        """```
            state_or_path

            default_state_name

            on_close(profile_key)         called from main thread

            on_load(state, profile_key)   called from main thread

            get_state(profile_key)        called from main thread
        ```

        State must contain only basic python types, include numpy types.
        """
        super().__init__()

        self._state_or_path = state_or_path
        self._default_state_name = default_state_name

        self._on_close = on_close
        self._on_load = on_load
        self._get_state = get_state

        self._main_thread = ax.get_current_thread()
        self._fg = ax.FutureGroup().dispose_with(self)

        if isinstance(state_or_path, FDict):
            state = state_or_path
        elif isinstance(state_or_path, Path):
            try:
                state = FDict.from_file(state_or_path, Path_func=lambda path: lib_path.abspath(path, state_or_path.parent))
            except Exception as e:
                state = FDict()
        else:
            raise ValueError()

        self._states = state.get('states', FDict() )

        self._mx_error = mx.TextEmitter().dispose_with(self)

        self._mx_state = mx.StateChoice(availuator=lambda: (default_state_name,) + tuple(self._states.keys()) ).dispose_with(self)
        self._mx_state.listen(lambda state, enter, bag=mx.Disposable().dispose_with(self): self._ref_state(state, enter, bag))

    def get_state(self) -> FDict:
        return FDict({  'states' : self._states, })

    @property
    def mx_error(self) -> mx.ITextEmitter_v: return self._mx_error
    @property
    def mx_state(self) -> mx.IStateChoice_v:
        """avail and loaded state"""
        return self._mx_state

    def _ref_state(self, state_name, enter : bool, bag : mx.Disposable):
        if not enter:
            bag.dispose_items()
            if (on_close := self._on_close) is not None:
                on_close(state_name)
        else:
            self._io_thread = ax.Thread(name='IO').dispose_with(bag)
            self._state_fg = ax.FutureGroup().dispose_with(bag)

            state = FDict()
            if state_name != self._default_state_name:
                state = self._states.get(state_name, state)

            if (on_load := self._on_load) is not None:
                on_load(state, state_name)

    def add_state(self, state_name : str):
        """add new state name"""
        if state_name not in self._states:
            self._states = self._states.set(state_name, FDict())
            self._mx_state.reevaluate()

    def remove_state(self, state_name : str):
        """"""
        if state_name in self._states:
            self._states = self._states.remove(state_name)
            self._mx_state.reevaluate()
            if self._mx_state.get() is None:
                self._mx_state.set(self._default_state_name)
            self._save_state_file()

    @ax.task
    def save(self, state_name : str):
        """
        Save current state to particular state_name.

        state_name must be in .state_names.
        """
        yield ax.switch_to(self._main_thread)
        yield ax.attach_to(self._state_fg)

        if state_name not in self._states:
            raise ValueError('state_name not in self._states')

        if (get_state := self._get_state) is not None:
            yield ax.wait(fut := get_state(state_name))
            if fut.succeeded:
                self._states = self._states.set(state_name, fut.result)
            else:
                raise fut.error
            self._save_state_file()

    def _save_state_file(self):
        state_path = self._state_or_path
        if isinstance(state_path, Path):
            state_path_part = state_path.parent / (state_path.name + '.part')

            state_dump = self.get_state().dumps(Path_func=lambda path: lib_path.relpath(path, state_path.parent))

            # Trying to save .part file
            err = None
            file = None
            try:
                file = open(state_path_part, 'wb')
                file.write(state_dump)
                file.flush()
                os.fsync(file.fileno())
            except Exception as e:
                err = e
            finally:
                if file is not None:
                    file.close()
                    file = None
                if err is not None:
                    state_path_part.unlink(missing_ok=True)

            if err is None:
                if state_path.exists():
                    state_path.unlink()
                state_path_part.rename(state_path)
