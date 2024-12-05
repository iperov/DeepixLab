from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Callable

from core import ax, mx
from core.lib import path as lib_path
from core.lib.collections import FDict


class MxFileStateManager(mx.Disposable):
    """
    Manages state dict stored in file selectable by user.
    """

    def __init__(self,  file_suffix : str,
                        on_close : Callable[ [], None ],
                        on_load : Callable[ [FDict], None ],
                        get_state : Callable[ [], ax.Future[FDict] ] ):
        """```
            file_suffix         example .dxf

            on_close       called from main thread

            on_load        called from main thread

            get_state      called from main thread
        ```

        Automatically closes on dispose.
        """
        super().__init__()

        self._on_close = on_close
        self._on_load = on_load
        self._get_state = get_state

        self._main_thread = ax.get_current_thread()

        self._io_thread = ax.Thread(name='IO').dispose_with(self)

        self._mx_error = mx.TextEmitter().dispose_with(self)
        self._mx_initialized = mx.State[bool]().set(False).dispose_with(self)
        self._mx_loading_progress = mx.Progress().dispose_with(self)

        bag = mx.Disposable().dispose_with(self)

        self._mx_path = mx.Path(mx.Path.Config(allow_open=True, allow_new=True, allow_rename=True, extensions=[file_suffix], desc=f'*{file_suffix}'),
                                        on_close=lambda bag=bag: bag.dispose_items(),
                                        on_open=lambda path, bag=bag: self._on_path_open(path, bag),
                                        on_new=lambda path, bag=bag: self._on_path_new(path, bag),
                                        on_rename=self._on_path_rename,
                                    ).dispose_with(self)

    def _on_path_open(self, path : Path, bag : mx.Disposable) -> bool:
        self._state_path = path
        self._initialize(path, rel_path=path.parent, bag=bag)
        return True

    def _on_path_new(self, path : Path, bag : mx.Disposable) -> bool:
        self._state_path = path
        self._initialize(rel_path=path.parent, bag=bag)
        return True

    def _on_path_rename(self, path : Path) -> bool:
        self._state_path = path
        return True

    @property
    def mx_error(self) -> mx.ITextEmitter_v|None: return self._mx_error
    @property
    def mx_path(self) -> mx.IPath_v: return self._mx_path
    @property
    def mx_initialized(self) -> mx.IState_rv[bool]: return self._mx_initialized
    @property
    def mx_loading_progress(self) -> mx.IProgress_rv:
        """avail if mx_initialized == False"""
        return self._mx_loading_progress

    # Below Avail when .mx_initialized == True
    @property
    def mx_save_progress(self) -> mx.IProgress_rv|None: return self._mx_save_progress
    @property
    def mx_backup_progress(self) -> mx.IProgress_rv|None: return self._mx_backup_progress
    @property
    def mx_autosave(self) -> mx.INumber_v|None: return self._mx_autosave
    @property
    def mx_autobackup(self) -> mx.INumber_v|None: return self._mx_autobackup
    @property
    def mx_backup_count(self) -> mx.INumber_v|None: return self._mx_backup_count


    @ax.task
    def _initialize(self, state_path : Path|None = None, rel_path : Path = None, bag : mx.Disposable = ...):
        yield ax.switch_to(self._main_thread)
        yield ax.attach_to(ax.FutureGroup().dispose_with(bag))

        self._rel_path = rel_path
        self._mx_loading_progress.start().set_caption('@(Loading)...').set(25, 100)

        yield ax.switch_to(self._io_thread)

        state = FDict()
        err = None
        if state_path is not None and state_path.exists():
            try:
                state = FDict.from_file(state_path, Path_func=lambda path: lib_path.abspath(path, state_path.parent))
            except Exception as e:
                err = e

        yield ax.switch_to(self._main_thread)

        self._mx_loading_progress.set(75)

        if err is None:
            self._on_load(state.get('state', FDict()))

        self._mx_loading_progress.finish()

        if err is None:
            self._mx_autosave = mx.Number(state.get('autosave', 25), config=mx.Number.Config(min=0, max=3600)).dispose_with(bag)
            self._mx_autosave.reflect(lambda _: setattr(self, '_last_save_time', time.time()))

            self._mx_autobackup = mx.Number(state.get('autobackup', 0), config=mx.Number.Config(min=0, max=3600)).dispose_with(bag)
            self._mx_autobackup.reflect(lambda _: setattr(self, '_last_backup_time', time.time()))

            self._mx_backup_count = mx.Number(state.get('backup_count', 8), config=mx.Number.Config(min=1, max=32)).dispose_with(bag)

            self._mx_save_progress = mx.Progress().dispose_with(bag)
            self._mx_backup_progress = mx.Progress().dispose_with(bag)

            self._save_fg = ax.FutureGroup().dispose_with(bag)
            self._auto_save_fg = ax.FutureGroup().dispose_with(bag)

            mx.CallOnDispose(lambda: self._on_close()).dispose_with(bag)
            mx.CallOnDispose(lambda: self._mx_initialized.set(False)).dispose_with(bag)

            self._run_auto_save_task()


            self._mx_initialized.set(True)
        else:
            self._mx_error.emit(str(err))
            yield ax.detach()
            self._mx_path.close()


    @ax.task
    def _run_auto_save_task(self):
        yield ax.switch_to(self._main_thread)
        yield ax.attach_to(self._auto_save_fg)

        while True:
            if (autosave := self._mx_autosave.get()) != 0 and \
               (time.time() - self._last_save_time) >= autosave*60:
                yield ax.wait(self.save())

            if (autobackup := self._mx_autobackup.get()) != 0 and \
               (time.time() - self._last_backup_time) >= autobackup*60:
                yield ax.wait(self.save(backup=True))

            yield ax.sleep(1)

    @ax.task
    def save(self, as_name : str = None, backup=False):
        """
        Save task.
        Avail in state==Initialized, otherwise cancelled.
        Task will be cancelled if other save task is already running.
        """
        yield ax.switch_to(self._main_thread)
        if not self._mx_initialized.get():
            # Nothing to save
            yield ax.cancel()
        yield ax.attach_to(self._save_fg, max_tasks=1)

        root_path = self._state_path.parent
        state_path = self._state_path
        if as_name is not None:
            state_path = root_path / f'{state_path.stem} — {as_name}.state'

        state_path_part = state_path.parent / (state_path.name + '.part')

        self._last_save_time = time.time()
        self._mx_save_progress.start().set(0, 100)
        if backup:
            self._last_backup_time = self._last_save_time
            self._mx_backup_progress.start().set(0, 100)

        # Collect state
        yield ax.wait(state_fut := self._get_state())

        err = None
        if state_fut.succeeded:
            state = state_fut.result
        else:
            err = state_fut.error

        if err is None:
            self._mx_save_progress.set(30)
            if backup:
                self._mx_backup_progress.set(30)

            yield ax.switch_to(self._io_thread)

            state = FDict({ 'state' : state,
                            'autosave' : self._mx_autosave.get(),
                            'autobackup' : self._mx_autobackup.get(),
                            'backup_count' : self._mx_backup_count.get(),
                             })

            try:
                state.dump_to_file(state_path_part, Path_func=lambda path: lib_path.relpath(path, self._rel_path))
            except Exception as e:
                err=e

            yield ax.switch_to(self._main_thread)

            self._mx_save_progress.set(60)
            if backup:
                self._mx_backup_progress.set(60)

            yield ax.switch_to(self._io_thread)

            if err is None:
                if state_path.exists():
                    state_path.unlink()
                state_path_part.rename(state_path)

                if backup:
                    try:
                        backup_count = self._mx_backup_count.get()

                        # Delete redundant backups
                        for filepath in lib_path.get_files_paths(root_path):
                            if filepath.suffix == state_path.suffix:
                                if len(splits := filepath.stem.split(f'{state_path.stem} — bckp — ')) == 2:
                                    backup_id = int(splits[1])
                                    if backup_id > backup_count:
                                        filepath.unlink()

                        # Renaming existing backups to free backup slot 01
                        for i in range(backup_count-1,0,-1):
                            p1 = root_path / f'{state_path.stem} — bckp — {i:02}{state_path.suffix}'
                            p2 = root_path / f'{state_path.stem} — bckp — {i+1:02}{state_path.suffix}'
                            if p2.exists():
                                p2.unlink()
                            if p1.exists():
                                p1.rename(p2)

                        # Copy saved state file to backup slot 01
                        shutil.copy(state_path, root_path / f'{state_path.stem} — bckp — 01{state_path.suffix}')

                    except Exception as e:
                        err = e

        yield ax.switch_to(self._main_thread)

        yield ax.sleep(0.2)

        self._mx_save_progress.finish()
        if backup:
            self._mx_backup_progress.finish()

        if err is not None:
            # Something goes wrong.
            # Close and go to error state in order not to waste comp time.

            yield ax.detach()
            self._mx_path.close()
            self._mx_error.emit(str(err))

