from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Callable, Dict

import torch
import torch.nn as nn
import torch.optim as opt

from ..collections import FDict, HFDict
from .device import Device, get_cpu_device


class ModuleManager:
    """
    Convenient class to operate multiple torch Module+Optimizer on demand by key.

    Non thread-safe.
    """
    @dataclass
    class _Record():
        module : nn.Module
        opt : opt.Optimizer|None
        device : Device|None
        train : bool

    def __init__(self, state : FDict|None = None):
        """
        ```
            module_dict     dict of callable by key which instantiates module on demand

            state           keeps state until first module request
        ```

        raise no errors.
        """
        super().__init__()

        self._module_factory = None
        self._optimizer_factory = None

        self._initial_state = HFDict(state)
        self._records : Dict[Any, ModuleManager._Record] = {}

    def set_module_factory(self, func : Callable[[ModuleManager, Any], nn.Module]):
        self._module_factory = func

    def set_optimizer_factory(self, func : Callable[[ModuleManager, nn.Module], opt.Optimizer]):
        self._optimizer_factory = func

    def get_state(self) -> FDict:
        state = FDict(self._initial_state)

        for key, record in self._records.items():
            model_state_bytes = io.BytesIO()
            torch.save(record.module.state_dict(), model_state_bytes)
            model_state_bytes = model_state_bytes.getbuffer().tobytes()

            state = state.set(f'{key}_state_bytes', model_state_bytes)

            if record.opt is not None:
                opt_state_bytes = io.BytesIO()
                torch.save(record.opt.state_dict(), opt_state_bytes)
                opt_state_bytes = opt_state_bytes.getbuffer().tobytes()

                state = state.set(f'{key}_opt_state_bytes', opt_state_bytes)

        return state

    def reset_all(self):
        self._initial_state = HFDict()
        self._records = {}

    def reset(self, *key_or_list):
        """
        Reset specific module, list of modules, or all modules if not specified.

        Module will be reinstantiated on next request."""

        if len(key_or_list) == 0:
            key_or_list = list(self._records.keys())

        for key in key_or_list:
            if key in self._records:
                self._records.pop(key)
            self._initial_state[f'{key}_state_bytes'] = None
            self._initial_state[f'{key}_opt_state_bytes'] = None

    def _get_record(self, key : Any) -> ModuleManager._Record:
        """raise on error"""

        record = self._records.get(key, None)
        if record is None:
            # Module does not exists. Instantiate new.

            # Try to instantiate module on CPU
            module = self._module_factory(self, key)
            if not isinstance(module, nn.Module):
                raise ValueError(f'Instantiation func must return nn.Module')

            # Load from initial state
            state_key = f'{key}_state_bytes'
            if (model_state_bytes := self._initial_state.get(state_key, None)) is not None:
                model_state = torch.load(io.BytesIO(model_state_bytes), map_location='cpu', weights_only=True)
            else:
                model_state = None

            if model_state is not None:
                try:
                    module.load_state_dict(model_state)#, strict=False
                except:
                    # Something goes wrong during loading.
                    # We need to instantiate again, because weights may be loaded partially that is unacceptable.
                    module = None # Delete reference in order to free RAM before reinstantiation
                    module = self._module_factory(self, key)

            # eval mode is default for just created Module
            module.eval()

            # No exception at this point, now we can update vars.

            # Consume initial state
            self._initial_state[state_key] = None
            # Create record
            record = self._records[key] = ModuleManager._Record(module=module, opt=None, device=get_cpu_device(), train=False)
        return record

    def get_module(self, key : Any, device : Device = None, train : bool = None) -> nn.Module:
        """
        Get/create module by key.

        module's `device` and `train` will be updated

        raises on error. If error: Module is unusable and should be resetted or request with CPU device.
        """
        record = self._get_record(key)

        if device is not None and record.device != device:
            # Update device to the requested
            try:
                record.module.to(device.device)
                if record.opt is not None:
                    # Has optimizer. Reload state_dict, thus device of parameters will be changed automatically
                    record.opt.load_state_dict(record.opt.state_dict())
            except Exception as e:
                # Exception during full or partially changed device
                # reset device in order to set again in next time.
                record.device = None
                # raise error assuming User will choose other device
                raise e

            # No exception at this point, success
            record.device = device

        if train is not None and record.train != train:
            # Update train/eval
            if train:
                record.module.train()
            else:
                record.module.eval()
            # No exception at this point, success
            record.train = train

        return record.module

    def get_optimizer(self, key : Any) -> nn.Module:
        """
        raises on error
        """
        record = self._get_record(key)

        if record.opt is None:
            opt = self._optimizer_factory(self, record.module)

            # Load from initial state
            state_key = f'{key}_opt_state_bytes'
            if (opt_state_bytes := self._initial_state.get(state_key, None)) is not None:
                opt_state = torch.load(io.BytesIO(opt_state_bytes), map_location='cpu', weights_only=True)
            else:
                opt_state = None

            if opt_state is not None:
                try:
                    opt.load_state_dict(opt_state)
                except:
                    # Something goes wrong during loading.
                    # We need to instantiate again, because weights may be loaded partially that is unacceptable.
                    opt = None # Delete reference in order to free RAM before reinstantiation
                    opt = self._optimizer_factory(self, record.module)

            # No exception at this point, now we can update vars.

            # Consume initial state
            self._initial_state[state_key] = None
            # Update record
            record.opt = opt

        return record.opt
