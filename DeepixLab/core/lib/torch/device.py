from __future__ import annotations

import threading
from dataclasses import dataclass
from functools import cached_property
from typing import List

import torch

from ..collections import FDict

torch.backends.cudnn.benchmark = True

try:
    import torch_directml
except:
    torch_directml = None

@dataclass
class DeviceInfo:
    type : str
    index : int|None
    name : str
    total_memory : int

class Device:
    """
    """
    @staticmethod
    def from_state(state : FDict|None = None) -> Device:
        state = FDict(state)
        return Device(type = state.get('type', 'cpu'), index = state.get('index', 0) )

    @staticmethod
    def from_device(device : torch.device) -> Device:
        return Device(device.type, device.index)

    def __init__(self, type : str, index : int|None):
        self._type = type
        self._index = None if type == 'cpu' else index

    @property
    def type(self) -> str: return self._type
    @property
    def index(self) -> int|None: return self._index
    @property
    def info(self) -> DeviceInfo:
        if not self.is_cpu:
            for info in get_avail_gpu_devices_info():
                if info.type == self._type and info.index == self._index:
                    return info
        # Fallback to CPU
        return get_cpu_device_info()

    @cached_property
    def device(self) -> torch.device:
        """get torch.device. If invalid, returns CPU device."""
        info = self.info
        type = info.type
        index = info.index
        if type == 'dml':
            if torch_directml is not None:
                return torch_directml.device(index)
        elif type == 'cpu':
            return torch.device('cpu')
        else:
            return torch.device(type, index)

    @property
    def is_cpu(self) -> bool: return self._type == 'cpu'

    def get_state(self) -> FDict: return FDict({'type' : self._type, 'index': self._index})

    def __hash__(self): return (self._type, self._index).__hash__()
    def __eq__(self, other):
        if isinstance(self, (Device, torch.device)) and isinstance(other, (Device, torch.device)):
            return self.type == other.type and self.index == other.index
        return False

    def __str__(self):
        info = self.info
        if info.type == 'cpu':
            s = f'[{info.type.upper()}]'
        else:
            s = f'[{info.type.upper()}:{info.index}] {info.name}'
            if (total_memory := info.total_memory) != 0:
                s += f' [{(total_memory / 1024**3) :.3}Gb]'
        return s

    def __repr__(self):
        return f'{self.__class__.__name__} object: ' + self.__str__()

_gpu_devices_info_lock = threading.Lock()
_gpu_devices_info = None

def get_cpu_device_info() -> DeviceInfo: return DeviceInfo(type='cpu', index=None, name='CPU', total_memory=0)

def get_avail_gpu_devices_info() -> List[DeviceInfo]:
    global _gpu_devices_info_lock
    global _gpu_devices_info
    with _gpu_devices_info_lock:
        if _gpu_devices_info is None:
            _gpu_devices_info = []

            for i in range (torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                _gpu_devices_info.append( DeviceInfo(type='cuda', index=i, name=device_props.name, total_memory=device_props.total_memory) )

            if torch_directml is not None:
                for i in range (torch_directml.device_count()):
                    _gpu_devices_info.append( DeviceInfo(type='dml', index=i, name=torch_directml.device_name(i), total_memory=0) )


    return _gpu_devices_info


def get_cpu_device() -> Device: return Device(type='cpu', index=None)

def get_avail_gpu_devices() -> List[Device]:
    return [ Device(type=dev.type, index=dev.index) for dev in get_avail_gpu_devices_info() ]

def get_avail_devices() -> List[Device]: return [get_cpu_device()] + get_avail_gpu_devices()

def get_best_device(devices : List[Device]) -> Device:
    if len(devices) == 0:
        raise ValueError('devices must have at least 1 device')

    gpu_devices = sorted([ device for device in devices if not device.is_cpu ], key=lambda device: device.info.total_memory, reverse=True )
    gpu_devices = gpu_devices + [ device for device in devices if device.is_cpu ]
    return gpu_devices[0]
