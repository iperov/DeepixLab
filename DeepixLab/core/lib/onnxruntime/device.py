from __future__ import annotations

import ctypes
import itertools
import os
import platform
import threading
from dataclasses import dataclass
from typing import List

import onnxruntime as rt

from ..collections import FDict

if platform.system() == 'Windows':
    from ..api.win32 import dxgi as lib_dxgi


@dataclass
class DeviceInfo:
    execution_provider : str
    index : int
    name : str
    total_memory : int
    free_memory : int

class DeviceRef:
    """
    Represents ONNXRuntime device info
    """
    @staticmethod
    def from_state(state : FDict = None) -> DeviceRef:
        state = state or FDict()
        return DeviceRef(execution_provider = state.get('execution_provider', 'CPUExecutionProvider'), index = state.get('index', -1) )


    def __init__(self, execution_provider=None, index=None):
        self._execution_provider : str = execution_provider
        self._index : int = index

    @property
    def execution_provider(self) -> str: return self._execution_provider
    @property
    def index(self) -> int: return self._index
    @property
    def info(self) -> DeviceInfo:
        if not self.is_cpu:
            for info in get_avail_gpu_devices_info():
                if info.execution_provider == self._execution_provider and info.index == self._index:
                    return info
        # Fallback to CPU
        return get_cpu_device_info()

    @property
    def is_cpu(self) -> bool: return self._execution_provider == 'CPUExecutionProvider'

    def get_state(self) -> FDict: return FDict({'execution_provider' : self._execution_provider, 'index': self._index})

    def __hash__(self): return (self._execution_provider, self._index).__hash__()
    def __eq__(self, other):
        if self is not None and other is not None and isinstance(self, DeviceRef) and isinstance(other, DeviceRef):
            return self._execution_provider == other._execution_provider and self._index == other._index
        return False

    def __str__(self):
        if self.is_cpu:
            return f"CPU"
        else:
            info = self.info

            ep = info.execution_provider
            if ep == 'CUDAExecutionProvider':
                return f"[{self._index}] {info.name} [{(info.total_memory / 1024**3) :.3}Gb] [CUDA]"
            elif ep == 'DmlExecutionProvider':
                return f"[{self._index}] {info.name} [{(info.total_memory / 1024**3) :.3}Gb] [DirectX12]"
            else:
                raise NotImplementedError()

    def __repr__(self):
        return f'{self.__class__.__name__} object: ' + self.__str__()

_devices_info_lock = threading.Lock()
_devices_info = None

def get_cpu_device_info() -> DeviceInfo:
    return DeviceInfo(index=-1, execution_provider='CPUExecutionProvider', name='CPU', total_memory=0, free_memory=0)

def get_avail_gpu_devices_info(include_cpu=True, cpu_only=False) -> List[DeviceInfo]:
    """returns a list of available DeviceInfo"""
    devices_info = []
    if not cpu_only:
        global _devices_info_lock
        global _devices_info
        with _devices_info_lock:
            if _devices_info is None:
                _devices_info = _get_ort_devices_info()
        devices_info += _devices_info
    if include_cpu:
        devices_info.append(get_cpu_device_info())

    return devices_info


def get_cpu_device() -> DeviceRef: return DeviceRef(execution_provider='CPUExecutionProvider', index=-1)

def get_avail_gpu_devices() -> List[DeviceRef]:
    return [ DeviceRef(execution_provider=dev.execution_provider, index=dev.index) for dev in get_avail_gpu_devices_info() ]

def get_best_device(devices : List[DeviceRef]) -> DeviceRef:
    if len(devices) == 0:
        raise ValueError('devices must have at least 1 device')

    gpu_devices = sorted([ device for device in devices if not device.is_cpu ], key=lambda device: device.info.total_memory, reverse=True )
    gpu_devices = gpu_devices + [ device for device in devices if device.is_cpu ]
    return gpu_devices[0]

def _get_ort_devices_info() -> List[DeviceInfo]:
    """
    Determine available ORT devices, and place info about them to os.environ,
    they will be available in spawned subprocesses.

    Using only python ctypes and default lib provided with NVIDIA drivers.
    """

    devices_info = []

    prs = rt.get_available_providers()
    if 'CUDAExecutionProvider' in prs: #not lib_appargs.get_arg_bool('NO_CUDA') and
        os.environ['CUDA_​CACHE_​MAXSIZE'] = '2147483647'
        try:
            libnames = ('nvcuda.dll', 'libcuda.so', 'libcuda.dylib')
            for libname in libnames:
                try:
                    cuda = ctypes.CDLL(libname)
                except:
                    continue
                else:
                    break
            else:
                return

            nGpus = ctypes.c_int()
            name = b' ' * 200
            cc_major = ctypes.c_int()
            cc_minor = ctypes.c_int()
            freeMem = ctypes.c_size_t()
            totalMem = ctypes.c_size_t()
            device = ctypes.c_int()
            context = ctypes.c_void_p()


            if cuda.cuInit(0) == 0 and \
                cuda.cuDeviceGetCount(ctypes.byref(nGpus)) == 0:
                for i in range(nGpus.value):
                    if cuda.cuDeviceGet(ctypes.byref(device), i) != 0 or \
                        cuda.cuDeviceGetName(ctypes.c_char_p(name), len(name), device) != 0 or \
                        cuda.cuDeviceComputeCapability(ctypes.byref(cc_major), ctypes.byref(cc_minor), device) != 0:
                        continue

                    if cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device) == 0:
                        if cuda.cuMemGetInfo_v2(ctypes.byref(freeMem), ctypes.byref(totalMem)) == 0:
                            cc = cc_major.value * 10 + cc_minor.value

                            devices_info.append( DeviceInfo(execution_provider = 'CUDAExecutionProvider',
                                                            index = i,
                                                            name = name.split(b'\0', 1)[0].decode(),
                                                            total_memory = totalMem.value,
                                                            free_memory = freeMem.value))
                        cuda.cuCtxDetach(context)
        except Exception as e:
            print(f'CUDA devices initialization error: {e}')

    if 'DmlExecutionProvider' in prs:
        # onnxruntime-directml has no device enumeration API for users. Thus the code must follow the same logic
        # as here https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/dml/dml_provider_factory.cc

        dxgi_factory = lib_dxgi.create_DXGIFactory4()
        if dxgi_factory is not None:
            for i in itertools.count():
                adapter = dxgi_factory.enum_adapters1(i)
                if adapter is not None:
                    desc = adapter.get_desc1()
                    if desc.Flags != lib_dxgi.DXGI_ADAPTER_FLAG.DXGI_ADAPTER_FLAG_SOFTWARE and \
                        not (desc.VendorId == 0x1414 and desc.DeviceId == 0x8c):

                        devices_info.append( DeviceInfo(execution_provider = 'DmlExecutionProvider',
                                                            index = i,
                                                            name = desc.Description,
                                                            total_memory = desc.DedicatedVideoMemory,
                                                            free_memory = desc.DedicatedVideoMemory))
                    adapter.Release()
                else:
                    break
            dxgi_factory.Release()

    return devices_info

