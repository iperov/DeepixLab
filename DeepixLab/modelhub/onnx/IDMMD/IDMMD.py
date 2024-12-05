from pathlib import Path
from typing import List

from core.lib import facedesc as fd
from core.lib import onnxruntime as lib_ort
from core.lib.image import FImage


class IDMMD:


    @staticmethod
    def get_available_devices() -> List[lib_ort.DeviceRef]:
        return [lib_ort.get_cpu_device()] + lib_ort.get_avail_gpu_devices()

    def __init__(self, device : lib_ort.DeviceRef ):
        """
        arguments

            device_info    ORTDeviceInfo

                use IDMMD.get_available_devices()
                to determine a list of avaliable devices accepted by model

        raises
            Exception
        """
        if device not in IDMMD.get_available_devices():
            raise Exception(f'device {device} is not in available devices for IDMMD')

        self._sess = sess = lib_ort.InferenceSession_with_device(Path(__file__).parent / 'IDMMD.onnx', device)
        self._input_name = sess.get_inputs()[0].name

    @property
    def input_size(self): return 112

    def extract(self, img : FImage) -> fd.FAnnoID:
        """
        arguments

            img    FImage

        """
        input_size = self.input_size
        feed_img = img.ch1().resize(input_size, input_size).f32().CHW()[None,...]

        out_t, = self._sess.run(None, {self._input_name: feed_img})
        out_t = out_t[0]

        return fd.FAnnoID(out_t)
