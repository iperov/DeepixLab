from pathlib import Path
from typing import List

import numpy as np

from core.lib import facedesc as fd
from core.lib import onnxruntime as lib_ort
from core.lib.image import FImage
from core.lib.math import FVec2fArray


class InsightFace2D106:

    @staticmethod
    def get_available_devices() -> List[lib_ort.DeviceRef]:
        return [lib_ort.get_cpu_device()] + lib_ort.get_avail_gpu_devices()

    def __init__(self, device : lib_ort.DeviceRef ):
        """
        arguments

            device_info    ORTDeviceInfo

                use InsightFace2D106.get_available_devices()
                to determine a list of avaliable devices accepted by model

        raises
            Exception
        """
        if device not in InsightFace2D106.get_available_devices():
            raise Exception(f'device {device} is not in available devices for InsightFace2D106')

        path = Path(__file__).parent / 'InsightFace2D106.onnx'
        self._sess = sess = lib_ort.InferenceSession_with_device(str(path), device)
        self._input_name = sess.get_inputs()[0].name

    def get_input_size(self) -> int:
        return 192

    def extract(self, img : FImage) -> fd.FAnnoLmrk2D106:
        """
        arguments

            img    FImage

        returns
        """
        img = img.bgr().swap_rb().u8()
        H, W, _ = img.shape

        input_size = self.get_input_size()

        h_scale = H / input_size
        w_scale = W / input_size

        feed_img = img.resize(input_size, input_size).CHW()[None,...]
        feed_img = feed_img.astype(np.float32)

        lmrks, = self._sess.run(None, {self._input_name: feed_img})
        lmrks = lmrks.reshape((1,106,2))[0]
        lmrks /= 2.0
        lmrks += (0.5, 0.5)
        lmrks *= (w_scale, h_scale)
        lmrks *= (W, H)

        # fix incorrect order of landmarks
        lmrks = lmrks[_lmrk_fix]

        return fd.FAnnoLmrk2D106(FVec2fArray(lmrks))

_lmrk_fix = [1, 9, 10, 11, 12, 13, 14, 15, 16, 2, 3, 4, 5, 6, 7, 8, 0, 24, 23, 22, 21,
     20, 19, 18, 32, 31, 30, 29, 28, 27, 26, 25, 17, 43, 48, 49, 51, 50, 46, 47,
     45, 44, 102, 103, 104, 105, 101, 100, 99, 98, 97, 72, 73, 74, 86, 75, 76, 77,
     78, 79, 80, 85, 84, 83, 82, 81, 35, 41, 40, 42, 39, 37, 33, 36, 34, 89, 95, 94,
     96, 93, 91, 87, 90, 92, 52, 64, 63, 71, 67, 68, 61, 58, 59, 53, 56, 55, 65, 66,
     62, 70, 69, 57, 60, 54, 38, 88]