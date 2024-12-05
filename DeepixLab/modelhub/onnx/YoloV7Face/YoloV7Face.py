from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from core.lib import facedesc as fd
from core.lib import math as lib_math
from core.lib import onnxruntime as lib_ort
from core.lib.functools import cached_method
from core.lib.image import FImage
from core.lib.math import FRectf

from ...BaseDetector import BaseDetector


class YoloV7Face(BaseDetector):
    """
    YoloV7-s face detection model.
    ```
        Easy    Medium  Hard
        94.8 	93.1 	85.2
    ```

    arguments

     device_info    ORTDeviceInfo

        use YoloV7Face.get_available_devices()
        to determine a list of avaliable devices accepted by model

    raises
     Exception
    """
    @dataclass
    class DetectJob:
        aug_imgs : Sequence[BaseDetector._AugmentedImage]
        minimum_confidence : float
        nms_threshold : float
        min_face_size : float

    @staticmethod
    def get_available_devices() -> Sequence[lib_ort.DeviceRef]:
        return [lib_ort.get_cpu_device()] + lib_ort.get_avail_gpu_devices()

    def __init__(self, device : lib_ort.DeviceRef ):
        """raise on error"""
        super().__init__()

        if device not in YoloV7Face.get_available_devices():
            raise Exception(f'device {device} is not in available devices for YoloV7Face')

        self._sess = sess = lib_ort.InferenceSession_with_device(Path(__file__).parent / 'YoloV7Face.onnx', device)
        self._input_name = sess.get_inputs()[0].name


    def detect(self,  img : FImage,
                      resolution : int = None,
                      pad_to_resolution=False,
                      augment_pyramid=False,
                      minimum_confidence : float = 0.3,
                      nms_threshold=0.3,
                      min_face_size=8,
                      ) -> Sequence[fd.FFace]:
        """
        arguments

            img    FImage

            min_face_size(8)

            resolution(None) int    scale img by largest side(W or H) to desired size for detection.
                                    Pros: decrease detection time if image is very large, for example 4k
                                          avoid delay if image sequence has different sizes
                                    Cons: loose accuracy of detected rectangles

            augment_pyramid(False) bool     augment by pyramid. Increases faces detected (even false positives), decreases performance.
        """
        return self.detect_job(self.prepare_job(img, resolution, pad_to_resolution, augment_pyramid, minimum_confidence, nms_threshold, min_face_size))


    def prepare_job(self,   img : FImage,
                            resolution : int = None,
                            pad_to_resolution=False,
                            augment_pyramid=False,
                            minimum_confidence : float = 0.3,
                            nms_threshold=0.3,
                            min_face_size=8,) -> DetectJob:
        return self.DetectJob(  aug_imgs = self._augment(   img.bgr().u8(),#.swap_rb().f32(),
                                                            grid_size=[32,32],
                                                            resolution=resolution,
                                                            pad_to_resolution=pad_to_resolution,
                                                            augment_pyramid=augment_pyramid,
                                                            ),
                                minimum_confidence=minimum_confidence,
                                nms_threshold=nms_threshold,
                                min_face_size=min_face_size )

    def detect_job(self, job : DetectJob) -> Sequence[fd.FFace]:
        preds = []
        for aug_img in job.aug_imgs:
            pred = self._get_preds(aug_img.img.CHW()[None,...])[0]
            pred[:,0:4] *= 1/aug_img.scale
            preds.append(pred)

        preds = np.concatenate(preds, 0)

        preds = preds[ preds[...,4] >= job.minimum_confidence ]
        preds = preds[ preds[...,2:4].min(-1) >= job.min_face_size ]

        x,y,w,h,confidence = preds.T

        w_half = w/2
        h_half = h/2

        l, t, r, b = x-w_half, y-h_half, x+w_half, y+h_half
        keep = lib_math.nms(l,t,r,b, confidence, job.nms_threshold)
        l, t, r, b, confidence = l[keep], t[keep], r[keep], b[keep], confidence[keep]

        return [ fd.FFace(FRectf(l,t,r,b)).set_confidence(confidence).set_detector_id('YoloV7Face')
                  for l,t,r,b,confidence in np.stack([l, t, r, b, confidence], -1) ]

    def _get_preds(self, img : np.ndarray):
        N,C,H,W = img.shape
        preds = self._sess.run(None, {self._input_name: img})

        # YoloV7Face returns 3x [N,C*21,H,W].
        # C = [cx,cy,w,h,thres, ... ]
        # Transpose and cut first 5 channels.
        pred0, pred1, pred2 = [pred.reshape( (N,C,21,pred.shape[-2], pred.shape[-1]) ).transpose(0,1,3,4,2)[...,0:5] for pred in preds]

        pred0 = self._process_pred(pred0, W, H, anchor=[ [4,5],[6,8],[10,12] ]  ).reshape( (N, -1, 5) )
        pred1 = self._process_pred(pred1, W, H, anchor=[ [15,19],[23,30],[39,52] ]  ).reshape( (N, -1, 5) )
        pred2 = self._process_pred(pred2, W, H, anchor=[ [72,97],[123,164],[209,297] ]  ).reshape( (N, -1, 5) )

        return np.concatenate( [pred0, pred1, pred2], 1 )[...,:5]

    def _process_pred(self, pred, img_w, img_h, anchor):
        pred_h = pred.shape[-3]
        pred_w = pred.shape[-2]
        anchor = np.float32(anchor)[None,:,None,None,:]

        grid = self._get_grid(pred_w, pred_h)

        stride = (img_w // pred_w, img_h // pred_h)

        pred = YoloV7Face._sigmoid(pred)

        pred[..., 0:2] = (pred[..., 0:2]*2 - 0.5 + grid) * stride
        pred[..., 2:4] = (pred[..., 2:4]*2)**2 * anchor
        return pred

    @cached_method
    def _get_grid(self, W, H) -> np.ndarray:
        _xv, _yv,  = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), )
        return np.stack((_xv, _yv), 2).reshape((1, 1, H, W, 2))

    @staticmethod
    def _sigmoid(x : np.ndarray) -> np.ndarray:
        positives = x >= 0
        negatives = ~positives

        exp_x_neg = np.exp(x[negatives])

        y = x.copy()
        y[positives] = 1 / (1 + np.exp(-x[positives]))
        y[negatives] = exp_x_neg / (1 + exp_x_neg)

        return y

