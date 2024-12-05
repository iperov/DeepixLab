import math

import numpy as np

from .FImage import FImage


class Patcher:
    """
    a class to cut and merge patches of image
    """
    def __init__(self, img : FImage, PW, PH = None, sample_count=1, use_padding=False):
        if PH is None:
            PH = PW

        self._use_padding = use_padding
        if use_padding:
            PADL = self._PADL = math.ceil(PW / 2)
            PADR = self._PADR = math.floor(PW / 2)
            PADT = self._PADT = math.ceil(PH / 2)
            PADB = self._PADB = math.floor(PH / 2)

            img = img.pad(PADL, PADT, PADR, PADB)

        self._img = img.HWC()
        self._pw = PW
        self._ph = PH


        H,W,C = img.shape
        if not use_padding and (H < PH or W < PW):
            raise ValueError('Image size less than patch size.')

        x_stride = max(1, PW // 2**(sample_count-1) )
        y_stride = max(1, PH // 2**(sample_count-1) )

        self._xs_count = xs_count = int(math.ceil((W - PW)  / x_stride))+1
        self._ys_count = ys_count = int(math.ceil((H - PH)  / y_stride))+1

        self._x_stride = (W - PW) / max(1,xs_count-1)
        self._y_stride = (H - PH) / max(1,ys_count-1)

        self._weight_patch = Patcher._get_weight_patch(PW, PH)[...,None]
        self._img_weight = np.zeros( (H,W,1), np.float32 )
        self._img_out = np.zeros( (H,W,C), np.float32 )

    @property
    def patch_count(self) -> int:
        return self._ys_count*self._xs_count

    def _get_patch_slice(self, patch_id) -> slice:
        ys = patch_id // self._xs_count
        xs = patch_id % self._xs_count

        x_start = int(xs*self._x_stride)
        x_end = x_start + self._pw

        y_start = int(ys*self._y_stride)
        y_end = y_start + self._ph
        return (slice(y_start, y_end), slice(x_start, x_end))

    def get_patch(self, patch_id) -> FImage:
        return FImage.from_numpy(self._img[self._get_patch_slice(patch_id)])

    def merge_patch(self, patch_id, img : FImage):
        H,W,_ = img.shape
        if H != self._ph or W != self._pw:
            raise ValueError('img must be equal to patch size')

        s = self._get_patch_slice(patch_id)

        self._img_out[s] += img.f32().HWC()*self._weight_patch
        self._img_weight[s] += self._weight_patch

    def get_merged_image(self,) -> FImage:
        img_weight = self._img_weight.copy()
        img_weight[img_weight == 0] = 1

        # Normalize image by weights
        img = self._img_out * (1 / img_weight)

        if self._use_padding:
            img = img[self._PADT:-self._PADB, self._PADL:-self._PADR, :]

        return FImage.from_numpy(img)

    @staticmethod
    def _get_weight_patch(W, H) -> np.ndarray:
        x_coords = np.linspace(0, W-1, W) +0.5
        y_coords = np.linspace(0, H-1, H) +0.5

        x_line = np.interp(x_coords, [0, W/2, W], [0,1,0])
        y_line = np.interp(y_coords, [0, H/2, H], [0,1,0])

        return x_line[None,:]*y_line[:,None]