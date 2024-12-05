from dataclasses import dataclass
from typing import List, Sequence, Tuple

from core.lib.image import FImage


class BaseDetector:
    def __init__(self):
        """base class for detectors. Provides common functionality"""

    @dataclass
    class _AugmentedImage:
        img : FImage
        scale : float

    def _augment(self,  img : FImage,
                        grid_size : Tuple[int, int],
                        resolution : int = None,
                        pad_to_resolution : bool = False,
                        augment_pyramid : bool=False,
                    ) -> Sequence[_AugmentedImage]:
        """augment image for detection"""

        base_img = img
        H, W, _ = img.shape

        base_scale = 1.0
        if resolution is not None:
            WH = max(W,H)
            if WH != resolution:
                base_scale = resolution / WH


        if base_scale != 1.0:
            base_img = base_img.resize(int(W*base_scale), int(H*base_scale))

            if pad_to_resolution:
                base_img = base_img.pad(0,0, max(0, resolution - base_img.width), max(0, resolution - base_img.height) )

        out : List[FImage] = [ self._AugmentedImage(img = base_img.pad_to_next_divisor(grid_size[0], grid_size[1]),
                                                    scale = base_scale) ]

        if augment_pyramid:
            base_H, base_W, _ = base_img.shape

            for n in range(1, 5):
                scale = (0.66**n)

                aug_img = self._AugmentedImage( img=base_img.resize(int(base_W*scale), int(base_H*scale))
                                                            .pad_to_next_divisor(grid_size[0], grid_size[1]),
                                                scale = base_scale*scale )

                out.append(aug_img)

        return out