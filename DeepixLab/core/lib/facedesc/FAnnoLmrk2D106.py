from __future__ import annotations

from ..collections import FDict
from ..image import FImage
from ..math import FAffMat2, FVec2fArray, FVec2i
from .FAnnoLmrk2D import FAnnoLmrk2D
from .FAnnoLmrk2D68 import FAnnoLmrk2D68


class FAnnoLmrk2D106(FAnnoLmrk2D):
    """Immutable class. Describes 2D 106 landmarks of the face"""

    @staticmethod
    def from_state(state : FDict|None) -> FAnnoLmrk2D68|None:
        if (anno := FAnnoLmrk2D.from_state(state)) is not None:
            return FAnnoLmrk2D106(anno.lmrks)
        return None

    def __init__(self, lmrks : FVec2fArray):
        if len(lmrks) != 106:
            raise ValueError('lmrks count must be == 106')
        super().__init__(lmrks)

    def to_2D68(self) -> FAnnoLmrk2D68:
        lmrks = self._lmrks.as_np()[ lmrks_106_to_68_mean_pairs ].reshape(68,2,2).mean(1)
        return FAnnoLmrk2D68(FVec2fArray(lmrks))

    def get_align_mat(self, coverage : float = 1.0, ux_offset = 0.0, uy_offset = 0.0, output_size = FVec2i(1,1)) -> FAffMat2:
        return self.to_2D68().get_align_mat(coverage=coverage, ux_offset=ux_offset, uy_offset=uy_offset, output_size=output_size)

    def generate_mask(self, W, H) -> FImage:
        return self.to_2D68().generate_mask(W,H)


lmrks_106_to_68_mean_pairs = [
     0,   2, # 0
     2,   4, # 1
     4,   6, # 2
     6,   8, # 3
     8,  10, # 4
    10,  12, # 5
    12,  14, # 6
    14,  16, # 7
    16,  16, # 8

    16,  18, # 9
    18,  20, # 10
    20,  22, # 11
    22,  24, # 12
    24,  26, # 13
    26,  28, # 14
    28,  30, # 15
    30,  32, # 16

    33,  33, # 17 Left eyebrow
    34,  34, # 18
    35,  35, # 19
    36,  36, # 20
    37,  37, # 21

    42,  42, # 22 Right eyebrow
    43,  43, # 23
    44,  44, # 24
    45,  45, # 25
    46,  46, # 26

    51,  52, # 27 Nose
    52,  53, # 28
    53,  54, # 29
    54,  60, # 30

    58,  59, # 31 bottom nose
    59,  60, # 32
    60,  60, # 33
    60,  61, # 34
    61,  62, # 35

    66,  66, # 36 Left eye
    67,  68, # 37
    68,  69, # 38
    70,  70, # 39
    71,  72, # 40
    72,  73, # 41

    75,  75, # 42 Right eye
    76,  77, # 43
    77,  78, # 44
    79,  79, # 45
    80,  81, # 46
    81,  82, # 47

    84,   84, # 48 Mouth
    85,   85, # 49
    86,   86, # 50
    87,   87, # 51
    88,   88, # 52
    89,   89, # 53
    90,   90, # 54
    91,   91, # 55
    92,   92, # 56
    93,   93, # 57
    94,   94, # 58
    95,   95, # 59
    96,   96, # 60
    97,   97, # 61
    98,   98, # 62
    99,   99, # 63
    100, 100, # 64
    101, 101, # 65
    102, 102, # 66
    103, 103, # 67
    ]
