from enum import IntEnum, auto

from .. import qt
from ..lib import os as lib_os

ArrowType = qt.Qt.ArrowType
LayoutDirection = qt.Qt.LayoutDirection
Orientation = qt.Qt.Orientation
ProcessPriority = lib_os.ProcessPriority
TextInteractionFlag = qt.Qt.TextInteractionFlag
WindowType = qt.Qt.WindowType

class Align(IntEnum):
    LeftE        = auto()
    LeftF        = auto()
    RightE       = auto()
    RightF       = auto()
    TopE         = auto()
    TopF         = auto()
    BottomE      = auto()
    BottomF      = auto()

    TopLeft     = auto()
    TopRight    = auto()
    BottomRight = auto()
    BottomLeft  = auto()

    CenterE      = auto()
    CenterF      = auto()
    CenterH      = auto()
    CenterV      = auto()


Align_to_AlignmentFlag = {
    Align.LeftE        : qt.Qt.AlignmentFlag.AlignLeft,
    Align.LeftF       : qt.Qt.AlignmentFlag.AlignLeft | qt.Qt.AlignmentFlag.AlignVCenter,

    Align.RightE       : qt.Qt.AlignmentFlag.AlignRight,
    Align.RightF       : qt.Qt.AlignmentFlag.AlignRight | qt.Qt.AlignmentFlag.AlignVCenter,

    Align.TopE         : qt.Qt.AlignmentFlag.AlignTop,
    Align.TopF         : qt.Qt.AlignmentFlag.AlignTop | qt.Qt.AlignmentFlag.AlignHCenter,

    Align.BottomE      : qt.Qt.AlignmentFlag.AlignBottom,
    Align.BottomF      : qt.Qt.AlignmentFlag.AlignBottom | qt.Qt.AlignmentFlag.AlignHCenter,

    Align.TopLeft     : qt.Qt.AlignmentFlag.AlignLeft | qt.Qt.AlignmentFlag.AlignTop,
    Align.TopRight    : qt.Qt.AlignmentFlag.AlignRight | qt.Qt.AlignmentFlag.AlignTop,
    Align.BottomRight : qt.Qt.AlignmentFlag.AlignRight | qt.Qt.AlignmentFlag.AlignBottom,
    Align.BottomLeft  : qt.Qt.AlignmentFlag.AlignLeft | qt.Qt.AlignmentFlag.AlignBottom,

    Align.CenterE      : qt.Qt.AlignmentFlag(0),
    Align.CenterF      : qt.Qt.AlignmentFlag.AlignCenter,
    Align.CenterH      : qt.Qt.AlignmentFlag.AlignHCenter,
    Align.CenterV      : qt.Qt.AlignmentFlag.AlignVCenter,
}

class Size(IntEnum):
    XXL = auto()
    XL = auto()
    L = auto()
    M = auto()
    S = auto()
    Default = M

icon_Size_to_int = {
    Size.XXL : 48,
    Size.XL : 32,
    Size.L : 24,
    Size.M : 16,
    Size.S : 10
}
