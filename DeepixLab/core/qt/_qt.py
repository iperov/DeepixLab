from typing import Any, Callable, TypeVar

import numpy as np
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from ..lib.image import FImage
from ..lib.math import FAffMat2, FBBox, FVec2f, FVec2, FLine2f

T = TypeVar('T')

_C_to_Format = {
        1: QImage.Format.Format_Grayscale8,
        3: QImage.Format.Format_BGR888,
        4: QImage.Format.Format_ARGB32
    }

_Format_to_C = { v : k for k,v in _C_to_Format.items() }


def wrap(q_object : T, func_name, wrapper : Callable[ [T, Callable], Any ]) -> Callable:
    """
    wrapper(q_object, super, *args, **kwargs)

    returns super
    """
    super = getattr(q_object, func_name, None)
    if super is None:
        raise ValueError(f'No method with name {func_name} found in q_object')

    wrapped = lambda *args, **kwargs: wrapper(q_object, super, *args, **kwargs)
    setattr(q_object, func_name, wrapped)
    return super

def unwrap(q_object, func_name, super):
    setattr(q_object, func_name, super)

def QImage_to_FImage(img : QImage) -> FImage:
    return FImage.from_numpy(QImage_to_np(img, QImage.Format.Format_BGR888))

def QImage_to_np(img : QImage, fmt : QImage.Format, copy=True) -> np.ndarray:
    C = _Format_to_C.get(fmt, None)
    if C is None:
        raise ValueError(f'Unsupported format {fmt}. Avail: {_Format_to_C.keys()}')

    img = img.convertToFormat(fmt)
    arr = np.frombuffer(img.constBits(), np.uint8).reshape(img.height(), img.width(), C)
    if copy:
        arr = arr.copy()
    return arr

def QTransform_from_mat(mat : FAffMat2) -> QTransform:
    mat = mat.as_np()
    return QTransform( mat[0,0], mat[1,0], 0,
                       mat[0,1], mat[1,1], 0,
                       mat[0,2], mat[1,2], 1)


def QImage_from_FImage(img : FImage) -> QImage:
    """"""
    return QImage_from_np(img.HWC())
    
def QImage_from_np(img : np.ndarray) -> QImage:
    """
    constructs QImage from image np.ndarray (HW HWC) uint8/float32

    img must not be changed, or provide a copy.
    """
    if img.dtype == np.float32:
        img = img * 255.0
        img = np.clip(img, 0, 255, out=img).astype(np.uint8)

    if img.dtype != np.uint8:
        raise ValueError('image.dtype must be np.uint8/np.float32')

    if len(img.shape) == 2:
        img = img[..., None]

    if len(img.shape) != 3:
        raise ValueError('img shape must be HW/HWC')

    H,W,C = img.shape

    format = _C_to_Format.get(C, None)
    if format is None:
        raise ValueError(f'Unsupported number of channels {C}')

    img = np.ascontiguousarray(img)

    q_image = QImage(img, W, H, W*C, format)
    q_image._np_img = img # save in order not to be garbage collected


    return q_image

def QImage_colorized(image : QImage, color : QColor ) -> QImage:
    img = QImage(image)
    qp = QPainter(img)
    qp.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    qp.fillRect(img.rect(), color)
    qp.end()
    return img

def QPixmap_from_np(img : np.ndarray) -> QPixmap: 
    """
    constructs QPixmap from image np.ndarray (HW HWC) uint8/float32

    img must not be changed, or provide a copy.
    """
    return QPixmap(QImage_from_np(img))


def QPixmap_from_FImage(img : FImage) -> QPixmap: 
    """"""
    return QPixmap(QImage_from_FImage(img))

def QPixmap_colorized(pixmap : QPixmap, color : QColor ) -> QPixmap:
    img = QPixmap(pixmap)
    qp = QPainter(img)
    qp.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    qp.fillRect(img.rect(), color)
    qp.end()
    return img



def QPoint_to_np(pt : QPoint) -> np.ndarray: return np.int32([pt.x(), pt.y()])

def QPoint_to_FVec2f(pt : QPoint) -> FVec2f: return FVec2f(pt.x(), pt.y())
def QPointF_to_FVec2f(pt : QPointF) -> FVec2f: return FVec2f(pt.x(), pt.y())

def QPoint_from_np(pt : np.ndarray) -> QPoint: return QPoint(*pt.astype(np.int32, copy=False))

def QPoint_from_FVec2(pt : FVec2) -> QPoint: return QPoint(int(pt.x), int(pt.y))

def QLine_from_FLine2f(line : FLine2f) -> QLine: return QLine(int(line.p0.x), int(line.p0.y), int(line.p1.x), int(line.p1.y),)

def QPointF_to_np(pt : QPoint) -> np.ndarray: return np.float32([pt.x(), pt.y()])

def QPointF_from_np(pt : np.ndarray) -> QPoint:return QPointF(*pt.astype(np.float32, copy=False))

def QRect_from_FBBox(r : FBBox) -> QRect: return QRect(int(r.x), int(r.y), int(r.width), int(r.height))
def QRectF_from_FBBox(r : FBBox) -> QRectF: return QRectF(r.x, r.y, r.width, r.height)
def QRect_to_FBBox(r : QRect) -> FBBox: return FBBox(r.x(), r.y(), r.width(), r.height())
def QRectF_to_FBBox(r : QRectF) -> FBBox: return FBBox(r.x(), r.y(), r.width(), r.height())



def QRect_fit_in(rect : QRect, rect_in : QRect) -> QRect:
    """fit QRect in QRect keeping aspect ratio"""
    w, h = rect_in.width(), rect_in.height()
    if w == 0 or h==0:
        return QRect(0,0,0,0)

    self_ap, rect_ap = rect.width() / rect.height(), w / h

    if self_ap < rect_ap:
        w_fit = w * (self_ap / rect_ap  )
        return QRect(rect_in.left() + (w-w_fit)//2, rect_in.top(), w_fit, h )
    elif self_ap > rect_ap:
        h_fit = h * (rect_ap / self_ap )
        return QRect(rect_in.left(), rect_in.top() + (h-h_fit)//2, w, h_fit )
    else:
        return QRect(rect_in)

def QRect_center_in(rect : QRect, rect_in : QRect) -> QRect:
    """center QRect in QRect"""
    l = rect_in.width() / 2.0
    t = rect_in.height() / 2.0
    w = rect.width()
    h = rect.height()
    return QRect(int(l - w/2), int(t - h/2), w, h)

def QSize_to_np(size : QSize) -> np.ndarray: return np.int32([size.width(), size.height()])
def QSize_to_FVec2(size : QSize) -> FVec2f: return FVec2f(size.width(), size.height())
def QSize_from_FVec2(pt : FVec2f) -> QSize: return QSize(int(pt.x), int(pt.y))
def QSizeF_from_FVec2(pt : FVec2f) -> QSizeF: return QSizeF(pt.x, pt.y)


def QKeyCombination_to_string(key_comb : QKeyCombination) -> str:
    return QKeySequence(key_comb).toString()