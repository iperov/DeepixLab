from __future__ import annotations

from core import mx, qx

from .MxImageFormat import MxImageFormat


class QxImageFormat(qx.QVBox):
    def __init__(self, ff : MxImageFormat):
        super().__init__()
        self._ff = ff

        self._quality_vbox = qx.QVBox()

        self.add(qx.QComboBoxMxStateChoice(ff.mx_image_format_type))
        self.add(self._quality_vbox, align=qx.Align.CenterF)

        ff.mx_image_format_type.reflect(lambda img_fmt_type, enter, bag=mx.Disposable().dispose_with(self):
                                             self._ref_export_file_format(img_fmt_type, enter, bag)).dispose_with(self)

    def _ref_export_file_format(self, img_fmt_type : MxImageFormat.FileFormatType, enter : bool, bag : mx.Disposable):
        if enter:
            if img_fmt_type in [MxImageFormat.ImageFormatType.JPEG,
                                MxImageFormat.ImageFormatType.JPEG2000,
                                MxImageFormat.ImageFormatType.JPEG2000_16,
                                MxImageFormat.ImageFormatType.WEBP, ]:

                self._quality_vbox.add(
                        qx.QHBox().dispose_with(bag)
                            .add( qx.QLabel().set_text('@(Quality)'))
                            .add( qx.QDoubleSpinBoxMxNumber(self._ff.mx_quality)), align=qx.Align.LeftF)
        else:
            bag.dispose_items()