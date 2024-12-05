from core import mx
from core.lib.collections import FDict, get_enum_id_by_name
from core.lib.image import (ImageFormat, ImageFormatType,
                            get_image_format_by_type)


class MxImageFormat(mx.Disposable):
    ImageFormatType = ImageFormatType


    def __init__(self, default_format_type=ImageFormatType.JPEG2000, state : FDict|None = None):#
        super().__init__()

        state = FDict(state)

        self._mx_image_format_type = mx.StateChoice[MxImageFormat.ImageFormatType](availuator=lambda: MxImageFormat.ImageFormatType).dispose_with(self)
        self._mx_image_format_type.set(get_enum_id_by_name(MxImageFormat.ImageFormatType, state.get('image_format_type', None), default_format_type))

        self._mx_quality = mx.Number(1, config=mx.Number.Config(min=1, max=100, step=1)).dispose_with(self)
        self._mx_quality.set(state.get('quality', 100))

    def get_state(self) -> FDict:
        return FDict({  'image_format_type' : self._mx_image_format_type.get().name,
                        'quality' : self._mx_quality.get(),  })

    @property
    def mx_image_format_type(self) -> mx.IStateChoice_v[ImageFormatType]: return self._mx_image_format_type
    @property
    def mx_quality(self) -> mx.INumber_v: return self._mx_quality

    @property
    def image_format(self) -> ImageFormat: return get_image_format_by_type(self._mx_image_format_type.get())
    @property
    def quality(self) -> int: return self._mx_quality.get()
