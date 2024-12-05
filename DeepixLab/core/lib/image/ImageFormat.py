from enum import StrEnum
from typing import Sequence


class ImageFormat:
    """
    describes image file format
    """
    def __init__(self, suffixes : Sequence[str], bits_per_ch : int, desc : str):
        super().__init__()
        self._suffixes = suffixes
        self._bits_per_ch = bits_per_ch
        self._desc = desc

    @property
    def suffixes(self) -> Sequence[str]:
        """all suffixes supported by format, example `['.jpg','.jpeg']`"""
        return self._suffixes
    @property
    def suffix(self) -> str:
        """main format suffix example `'.jpg'`"""
        return self._suffixes[0]
    @property
    def bits_per_ch(self) -> int:
        return self._bits_per_ch
    @property
    def desc(self) -> str: return self._desc

    def __hash__(self) -> int: return (self._suffixes, self._bits_per_ch)
    def __eq__(self, other) -> bool:
        if self is other:
            return True
        if isinstance(other, ImageFormat):
            return self._suffixes == other._suffixes and self._bits_per_ch == other._bits_per_ch
        return False

    def __str__(self):
        return self._desc

FORMAT_JPEG        = ImageFormat(suffixes=['.jpg','.jpe','.jpeg'], bits_per_ch=8, desc='JPEG')
FORMAT_JPEG2000    = ImageFormat(suffixes=['.jp2'], bits_per_ch=8, desc='JPEG2000')
FORMAT_JPEG2000_16 = ImageFormat(suffixes=['.jp2'], bits_per_ch=16, desc='JPEG2000-16bit')
FORMAT_PNG         = ImageFormat(suffixes=['.png'], bits_per_ch=8, desc='PNG')
FORMAT_PNG_16      = ImageFormat(suffixes=['.png'], bits_per_ch=16, desc='PNG-16bit')
FORMAT_TIFF_16     = ImageFormat(suffixes=['.tif','.tiff'], bits_per_ch=16, desc='TIFF-16bit')
FORMAT_WEBP        = ImageFormat(suffixes=['.webp'], bits_per_ch=8, desc='WEBP')
FORMAT_RGB8        = ImageFormat(suffixes=['.rgb8'], bits_per_ch=8, desc='RGB-8bit raw')

class ImageFormatType(StrEnum):
    """supported image format types """

    JPEG        = FORMAT_JPEG.desc
    JPEG2000    = FORMAT_JPEG2000.desc
    JPEG2000_16 = FORMAT_JPEG2000_16.desc
    PNG         = FORMAT_PNG.desc
    PNG_16      = FORMAT_PNG_16.desc
    TIFF_16     = FORMAT_TIFF_16.desc
    WEBP        = FORMAT_WEBP.desc
    RGB8        = FORMAT_RGB8.desc



_fmt_by_type = {ImageFormatType.JPEG : FORMAT_JPEG,
                ImageFormatType.JPEG2000 : FORMAT_JPEG2000,
                ImageFormatType.JPEG2000_16 : FORMAT_JPEG2000_16,
                ImageFormatType.PNG : FORMAT_PNG,
                ImageFormatType.PNG_16 : FORMAT_PNG_16,
                ImageFormatType.TIFF_16 : FORMAT_TIFF_16,
                ImageFormatType.WEBP : FORMAT_WEBP,
                ImageFormatType.RGB8 : FORMAT_RGB8,
                }

ImageFormatSuffixes = sum([fmt.suffixes for fmt in _fmt_by_type.values()], [])
ImageFormatSuffixes = tuple(sorted(tuple(set(ImageFormatSuffixes))))

def get_image_format_by_type(t : ImageFormatType) -> ImageFormat:
    return _fmt_by_type[t]
