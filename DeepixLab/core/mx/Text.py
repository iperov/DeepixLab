from .Property import IProperty_v, Property


class IText_v(IProperty_v[str]): ...

class Text(Property[str], IText_v): ...