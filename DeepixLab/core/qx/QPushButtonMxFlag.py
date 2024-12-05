from .. import mx
from .QPushButton import QPushButton


class QPushButtonMxFlag(QPushButton):
    def __init__(self, flag : mx.IFlag_v):
        super().__init__()
        self._flag = flag
        self.set_checkable(True)

        self._conn = self.mx_toggled.listen(lambda checked: flag.set(checked))

        flag.reflect(self._ref_flag).dispose_with(self)

    def _ref_flag(self, flag):
        with self._conn.disabled_scope():
            self.set_checked(flag)
