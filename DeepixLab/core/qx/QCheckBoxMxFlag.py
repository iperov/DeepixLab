from .. import mx
from .QCheckBox import QCheckBox


class QCheckBoxMxFlag(QCheckBox):
    def __init__(self, flag : mx.IFlag_v, **kwargs):
        super().__init__(**kwargs)
        self._flag = flag

        self._conn = self.mx_toggled.listen(lambda checked: flag.set(checked))

        flag.reflect(self._ref_flag).dispose_with(self)

    def _ref_flag(self, flag):
        with self._conn.disabled_scope():
            self.set_checked(flag)

