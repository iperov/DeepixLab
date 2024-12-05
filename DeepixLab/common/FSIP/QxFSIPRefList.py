from common.List import QxList
from core import qx

from .MxFSIPRef import MxFSIPRef
from .MxFSIPRefList import MxFSIPRefList
from .QxFSIPRef import QxFSIPRef


class QxFSIPRefList(QxList[MxFSIPRef]):

    def __init__(self, list : MxFSIPRefList):
        super().__init__(list)
        self._list = list

    def _on_add_item(self):
        self._list.append_new()

    def _on_build_value(self, index, value : MxFSIPRef, value_hbox : qx.QHBox):
        value_hbox.add(QxFSIPRef(value))
