from core import mx, qx

from .MxList import MxList


class QxList[T](qx.QVBox):
    def __init__(self, list : MxList[T]):
        """base class for MxList items"""
        super().__init__()
        self.__list = list

        self.__values_vbox = qx.QVBox()
        self.__values_bag = mx.Disposable().dispose_with(self)

        (self
            .add(self.__values_vbox)
            .add(qx.QPushButton().set_icon(qx.IconDB.add).set_tooltip('@(Add_item)')
                    .inline(lambda btn: btn.mx_clicked.listen(lambda: self._on_add_item() ))))

        list.mx_added.listen(self.__on_item_added).dispose_with(self)
        list.mx_remove.listen(self.__on_item_remove).dispose_with(self)

        self.__rebuild_values()


    def _on_add_item(self):
        # Add item requested
        raise NotImplementedError()

    def _on_build_value(self, index, value : T, value_hbox : qx.QHBox):
        raise NotImplementedError()

    def __build_value(self, index, value : T):
        value_button = (qx.QPushButton().h_compact().set_icon(qx.IconDB.remove).set_tooltip('@(Remove_item)')
                                .inline(lambda btn: btn.mx_clicked.listen(lambda: self.__list.pop( self.__value_buttons.index(btn) ))))

        value_holder = qx.QHBox().dispose_with(self.__values_bag).add(value_button).add(value_hbox := qx.QHBox())
        self._on_build_value(index, value, value_hbox)

        self.__value_holders.insert(index, value_holder)
        self.__value_buttons.insert(index, value_button)
        self.__values_vbox.insert(index, value_holder)

    def __rebuild_values(self):
        self.__values_bag.dispose_items()
        self.__value_holders = []
        self.__value_buttons = []
        for index, value in enumerate(self.__list.values):
            self.__build_value(index, value)

    def __on_item_added(self, index, item : T):
        self.__build_value(index, item)

    def __on_item_remove(self, index, item : T):
        self.__value_buttons.pop(index)
        holder = self.__value_holders.pop(index)
        holder.dispose()