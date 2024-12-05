from __future__ import annotations

from collections import deque
from typing import (TYPE_CHECKING, Callable, Deque, Dict, Sequence, Type, Self, TypeVar)

from .. import mx, qt
from ._helpers import q_init
from .QFuncWrap import QFuncWrap
from .QSettings import QSettings

if TYPE_CHECKING:
    from .QApplication import QApplication

T = TypeVar('T')

class QObject(mx.Disposable):

    def __init__(self, **kwargs):
        """
        kwargs:

            wrap_mode(False)    if True we don't touch wrapped qt object on dispose,
                                no hide, no delete later, etc

        """
        if QObject._QApplication is None:
            raise Exception('QApplication must be instantiated first.')

        super().__init__()
        self.__q_object = q_object = q_init('q_object', qt.QObject, **kwargs)
        self.__wrap_mode = kwargs.get('wrap_mode', False)

        #self.__ui_bag = mx.Disposable()

        # save wrapper reference in QObject
        setattr(self.__q_object, '__owner', self)

        self.__object_name = self.__class__.__qualname__
        self.__object_name_id = 0
        self.__parent : QObject = None
        self.__childs : Deque[QObject] = deque()
        self.__obj_name_counter : Dict[str, int] = {}
        self.__obj_name_count : Dict[str, int] = {}

        # Visible to parent listeners
        self.__vtp_listeners = set()

        # Flag indicates that object should be registered in QApp state operation on _visible_to_parent_event
        self.__is_operate_settings = False

        if self._visible_to_parent_event.__func__ != QObject._visible_to_parent_event or \
           self._hiding_from_parent_event.__func__ != QObject._hiding_from_parent_event:
            # _visible_to_parent_event-related methods are overriden, subscribe self
            self.__vtp_listeners.add(self)

        self.__eventFilter_wrap = QFuncWrap(q_object, 'eventFilter', lambda *args, **kwargs: self._event_filter(*args, **kwargs)).dispose_with(self)

        self.__event_wrap = QFuncWrap(q_object, 'event', lambda *args, **kwargs: self._event(*args, **kwargs)).dispose_with(self)

        self.__mx_settings = mx.State[QSettings]().dispose_with(self)
    
    
    def dispose_items(self):
        raise Exception('You must not to call .dispose_items() on q_objects')
    
    # @property
    # def _ui_bag(self):
    #     """class local property

    #     mx.Disposable which is disposed first before childs
    #     """
    #     return self.__ui_bag

    @property
    def _mx_settings(self) -> mx.IState_rv[QSettings]:
        """
        class local property

        Appears every time when object's settings is available/unavailable due to:

        - loaded on visible to QApplication, the state is loaded by object's tree_name

        - unloaded when object's breaks visibility to QApplication

        - unloaded/loaded when user requested "reset UI"
        """
        if self in self.__vtp_listeners:
            if not self.__is_operate_settings:
                self.__is_operate_settings = True

                if isinstance(app := self.top_parent, self._QApplication):
                    app._obj_settings_subscribe(self)
        else:
            self.__is_operate_settings = True

            # add to __vtp_listeners
            parent = self
            while parent is not None:
                #print('added to', self, parent, )
                parent.__vtp_listeners.add(self)

                if not (parent is self):
                    self._visible_to_parent_event(parent)

                parent = parent.__parent

        return self.__mx_settings

    def inline(self, func : Callable[ [Self], None ]) -> Self:
        """syntax sugar"""
        func(self)
        return self

    def _is_wrap_mode(self) -> bool:
        return self.__wrap_mode

    def __dispose__(self):
        """Dispose this object. All childs will be disposed. This object will be removed from parent."""
        self._dispose_items()

        if not self._is_wrap_mode():
            self.set_parent(None)
            self.__q_object.deleteLater()

        self.__q_object.__owner = None
        self.__q_object = None

        super().__dispose__()

    @property
    def q_object(self) -> qt.QObject: return self.__q_object
    @property
    def childs(self) -> Sequence[QObject]: return reversed(tuple(self.__childs))
    @property
    def parent(self) -> QObject|None: return self.__parent
    @property
    def object_name(self) -> str:
        """object name. Not unique relative to parent. Example: XLabel"""
        return self.__object_name
    @property
    def object_name_id(self) -> str:
        """object name id relative to parent. Example: 0"""
        return self.__object_name_id
    @property
    def name(self) -> str:
        """unique name relative to parent (if exists). Example: XLabel:0"""
        return f'{self.__object_name}:{self.__object_name_id}'
    @property
    def tree_name(self,) -> str:
        """Unique tree name up to current top parent (if exists). Example: QWindow:0/XLabel:0"""
        s = deque()
        parent = self
        while parent is not None:
            s.appendleft(parent.name )
            parent = parent.__parent
        return '/'.join(s)

    @property
    def top_parent(self) -> QObject|None:
        parent = self
        while parent is not None:
            top_parent = parent
            parent = parent.__parent
        return top_parent

    def get_top_parent_by_class(self, cls_ : Type[T]) -> T|None:
        parent = self
        top_parent = None
        while parent is not None:
            if isinstance(parent, cls_):
                top_parent = parent
            parent = parent.__parent
        return top_parent

    def get_first_parent_by_class(self, cls_ : Type[T]) -> T|None:
        parent = self
        while parent is not None:
            if isinstance(parent, cls_):
                return parent
            parent = parent.__parent
        return None

    def set_object_name(self, name : str|None):
        if self.__parent is not None:
            raise Exception('object_name must be set before parent')
        self.__object_name = f"{self.__class__.__qualname__}{ ('_'+name) if name is not None else ''}"
        return self

    def set_parent(self, new_parent : QObject|None):
        if self._is_wrap_mode():
            raise Exception('set_parent is not allowed in wrap_mode')

        if self.__parent != new_parent:
            if self.__parent is not None:
                self.__parent._child_remove_event(self)
                self.__q_object.setParent(None)

            self.__parent = new_parent

            if new_parent is not None:
                new_parent._child_added_event(self)

        return self

    def _visible_to_parent_event(self, parent : QObject):
        #print('_visible_to_parent_event', self, parent)
        """inheritable at first. Event appears when this object became visible to parent or far parent via parent-child traverse."""
        if isinstance(app := parent, self._QApplication):
            if self.__is_operate_settings:
                app._obj_settings_subscribe(self)

    def _hiding_from_parent_event(self, parent : QObject):
        #print('_hiding_from_parent_event', self, parent)
        """inheritable at last. Event appears when this object about to become invisible to parent or far parent via parent-child traverse."""
        if isinstance(app := parent, self._QApplication):
            if self.__is_operate_settings:
                app._obj_settings_unsubscribe(self)

    def _event_filter(self, object : qt.QObject, ev : qt.QEvent) -> bool:
        """inheritable. Return True to eat the event"""
        return self.__eventFilter_wrap.get_super()(object, ev)

    def _event(self, ev : qt.QEvent) -> bool:
        """inheritable"""
        return self.__event_wrap.get_super()(ev)

    def _child_added_event(self, child : QObject):
        """inheritable at first. Called when child QObject has been added"""
        obj_name = child.object_name

        obj_name_count = self.__obj_name_count
        obj_name_count[obj_name] = obj_name_count.get(obj_name, 0) + 1

        obj_name_counter = self.__obj_name_counter
        count = child.__object_name_id = obj_name_counter.get(obj_name, 0)
        obj_name_counter[obj_name] = count + 1

        self.__childs.appendleft(child)

        child.dispose_with(self)

        child_vtp_listeners = tuple(child.__vtp_listeners)

        parent = self
        while parent is not None:
            #print('_child_added_event, loop parent:', parent)
            parent.__vtp_listeners.update(child_vtp_listeners)
            #print('start for obj in child.__vtp_listeners', child, parent)
            for obj in child_vtp_listeners:
                obj._visible_to_parent_event(parent)

            parent = parent.__parent

        # self.__mx_child_added.emit(child)

    def _child_remove_event(self, child : QObject):
        """inheritable at last. Called when child QObject will be removed from childs."""
        # self.__mx_child_remove.emit(child, reverse=True)

        child_vtp_listeners = tuple(child.__vtp_listeners)

        parent = self
        while parent is not None:
            for obj in child_vtp_listeners:
                obj._hiding_from_parent_event(parent)
            parent.__vtp_listeners.difference_update(child_vtp_listeners)
            parent = parent.__parent

        child.undispose_with(self)
        self.__childs.remove(child)

        obj_name = child.object_name
        obj_name_count = self.__obj_name_count
        new_count = obj_name_count[obj_name] = obj_name_count[obj_name] - 1
        if new_count == 0:
            obj_name_counter = self.__obj_name_counter
            obj_name_counter[obj_name] = 0
        child.__object_name_id = 0

    def __repr__(self): return self.__str__()
    def __str__(self):  return f'{super().__str__()}[{self.tree_name}][Childs:{len(self.__childs)}]'#

    _QApplication : QApplication = None
