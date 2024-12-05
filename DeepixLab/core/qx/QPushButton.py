from typing import Tuple, overload

from .. import lx, qt
from ._constants import Align, Align_to_AlignmentFlag, Size, icon_Size_to_int
from ._helpers import q_init
from .QAbstractButton import QAbstractButton
from .QApplication import QApplication
from .QFontDB import FontDB, QFontDB
from .QIconDB import IconDB, QIconDB
from .QIconWidget import _QIconWidget
from .StyleColor import StyleColor


class QPushButton(QAbstractButton):
    def __init__(self, **kwargs):
        super().__init__(q_abstract_button=q_init('q_pushbutton', qt.QPushButton, **kwargs), **kwargs)
        q_btn = self.q_pushbutton
        
        icon = self.__icon = _QIconWidget()
        icon.setAttribute(qt.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        icon.hide()
        
        label = self.__label = qt.QLabel()
        label.setAttribute(qt.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        label.hide()
        label.setAlignment(qt.Qt.AlignmentFlag.AlignLeft)
        
        wl = qt.QHBoxLayout()
        wl.setSpacing(0)
        wl.addWidget(icon)
        wl.addWidget(label)
        
        w = qt.QWidget()
        w.setLayout(wl)
        l = qt.QVBoxLayout()
        l.setSpacing(0)
        l.addWidget(w,alignment=qt.Qt.AlignmentFlag.AlignHCenter)
        
        q_btn.setLayout(l)
        
        self.set_icon_size(Size.Default)
        
    
    @property
    def text(self) -> str: return self.__label.text()
    @property
    def q_pushbutton(self) -> qt.QPushButton: return self.q_abstract_button
    
    def _minimum_size_hint(self) -> qt.QSize:
        size = qt.QSize(0,0)
        has_text = len(self.__label.text()) != 0
        has_icon = self.__icon.get_icon() is not None
        
        if has_text:
            size = size.expandedTo(self.__label.minimumSizeHint())
        if has_icon:
            size = size.expandedTo(self.__icon.minimumSizeHint())
        if has_text and has_icon:
            size = size.grownBy(qt.QMargins(0,0,8,0))
            
        return size
    
    def _size_hint(self) -> qt.QSize:
        size = qt.QSize(0,0)
        has_text = len(self.__label.text()) != 0
        has_icon = self.__icon.get_icon() is not None
        
        if has_text:
            size = size.expandedTo(self.__label.minimumSizeHint())
        if has_icon:
            size = size.expandedTo(self.__icon.minimumSizeHint())
        if has_text and has_icon:
            size = size.grownBy(qt.QMargins(0,0,8,0))
        return size

    
    def set_font(self, font : qt.QFont | FontDB):
        if isinstance(font, FontDB):
            font = QFontDB.instance().get(font)
        self.__label.setFont(font)
        return self
    
    def set_alignment(self, align : Align):
        self.__label.setAlignment(Align_to_AlignmentFlag[align])
        return self
    
    def set_text(self, text: str | None):
        self.__label.setVisible(text is not None)
        #self.__update_layout()
        
        if (disp := getattr(self, '_QPushButton_text_disp', None)) is not None:
            disp.dispose()
        self._QAbstractButton_text_disp = QApplication.instance().mx_language.reflect(lambda lang:self.__label.setText(lx.L(text, lang))).dispose_with(self)
        return self

    @overload
    def set_icon(self, icon : qt.QIcon|None): ...
    @overload
    def set_icon(self, icon : IconDB, color : qt.QColor = StyleColor.ButtonText): ...
    def set_icon(self, *args, **kwargs):
        if len(args) == 1:
            arg0 = args[0]
            if arg0 is None:
                self.__icon.hide()
                #self.__update_layout()
            elif isinstance(arg0, qt.QIcon):
                self.__icon.set_icon(arg0)
                self.__icon.show()
                #self.__update_layout()
            else:
                args = (arg0, kwargs.get('color', StyleColor.ButtonText))
            
        if len(args) == 2:
            self.set_icon(QIconDB.instance().icon(args[0], args[1]))
            
        return self

    def set_icon_size(self, size : Tuple[int, int] | Size):
        if isinstance(size, Size):
            size = (icon_Size_to_int[size],)*2
        self.__icon.set_icon_size(qt.QSize(*size))
        return self
    
    # def __update_layout(self):
    #     #self.q_pushbutton.setLayout(self.__icon_text_layout)
    #     # if self.__text_visible and self.__icon_visible:
    #     #     self.q_pushbutton.setLayout(self.__icon_text_layout)
    #     # elif self.__text_visible:
    #     #     self.q_pushbutton.setLayout(self.__text_layout)
    #     # elif self.__icon_visible:
    #     #     self.q_pushbutton.setLayout(self.__icon_layout)
        
    #     self.update_geometry()
        
        