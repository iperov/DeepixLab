from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

class QFusionPalette:
    # based on https://github.com/qt/qtbase/tree/dev/src/widgets/styles/qfusionstyle_p_p.h
    
    def __init__(self, pal : QPalette):
        self._pal = pal
        
    def button_color(self) -> QColor:
        button_color = self._pal.color(QPalette.ColorRole.Button)
        
        val = (button_color.red()*11+button_color.green()*16+button_color.blue()*5)//32

        button_color = button_color.lighter(100 + max(1, (180 - val)//6)  )
        button_color.setHsv(button_color.hue(), int(button_color.saturation() * 0.75), int(button_color.value()))
        return button_color

    def outline_color(self) -> QColor:
        return self._pal.color(QPalette.ColorRole.Window).darker(140)
        
    def highlight(self) -> QColor:
        #if (isMacSystemPalette(pal))
        #    return QColor(60, 140, 230);
        return self._pal.color(QPalette.ColorRole.Highlight)
    
    def innerContrastLine(self) -> QColor:
        return QColor(255, 255, 255, 30)