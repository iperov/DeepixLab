from ... import qt
from enum import Enum, auto


class QDarkFusionPalette:
    
    class Direction(Enum):
        TopDown = auto()
    
    # based on  https://github.com/qt/qtbase/tree/dev/src/widgets/styles/qfusionstyle_p_p.h
    #           https://github.com/qt/qtbase/tree/dev/src/widgets/styles/qfusionstyle.cpp
    
    def __init__(self, pal : qt.QPalette):
        self._pal = pal
    
    def gradient(self, rect : qt.QRect, base_color : qt.QColor, direction : Direction = Direction.TopDown) -> qt.QLinearGradient:
        x = rect.center().x()
        y = rect.center().y()
        
        if direction == QDarkFusionPalette.Direction.TopDown:
            gradient = qt.QLinearGradient(x, rect.top(), x, rect.bottom())
            
        
        gradient.setColorAt(0, base_color.lighter(124))
        gradient.setColorAt(1, base_color.lighter(102))
        return gradient
        
#         static QLinearGradient qt_fusion_gradient(const QRect &rect, const QBrush &baseColor, Direction direction = TopDown)

#     case FromLeft:
#         gradient = QLinearGradient(rect.left(), y, rect.right(), y);
#         break;
#     case FromRight:
#         gradient = QLinearGradient(rect.right(), y, rect.left(), y);
#         break;
#     case BottomUp:
#         gradient = QLinearGradient(x, rect.bottom(), x, rect.top());
#         break;
#     if (baseColor.gradient())
#         gradient.setStops(baseColor.gradient()->stops());
#     else {
#     return gradient;
# }

    def button_color(self) -> qt.QColor:
        button_color = self._pal.color(qt.QPalette.ColorRole.Button)
        
        val = (button_color.red()*11+button_color.green()*16+button_color.blue()*5)//32

        button_color = button_color.lighter(100 + max(1, (180 - val)//6)  )
        button_color.setHsv(button_color.hue(), int(button_color.saturation() * 0.75), int(button_color.value()))
        return button_color

    def outline_color(self) -> qt.QColor:
        return self._pal.color(qt.QPalette.ColorRole.Window).darker(140)
        
    def highlight(self) -> qt.QColor:
        #if (isMacSystemPalette(pal))
        #    return QColor(60, 140, 230);
        return self._pal.color(qt.QPalette.ColorRole.Highlight)
    
    def innerContrastLine(self) -> qt.QColor:
        return qt.QColor(255, 255, 255, 30)