from ... import qt
from ...qt import QStyle
from .._constants import Size, icon_Size_to_int
from ..StyleColor import StyleColor

class QDarkFusionStyle(qt.QProxyStyle):
    def __init__(self):
        super().__init__('Fusion')
        self._widget_filter = QWidgetFilter()
        
        StyleColor.Base = qt.QColor(25, 25, 25)
        
        StyleColor.AlternateBase = qt.QColor(56, 56, 56)
        StyleColor.NoRole = qt.QColor(56, 56, 56)
        
        StyleColor.Shadow = qt.QColor(44, 44, 48)
        StyleColor.Dark = qt.QColor(56, 56, 60)
        StyleColor.Mid = qt.QColor(68, 68, 72)
        StyleColor.Midlight = qt.QColor(80, 80, 84)
        StyleColor.Light = qt.QColor(92, 92, 96)
        
        StyleColor.Window = StyleColor.Dark
        StyleColor.Button = StyleColor.Dark
        
        StyleColor.Text = qt.QColor(192, 192, 192)
        StyleColor.TextDisabled = qt.QColor(128, 128, 128)
        StyleColor.BrightText = qt.QColor(255, 0, 0)
        StyleColor.ButtonText = qt.QColor(192, 192, 192)
        StyleColor.ButtonTextDisabled = qt.QColor(128, 128, 128)
        StyleColor.PlaceholderText = qt.QColor(169, 169, 169)
        StyleColor.WindowText = qt.QColor(192, 192, 192)
        StyleColor.WindowTextDisabled = qt.QColor(128, 128, 128)
        
        StyleColor.Highlight = qt.QColor(42, 130, 218)
        StyleColor.HighlightedText = qt.QColor(0, 0, 0)
        
        StyleColor.Link = qt.QColor(192, 192, 255)
        StyleColor.LinkVisited = qt.QColor(192, 192, 255)
        
        StyleColor.ToolTipBase = qt.QColor(200, 200, 200)
        StyleColor.ToolTipText = qt.QColor(0, 0, 0)
        
        StyleColor.SelectColor = qt.QColor(60, 148, 237)
        
    def polish(self, obj : qt.QObject):
        if isinstance(obj, qt.QWidget):
            obj.installEventFilter(self._widget_filter)
            obj.setFocusPolicy(qt.Qt.FocusPolicy.ClickFocus)
            
        if isinstance(obj, qt.QAbstractButton):
            obj.setFocusPolicy(qt.Qt.FocusPolicy.NoFocus)
        
        if isinstance(obj, qt.QPalette):
            pal = obj
            pal.setColor(qt.QPalette.ColorRole.Base, StyleColor.Base)
            pal.setColor(qt.QPalette.ColorRole.AlternateBase, StyleColor.AlternateBase)
            pal.setColor(qt.QPalette.ColorRole.NoRole, StyleColor.NoRole)
            
            pal.setColor(qt.QPalette.ColorRole.Shadow, StyleColor.Shadow)
            pal.setColor(qt.QPalette.ColorRole.Dark, StyleColor.Dark)
            pal.setColor(qt.QPalette.ColorRole.Mid, StyleColor.Mid)
            pal.setColor(qt.QPalette.ColorRole.Midlight, StyleColor.Midlight)
            pal.setColor(qt.QPalette.ColorRole.Light, StyleColor.Light)
            
            pal.setColor(qt.QPalette.ColorRole.Window, StyleColor.Window)
            pal.setColor(qt.QPalette.ColorRole.Button, StyleColor.Button)
            
            pal.setColor(qt.QPalette.ColorRole.Text, StyleColor.Text)
            pal.setColor(qt.QPalette.ColorRole.BrightText, StyleColor.BrightText)
            pal.setColor(qt.QPalette.ColorRole.ButtonText, StyleColor.ButtonText)
            pal.setColor(qt.QPalette.ColorRole.PlaceholderText, StyleColor.PlaceholderText)
            pal.setColor(qt.QPalette.ColorRole.WindowText, StyleColor.WindowText)
            
            pal.setColor(qt.QPalette.ColorRole.Highlight, StyleColor.Highlight)
            pal.setColor(qt.QPalette.ColorRole.HighlightedText, StyleColor.HighlightedText)
            
            pal.setColor(qt.QPalette.ColorRole.Link, StyleColor.Link)
            pal.setColor(qt.QPalette.ColorRole.LinkVisited, StyleColor.LinkVisited)
            
            pal.setColor(qt.QPalette.ColorRole.ToolTipBase, StyleColor.ToolTipBase)
            pal.setColor(qt.QPalette.ColorRole.ToolTipText, StyleColor.ToolTipText)

            pal.setColor(qt.QPalette.ColorGroup.Active, qt.QPalette.ColorRole.ButtonText, StyleColor.ButtonText)
            pal.setColor(qt.QPalette.ColorGroup.Inactive, qt.QPalette.ColorRole.ButtonText, StyleColor.ButtonText)
            pal.setColor(qt.QPalette.ColorGroup.Disabled, qt.QPalette.ColorRole.ButtonText, StyleColor.ButtonTextDisabled)

            pal.setColor(qt.QPalette.ColorGroup.Active, qt.QPalette.ColorRole.WindowText, StyleColor.WindowText)
            pal.setColor(qt.QPalette.ColorGroup.Inactive, qt.QPalette.ColorRole.WindowText, StyleColor.WindowText)
            pal.setColor(qt.QPalette.ColorGroup.Disabled, qt.QPalette.ColorRole.WindowText, StyleColor.WindowTextDisabled)
            
            pal.setColor(qt.QPalette.ColorGroup.Disabled, qt.QPalette.ColorRole.Text, StyleColor.TextDisabled)
        
        if isinstance(obj, qt.QFrame):
            obj.setFrameShape(qt.QFrame.Shape.NoFrame)
            
        if isinstance(obj, qt.QLabel):          
            obj.setBackgroundRole(qt.QPalette.ColorRole.Mid)
            
        if isinstance(obj, qt.QStackedWidget):          
            obj.setAutoFillBackground(True)  
            obj.setBackgroundRole(qt.QPalette.ColorRole.Dark)
            
        if isinstance(obj, qt.QMenuBar):
            pal = qt.QPalette(obj.palette())
            pal.setColor(qt.QPalette.ColorRole.Window, pal.color(qt.QPalette.ColorRole.Mid) )
            obj.setPalette(pal)
            
        # if isinstance(obj, (qt.QTabWidget, qt.QTabBar) ):
        #     obj.setAutoFillBackground(True)
        #     obj.setBackgroundRole(qt.QPalette.ColorRole.Dark)
            
        #     pal = obj.palette()
        #     pal.setColor( qt.QPalette.ColorRole.Window, qt.QColor(0,0,0))
        #     pal.setColor( qt.QPalette.ColorRole.Midlight, qt.QColor(0,0,0))
        #     pal.setColor( qt.QPalette.ColorRole.Light, qt.QColor(0,0,0))
        #     obj.setPalette(pal)

        # if isinstance(obj, qt.QTextEdit):
        #     pal = obj.palette()
        #     #pal.setColor( qt.QPalette.ColorRole.Base, qt.QColor(0,0,0,255) )
        #     obj.setPalette(pal)

        if isinstance(obj, qt.QApplication):
            obj.setStyleSheet(f"""
QLabel {{
    padding: 4px;
}}

QTabWidget::tab-bar {{
            alignment: center;
        }}
""")
        return None

    def pixelMetric(self, metric : QStyle.PixelMetric, option: qt.QStyleOption, widget : qt.QWidget) -> int:
        
        if metric in [  QStyle.PixelMetric.PM_LayoutTopMargin, 
                        QStyle.PixelMetric.PM_LayoutBottomMargin, 
                        QStyle.PixelMetric.PM_LayoutLeftMargin, 
                        QStyle.PixelMetric.PM_LayoutRightMargin,
                        # QStyle.PixelMetric.PM_MenuPanelWidth, 
                        # QStyle.PixelMetric.PM_MenuBarVMargin, 
                        # QStyle.PixelMetric.PM_MenuBarHMargin, 
                        # QStyle.PixelMetric.PM_MenuHMargin, 
                        # QStyle.PixelMetric.PM_MenuVMargin,
                        QStyle.PixelMetric.PM_LayoutHorizontalSpacing, 
                        QStyle.PixelMetric.PM_LayoutVerticalSpacing,
                      ]:
            return 0
        elif metric in [QStyle.PixelMetric.PM_TabBarTabVSpace, 
                        QStyle.PixelMetric.PM_TabBarTabHSpace,
                      ]:
            return 12
        elif metric == QStyle.PixelMetric.PM_ButtonIconSize:
            return icon_Size_to_int[Size.Default]
            
                        
        return super().pixelMetric(metric, option, widget)
        
class QWidgetFilter(qt.QObject):
    def __init__(self):
        super().__init__()
        
        self._funcs = { qt.QEvent.Type.Wheel: self._wheel_filter,
                        qt.QEvent.Type.KeyPress: self._keypress_filter,
                        qt.QEvent.Type.Leave:  self._leave_filter, }
      
    def _wheel_filter(self, obj : qt.QObject, ev : qt.QWheelEvent):
        
        if isinstance(obj, (qt.QAbstractSpinBox, qt.QAbstractSlider, qt.QTabBar, qt.QComboBox) ) \
           and not isinstance(obj, qt.QScrollBar):
            if qt.QApplication.focusWidget() != obj:
                ev.ignore()
                return True
            
        return False
        
    def _keypress_filter(self, obj : qt.QObject, ev : qt.QWheelEvent):
        if isinstance(obj, qt.QAbstractButton):
            if ev.key() == qt.Qt.Key.Key_Space:
                return True
        elif isinstance(obj, qt.QWidget):
            if ev.key() == qt.Qt.Key.Key_Tab:
                return True
            
        return False
    
    def _leave_filter(self, obj : qt.QObject, ev : qt.QWheelEvent):
        if isinstance(obj, (qt.QAbstractButton, qt.QAbstractSpinBox, qt.QAbstractSlider, qt.QComboBox)):
            obj.clearFocus()
        return False
    
    def eventFilter(self, obj : qt.QObject, ev : qt.QEvent):
        func = self._funcs.get(ev.type(), None)
        if func is not None:
            return func(obj, ev)
        return False