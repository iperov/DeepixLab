from __future__ import annotations

from ... import qt
from ..Color import Color
from ...qt import QStyle
from ..Size import Size, icon_Size_to_int

class QDarkFusionStyle(qt.QProxyStyle):
    def __init__(self):
        super().__init__('Fusion')
        
    # def get_text_color(self) -> qt.QColor: return qt.QColor(211, 211, 211)

    # def get_dark_color(self) -> qt.QColor: return qt.QColor(56, 56, 56)

    # def get_highdark_color(self) -> qt.QColor:return qt.QColor(64, 64, 70)

    # def get_mid_color(self) -> qt.QColor: return qt.QColor(72, 72, 72)

    # def get_light_color(self) -> qt.QColor: return qt.QColor(92, 92, 92)

    # def get_highmid_color(self) -> qt.QColor: return qt.QColor(72, 72, 80)

    # def get_highlight_color(self) -> qt.QColor: return qt.QColor(92, 92, 100)

    # def get_selected_color(self) -> qt.QColor: return qt.QColor(78, 155, 233)

    def polish(self, obj : qt.QObject):
        if isinstance(obj, (qt.QTabBar, ) ):
            #obj.setAutoFillBackground(True)
            #obj.setBackgroundRole(QPalette.ColorRole.Dark)
            pal = obj.palette()
            pal.setColor( qt.QPalette.ColorRole.Window, qt.QColor(68,68,68) )
            pal.setColor( qt.QPalette.ColorRole.Light, qt.QColor(68,68,68))
            obj.setPalette(pal)

        if isinstance(obj, qt.QTextEdit):
            pal = obj.palette()
            pal.setColor( qt.QPalette.ColorRole.Base, qt.QColor(0,0,0,0) )
            obj.setPalette(pal)

        if isinstance(obj, qt.QApplication):
            obj.setStyleSheet(f"""
QLabel {{
    padding: 4px;
}}
""")

        if isinstance(obj, qt.QPalette):
            pal = obj
            pal.setColor(qt.QPalette.ColorRole.WindowText, StyleColor.silver )
            pal.setColor(qt.QPalette.ColorRole.Shadow, qt.QColor(44, 44, 44))
            pal.setColor(qt.QPalette.ColorRole.Window, qt.QColor(56, 56, 56))
            pal.setColor(qt.QPalette.ColorRole.Button, qt.QColor(56, 56, 56))
            pal.setColor(qt.QPalette.ColorRole.Dark, qt.QColor(56, 56, 56))
            pal.setColor(qt.QPalette.ColorRole.Mid, qt.QColor(68, 68, 68))
            pal.setColor(qt.QPalette.ColorRole.Midlight, qt.QColor(80, 80, 80))
            pal.setColor(qt.QPalette.ColorRole.Light, qt.QColor(92, 92, 92))
            pal.setColor(qt.QPalette.ColorRole.Text, StyleColor.silver )
            pal.setColor(qt.QPalette.ColorRole.BrightText, StyleColor.red)
            pal.setColor(qt.QPalette.ColorRole.ButtonText, StyleColor.white)
            pal.setColor(qt.QPalette.ColorRole.Base, qt.QColor(25, 25, 25))
            pal.setColor(qt.QPalette.ColorRole.Highlight, qt.QColor(42, 130, 218))
            pal.setColor(qt.QPalette.ColorRole.HighlightedText, StyleColor.black)
            pal.setColor(qt.QPalette.ColorRole.Link, qt.QColor(42, 130, 218))
            #pal.setColor(qt.QPalette.ColorRole.LinkVisited, qt.QColor(42, 130, 218))
            pal.setColor(qt.QPalette.ColorRole.AlternateBase, qt.QColor(56, 56, 56))
            #pal.setColor(qt.QPalette.ColorRole.NoRole, qt.QColor(56, 56, 56))
            pal.setColor(qt.QPalette.ColorRole.ToolTipBase, StyleColor.silver )
            pal.setColor(qt.QPalette.ColorRole.ToolTipText, StyleColor.silver )
            pal.setColor(qt.QPalette.ColorRole.PlaceholderText, StyleColor.darkgray)
            #pal.setColor(qt.QPalette.ColorRole.NColorRoles, qt.QColor(56, 56, 56))

            pal.setColor(qt.QPalette.ColorGroup.Active, qt.QPalette.ColorRole.ButtonText, StyleColor.silver)
            pal.setColor(qt.QPalette.ColorGroup.Inactive, qt.QPalette.ColorRole.ButtonText, StyleColor.silver)
            pal.setColor(qt.QPalette.ColorGroup.Disabled, qt.QPalette.ColorRole.ButtonText, StyleColor.gray)

            pal.setColor(qt.QPalette.ColorGroup.Active, qt.QPalette.ColorRole.WindowText, StyleColor.silver)
            pal.setColor(qt.QPalette.ColorGroup.Inactive, qt.QPalette.ColorRole.WindowText, StyleColor.silver)
            pal.setColor(qt.QPalette.ColorGroup.Disabled, qt.QPalette.ColorRole.WindowText, StyleColor.gray)
            pal.setColor(qt.QPalette.ColorGroup.Disabled, qt.QPalette.ColorRole.Text, StyleColor.gray)
            return pal
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

# class QStyleWidgetFilter(qt.QObject):
#     def __init__(self):
#         super().__init__()

#         self._funcs = { #qt.QEvent.Type.Wheel: self._wheel_filter,
#                         #qt.QEvent.Type.KeyPress: self._keypress_filter,
#                         #qt.QEvent.Type.Leave:  self._leave_filter, 
#                         }

#     def _wheel_filter(self, obj : qt.QObject, ev : qt.QWheelEvent):

#         if isinstance(obj, (qt.QAbstractSpinBox, qt.QAbstractSlider, qt.QComboBox) ): #and not isinstance(obj, qt.QScrollBar)
           
#             if qt.QApplication.focusWidget() != obj:
#                 ev.ignore()
#                 return True

#         return False

#     # def _keypress_filter(self, obj : qt.QObject, ev : qt.QWheelEvent):
#     #     if isinstance(obj, qt.QWidget):
#     #         if ev.key() == qt.Qt.Key.Key_Tab:
#     #             return True
#     #     return False

#     def _leave_filter(self, obj : qt.QObject, ev : qt.QWheelEvent):
#         if isinstance(obj, (qt.QAbstractSpinBox, qt.QAbstractSlider, qt.QComboBox)):
#             obj.clearFocus()
#         return False

#     def eventFilter(self, obj : qt.QObject, ev : qt.QEvent):
#         func = self._funcs.get(ev.type(), None)
#         if func is not None:
#             return func(obj, ev)
#         return False


# text_color = QColor(200,200,200)
# pal = QPalette()
# pal.setColor(QPalette.ColorRole.Window, QColor(56, 56, 56))

# pal.setColor(QPalette.ColorRole.Light, QColor(80, 80, 80))
# pal.setColor(QPalette.ColorRole.Mid, QColor(68, 68, 68))
# pal.setColor(QPalette.ColorRole.Dark, QColor(56, 56, 56))
# pal.setColor(QPalette.ColorRole.Shadow, QColor(44, 44, 44))

# pal.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
# pal.setColor(QPalette.ColorRole.AlternateBase, QColor(56, 56, 56))
# pal.setColor(QPalette.ColorRole.ToolTipBase, text_color )
# pal.setColor(QPalette.ColorRole.ToolTipText, text_color )
# pal.setColor(QPalette.ColorRole.Text, text_color )
# pal.setColor(QPalette.ColorRole.Button, QColor(56, 56, 56))
# pal.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
# pal.setColor(QPalette.ColorRole.PlaceholderText, Qt.GlobalColor.darkGray)
# pal.setColor(QPalette.ColorRole.WindowText, text_color )
# pal.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
# pal.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
# pal.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
# pal.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)

# pal.setColor(QPalette.ColorGroup.Active, QPalette.ColorRole.ButtonText, text_color)
# pal.setColor(QPalette.ColorGroup.Inactive, QPalette.ColorRole.ButtonText, text_color)
# pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, Qt.GlobalColor.gray)

# pal.setColor(QPalette.ColorGroup.Active, QPalette.ColorRole.WindowText, text_color)
# pal.setColor(QPalette.ColorGroup.Inactive, QPalette.ColorRole.WindowText, text_color)
# pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, Qt.GlobalColor.gray)
# pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, Qt.GlobalColor.gray)

# self._pal = pal


        # if isinstance(obj, QFrame):
        #     obj.setFrameShape(QFrame.Shape.NoFrame)
        #     if isinstance(obj,  QFrame) and not isinstance(obj, (QLabel, QScrollArea, QSplitter, QTextEdit) ):
        #         obj.setAutoFillBackground(True)
        #         obj.setFrameShape( QFrame.Shape.NoFrame)
        # #         #obj.setBackgroundRole(QPalette.ColorRole.Mid)

        # if isinstance(obj, QPushButton):
        #     ...
        #     #obj.setIconSize()

        # #    obj.minimumSizeHint = lambda *_, self=obj, super=obj.minimumSizeHint: super(self)

        # # if isinstance(obj, (QTabBar, ) ):
        # #     #obj.setAutoFillBackground(True)
        # #     #obj.setBackgroundRole(QPalette.ColorRole.Dark)
        # #     pal = obj.palette()
        # #     pal.setColor( QPalette.ColorRole.Window, qt.QColor(68,68,68) )
        # #     #pal.setColor( QPalette.ColorRole.Dark, qt.QColor(0,0,0,0) )
        # #     #pal.setColor( QPalette.ColorRole.Mid, qt.QColor(0,0,0,0) )
        # #     #pal.setColor( QPalette.ColorRole.Midlight, qt.QColor(0,0,0,0))
        # #     pal.setColor( QPalette.ColorRole.Light, qt.QColor(68,68,68))

        # #     obj.setPalette(pal)

        # # if isinstance(obj, QTextEdit):
        # #     pal = obj.palette()
        # #     pal.setColor( QPalette.ColorRole.Base, qt.QColor(0,0,0,0) )
        # #     obj.setPalette(pal)
        # # #border-top: 1px solid rgb(68, 68, 68);


# QSplitter::handle:pressed {{
# }}
        # elif isinstance(obj, QPalette):
        #     text_color = qt.QColor(211,211,211)
        #     pal = obj
        #     pal.setColor(QPalette.ColorRole.Window, qt.QColor(56, 56, 56))

        #     pal.setColor(QPalette.ColorRole.Light, qt.QColor(80, 80, 80))
        #     pal.setColor(QPalette.ColorRole.Mid, qt.QColor(68, 68, 68))
        #     pal.setColor(QPalette.ColorRole.Dark, qt.QColor(56, 56, 56))
        #     pal.setColor(QPalette.ColorRole.Shadow, qt.QColor(44, 44, 44))


        #     pal.setColor(QPalette.ColorRole.Base, qt.QColor(25, 25, 25))
        #     pal.setColor(QPalette.ColorRole.AlternateBase, qt.QColor(56, 56, 56))
        #     pal.setColor(QPalette.ColorRole.ToolTipBase, text_color )
        #     pal.setColor(QPalette.ColorRole.ToolTipText, text_color )
        #     pal.setColor(QPalette.ColorRole.Text, text_color )
        #     pal.setColor(QPalette.ColorRole.Button, qt.QColor(56, 56, 56))
        #     pal.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        #     pal.setColor(QPalette.ColorRole.PlaceholderText, Qt.GlobalColor.darkGray)
        #     pal.setColor(QPalette.ColorRole.WindowText, text_color )
        #     pal.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        #     pal.setColor(QPalette.ColorRole.Link, qt.QColor(42, 130, 218))
        #     pal.setColor(QPalette.ColorRole.Highlight, qt.QColor(42, 130, 218))
        #     pal.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)

        #     pal.setColor(QPalette.ColorGroup.Active, QPalette.ColorRole.ButtonText, text_color)
        #     pal.setColor(QPalette.ColorGroup.Inactive, QPalette.ColorRole.ButtonText, text_color)
        #     pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, Qt.GlobalColor.gray)

        #     pal.setColor(QPalette.ColorGroup.Active, QPalette.ColorRole.WindowText, text_color)
        #     pal.setColor(QPalette.ColorGroup.Inactive, QPalette.ColorRole.WindowText, text_color)
        #     pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, Qt.GlobalColor.gray)
        #     pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, Qt.GlobalColor.gray)
        #     return pal


# PixelMetric = qt.QStyle.PixelMetric

# class QStyle(qt.QProxyStyle):
#     _instance : QStyle = None

#     @staticmethod
#     def instance() -> QStyle:
#         if QStyle._instance is None:
#             raise Exception('No QStyle instance.')
#         return QStyle._instance

#     def __init__(self):
#         super().__init__()
#         QStyle._instance = self

#         self._widget_filter = QStyleWidgetFilter()


#     # def get_selection_frame_color(self) -> qt.QColor:
#     #     return qt.QColor(92, 124, 180)

#     def polish(self : qt.QProxyStyle, obj : qt.QObject):

#         if isinstance(obj, qt.QWidget):
#             obj.installEventFilter(self._widget_filter)
#             obj.setFocusPolicy( qt.Qt.FocusPolicy.ClickFocus)

#         if isinstance(obj, qt.QTabWidget):
#             qt.wrap(obj, 'minimumSizeHint', lambda obj, super: qt.QSize(super().width()-6, super().height()-6) )
#             qt.wrap(obj, 'sizeHint', lambda obj, super: qt.QSize(super().width()-6, super().height()-6) )

#         if isinstance(obj, qt.QApplication):



#             

#         """

#         # QSlider::groove:horizontal {{
#         #     border: 1px solid #999999;
#         #     height: 8px; /* the groove expands to the size of the slider by default. by giving it a height, it has a fixed size */
#         #     background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #B1B1B1, stop:1 #c4c4c4);
#         #     margin: 2px 0;
#         # }}

#         # QSlider::handle:horizontal {{
#         #     background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b4b4b4, stop:1 #8f8f8f);
#         #     border: 1px solid #5c5c5c;
#         #     width: 18px;
#         #     margin: -2px 0; /* handle is placed by default on the contents rect of the groove. Expand outside the groove */
#         #     border-radius: 3px;
#         # }}
#         """
#     def pixelMetric(self, metric : PixelMetric, option: qt.QStyleOption, widget : qt.QWidget) -> int:
#         if metric in [PixelMetric.PM_LayoutTopMargin, PixelMetric.PM_LayoutBottomMargin, PixelMetric.PM_LayoutLeftMargin, PixelMetric.PM_LayoutRightMargin,
#                     PixelMetric.PM_LayoutHorizontalSpacing, PixelMetric.PM_LayoutVerticalSpacing,
#                     #PixelMetric.PM_DefaultFrameWidth,
#                     #PixelMetric.PM_MenuPanelWidth, PixelMetric.PM_MenuBarVMargin, PixelMetric.PM_MenuBarHMargin, PixelMetric.PM_MenuHMargin, PixelMetric.PM_MenuVMargin,
#                     ]:
#             return 0
#         # elif metric in [PixelMetric.PM_TabBarTabVSpace, PixelMetric.PM_TabBarTabHSpace,
#         #               ]:
#         #     return 12

#         return super().pixelMetric(metric, option, widget)


#             text_color = self.get_text_color().name()
#             dark_color = self.get_dark_color().name()
#             highdark_color = self.get_highdark_color().name()
#             mid_color = self.get_mid_color().name()
#             light_color = self.get_light_color().name()

#             highlight_color = self.get_highlight_color().name()

#             selected_color = self.get_selected_color().name()
            
#             p = Path(__file__).parent / 'assets' / 'icons' / 'arrow_left.png'
#             print(p.as_posix())
            
#             obj.setStyleSheet(f"""
# QWidget {{
#     background: {dark_color};
#     color: {text_color};
# }}
# QToolTip {{
#     color: black;
#     background: {text_color};
#     border: 1px solid black;
# }}
# QLabel {{
#     background: {mid_color};
#     padding: 4px;
#     margin-right: 1px;
#     margin-bottom: 1px;
# }}
# QFrame {{
#     background: {mid_color};
#     margin-right: 1px;
#     margin-bottom: 1px;
# }}
# QLineEdit {{
#     background: {highdark_color};
#     border: 0px;
#     margin-right: 1px;
#     margin-bottom: 1px;
#     min-width: 200px;
#     selection-background-color: rgb(78,155,233);
#     selection-color: rgb(0,0,0);
# }}
# QTextEdit {{
#     background: {highdark_color};
#     border: 0px;
#     margin-right: 1px;
#     margin-bottom: 1px;
#     min-width: 200px;
#     selection-background-color: rgb(78,155,233);
#     selection-color: rgb(0,0,0);
# }}
# QDoubleSpinBox {{
#     background: {highdark_color};
#     padding: 2px;
#     margin-right: 1px;
#     margin-bottom: 1px;
#     /*color: rgb(50,200,255);*/
#     selection-background-color: rgb(78,155,233);
#     selection-color: rgb(0,0,0);
# }}
# QDoubleSpinBox::up-button {{
#     width: 16;
# }}
# QDoubleSpinBox::down-button {{
#     width: 16;
# }}
# QCheckBox {{
#     background: {mid_color};
#     padding: 2px;
#     margin-right: 1px;
#     margin-bottom: 1px;
# }}
# QCheckBox::indicator {{
#     width: 14px;
#     height: 14px;
# }}
# QCheckBox::indicator:unchecked {{
#     background: {dark_color};
#     border: 1px solid {dark_color};
#     border-radius: 8px;
# }}
# QCheckBox::indicator:checked {{
#     background: rgb(78,155,233);
#     border: 1px solid {dark_color};
#     border-radius: 8px;
# }}
# QProgressBar {{
#     background: {mid_color}; /*qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0, stop: 0 rgb(200,0,0), stop: 1 rgb(0,200,0));*/
#     border: 0px;
#     margin-right: 1px;
#     margin-bottom: 1px;
#     color : rgb(255,200,50);
# }}
# QProgressBar::chunk {{
#     background: {text_color};
# }}
# QPushButton {{
#     border: 0px;
#     background: {mid_color};
#     padding: 4px;
#     margin-right: 1px;
#     margin-bottom: 1px;
# }}
# QPushButton:hover {{
#     background: {highlight_color};
# }}
# QPushButton:pressed {{
#     background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 rgba(68,68,68,255) stop:0.5 rgba(68,68,68,0) stop:1 rgba(68,68,68,255) );
# }}
# QToolButton {{
#     border: 0px;
#     background: {mid_color};
#     padding: 4px;
#     margin-right: 1px;
#     margin-bottom: 1px;
# }}
# QToolButton:hover {{
#     background: {highlight_color};
# }}
# QToolButton:pressed {{
#     background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 rgba(68,68,68,255) stop:0.5 rgba(68,68,68,0) stop:1 rgba(68,68,68,255) );
# }}
# QToolButton:checked {{
#     background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 rgba(68,68,68,255) stop:0.5 rgba(68,68,68,0) stop:1 rgba(68,68,68,255) );
# }}
# QToolButton:checked:hover {{
#     background: {highlight_color};
# }}


# QScrollBar:horizontal {{
#     border: None;
#     background: {mid_color};
#     height: 15px;
#     margin: 0px 20px 1 20px;
# }}

# QScrollBar::handle:horizontal {{
#     background: {highlight_color};
#     min-width: 20px;
# }}

# QScrollBar::sub-line:horizontal {{
#     border: None;
#     background: {mid_color};
#     width: 19px;
#     subcontrol-position: left;
#     subcontrol-origin: margin;
#     margin: 0 1 1 0;
# }}

# QScrollBar::sub-line:horizontal:hover {{
#     background: {highlight_color};
# }}

# QScrollBar::add-line:horizontal {{
#     border: None;
#     background: {highlight_color};
#     width: 20px;
#     subcontrol-position: right;
#     subcontrol-origin: margin;
#     margin: 0 1 1 0;
# }}

# QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal, QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
#     background: none;
# }}

# QScrollBar:left-arrow:horizontal {{
#     background-image: url({p.as_posix()});

# }}

# QScrollArea {{
#     border: 0px;
# }}
# QMenuBar {{
#     background: {dark_color};
#     spacing: 0px;
# }}
# QMenuBar::item {{
#     background: {mid_color};
#     margin-right: 1px;
#     margin-bottom: 1px;
#     padding: 4px 8px;
# }}
# QMenuBar::item:selected {{
#     background: {highlight_color};
# }}
# QMenuBar::item:pressed {{
#     background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 rgba(68,68,68,255) stop:0.5 rgba(68,68,68,0) stop:1 rgba(68,68,68,255) );
# }}
# QMenu {{
#     background: {dark_color};
#     margin-right: 1px;
#     margin-bottom: 1px;
# }}
# QMenu::item {{
#     background: {mid_color};
#     margin-right: 1px;
#     margin-bottom: 1px;
#     padding: 4px 12px;
# }}
# QMenu::item:selected {{
#     background: {highlight_color};
# }}
# QMenu::item:pressed {{
#     background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 rgba(68,68,68,255) stop:0.5 rgba(68,68,68,0) stop:1 rgba(68,68,68,255) );
# }}
# QTabBar::tab {{
#     background: {mid_color};
#     border: 0px;
#     min-width: 8ex;
#     padding: 4px;
#     margin-right: 1px;
#     margin-bottom: 1px;
# }}
# QTabWidget::tab-bar {{
#     alignment: left;
# }}
# QTabBar::tab:selected {{
#     background: {highlight_color};
# }}
# QTabBar::tab:hover {{
#     background: {highlight_color};
# }}
# QTabWidget::pane {{
#     border-top: 0px solid {dark_color};
# }}
# /*QAbstractItemView*/
# QComboBox  {{
#     background: {mid_color};
#     border: 0px;
#     padding: 4px;
#     margin-right: 1px;
#     margin-bottom: 1px;
# }}
# QComboBox::drop-down {{
#     /*subcontrol-origin: padding;
#     subcontrol-position: top right;*/
#     width: 25px;
# /*
#     border-left-width: 1px;
#     border-left-color: darkgray;
#     border-left-style: solid;
#     border-top-right-radius: 3px;
#     border-bottom-right-radius: 3px;*/
# }}
# QListView {{
#     background: {dark_color};
#     border: 0px;
#     outline: 0;
# }}
# QListView::item {{
#     background: {mid_color};
#     padding: 4px;
#     margin-right: 1px;
#     margin-bottom: 1px;
# }}
# QListView::item:selected {{
#     background: {selected_color};
# }}
# QSplitter {{
#     background: {dark_color};
#     width: 6px;
#     height: 6px;
# }}
# QSplitter::handle {{
#     /*background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0, stop: 0.25 {dark_color}, stop: 0.5 rgb(92,92,92), stop: 0.75 {dark_color});*/
#     /*margin-right: 1px;
#     margin-bottom: 1px;*/
# }}
# QSplitterHandle:hover{{}}
# QSplitter::handle:horizontal:hover {{
#     background: {highlight_color};
# }}
# QSplitter::handle:vertical:hover {{
#     background: {highlight_color};
# }}
# QSplitter::handle:horizontal {{
#     width: 2px;
# }}
# QSplitter::handle:vertical {{
#     height: 2px;
# }}
# QSlider{{
#     background: {mid_color};
#     border: 0;
#     padding: 0;
#     margin-top: 0px;
#     margin-right: 1px;
#     margin-bottom: 1px;
# }}
# QSlider::groove:horizontal {{
#     border: 0px;
#     height: 8px;
#     background: {dark_color};
# }}
# QSlider::handle:horizontal {{
#     background: {highlight_color};
#     border: 0px;
#     width: 18px;
# }}
# QSlider::groove:vertical {{
#     border: 0px;
#     width: 8px;
#     background: {dark_color};
# }}
# QSlider::handle:vertical {{
#     background: {highlight_color};
#     border: 0px;
#     height: 18px;
# }}
# """)