from typing import Tuple

from .. import mx, qt
from .QWidget import QWidget
from .StyleColor import StyleColor


class QHRangeSlider(QWidget):
    def __init__(self):
        super().__init__()
        self.__minimum = 0
        self.__maximum = 10
        self.__moving_slider_id = 0
        self.__read_only = False

        self.__mx_slider_pressed = mx.Event1[int]().dispose_with(self)
        self.__mx_slider_released = mx.Event1[int]().dispose_with(self)

        self.__mx_values = mx.Property[ Tuple[int,int] ]((0,10), filter=self._flt_values).dispose_with(self)
        self.__mx_values.listen(lambda _: self.update())

        self.set_mouse_tracking(True)

    @property
    def mx_values(self) -> mx.IProperty_v[ Tuple[int,int] ]:
        return self.__mx_values

    def get_range(self) -> Tuple[int, int]:
        return self.__minimum, self.__maximum

    def set_range(self, minimum : int, maximum : int):
        if self.__minimum != minimum or self.__maximum != maximum:
            self.__minimum = minimum
            self.__maximum = max(minimum, maximum)
            self.__mx_values.set(self.__mx_values.get())

    def set_read_only(self, r : bool):
        self.__read_only = r
        return self

    def _flt_values(self, new_values, values):
        l, r = new_values
        l = min( max(l, self.__minimum), self.__maximum, r )
        r = max( min(r, self.__maximum), self.__minimum, l )
        return (l, r)

    def _size_hint(self) -> qt.QSize:
        super()._size_hint()

        opt = self._get_opt()
        w = 84
        h = self.q_style.pixelMetric(qt.QStyle.PixelMetric.PM_SliderThickness, opt, self.q_widget)

        return qt.QSize(w, h)

    def _mouse_press_event(self, ev: qt.QMouseEvent):
        if not self.__read_only:
            style = self.q_style
            pos = ev.pos()

            l_value, r_value = self.__mx_values.get()

            opt = self._get_opt()
            opt.sliderPosition = l_value
            is_first_slider = style.hitTestComplexControl(qt.QStyle.ComplexControl.CC_Slider, opt, pos, self.q_widget) == qt.QStyle.SubControl.SC_SliderHandle
            opt.sliderPosition = r_value
            is_second_slider = style.hitTestComplexControl(qt.QStyle.ComplexControl.CC_Slider, opt, pos, self.q_widget) == qt.QStyle.SubControl.SC_SliderHandle

            if is_first_slider and is_second_slider:
                is_first_slider = pos.x() < style.subControlRect(qt.QStyle.ComplexControl.CC_Slider, opt, qt.QStyle.SubControl.SC_SliderHandle).center().x()
                is_second_slider = not is_first_slider

            self.__moving_slider_id = 1 if is_first_slider else 2 if is_second_slider else 0

            if self.__moving_slider_id != 0:
                self.__mx_slider_pressed.emit(self.__moving_slider_id)

    def _mouse_move_event(self, ev: qt.QMouseEvent):
        if self.__moving_slider_id != 0:
            style = self.q_style
            opt = self._get_opt()
            sr = style.subControlRect(qt.QStyle.ComplexControl.CC_Slider, opt, qt.QStyle.SubControl.SC_SliderHandle)

            pos = self._pixelPosToRangeValue( ev.pos().x() - sr.width()/2 )

            l_value, r_value = self.__mx_values.get()

            if self.__moving_slider_id == 1:
                if pos <= r_value:
                    self.__mx_values.set((pos, r_value))
            elif self.__moving_slider_id == 2:
                if pos >= l_value:
                    self.__mx_values.set((l_value, pos))

    def _mouse_release_event(self, ev: qt.QMouseEvent):
        if self.__moving_slider_id != 0:
            self.__mx_slider_released.emit(self.__moving_slider_id)
            self.__moving_slider_id = 0

    def _pixelPosToRangeValue(self, pos : int):
        style = self.q_style
        opt = self._get_opt()
        gr = style.subControlRect(qt.QStyle.ComplexControl.CC_Slider, opt, qt.QStyle.SubControl.SC_SliderGroove)
        sr = style.subControlRect(qt.QStyle.ComplexControl.CC_Slider, opt, qt.QStyle.SubControl.SC_SliderHandle)
        sliderMin = gr.x()
        sliderMax = gr.right() - sr.width() + 1
        return style.sliderValueFromPosition(self.__minimum, self.__maximum, pos - sliderMin, sliderMax-sliderMin)

    def _paint_event(self, ev: qt.QPaintEvent):
        style = self.q_style
        qp = qt.QStylePainter(self.q_widget)

        opt = self._get_opt()
        opt.sliderPosition = opt.minimum
        opt.subControls = qt.QStyle.SubControl.SC_SliderGroove | qt.QStyle.SubControl.SC_SliderTickmarks

        groove_rect = style.subControlRect(qt.QStyle.ComplexControl.CC_Slider, opt, qt.QStyle.SubControl.SC_SliderGroove)

        # Draw empty groove
        qp.fillRect(groove_rect, StyleColor.Shadow)

        #qp.drawComplexControl(qt.QStyle.ComplexControl.CC_Slider, opt)

        # Draw filled groove between handles
        l_value, r_value = self.__mx_values.get()
        opt.sliderPosition = l_value
        lx = style.subControlRect(qt.QStyle.ComplexControl.CC_Slider, opt, qt.QStyle.SubControl.SC_SliderHandle).center().x()

        opt.sliderPosition = r_value
        rx = style.subControlRect(qt.QStyle.ComplexControl.CC_Slider, opt, qt.QStyle.SubControl.SC_SliderHandle).center().x()

        groove_rect.setLeft(lx)
        groove_rect.setRight(rx)

        # qp.setClipRect( clip_rect )

        # opt.sliderPosition = opt.maximum
        # opt.subControls = qt.QStyle.SubControl.SC_SliderGroove

        qp.fillRect(groove_rect, StyleColor.Mid)

        qp.setClipping(False)

        # Draw first handle
        # clip_rect = QRect(opt.rect)
        # clip_rect.setLeft(0)
        # clip_rect.setRight(lx-1)
        # qp.setClipRect( clip_rect )

        opt.subControls = qt.QStyle.SubControl.SC_SliderHandle
        opt.sliderPosition = l_value
        qp.drawComplexControl(qt.QStyle.ComplexControl.CC_Slider, opt)

        # Draw second handle
        # # clip_rect = QRect(opt.rect)
        # # clip_rect.setLeft(rx)
        # # qp.setClipRect( clip_rect )

        opt.sliderPosition = r_value
        qp.drawComplexControl(qt.QStyle.ComplexControl.CC_Slider, opt)

    def _get_opt(self) -> qt.QStyleOptionSlider:
        opt = qt.QStyleOptionSlider()
        opt.initFrom(self.q_widget)
        opt.minimum = self.__minimum
        opt.maximum = self.__maximum
        opt.tickPosition = qt.QSlider.TickPosition.NoTicks#self._tick_position
        opt.tickInterval = 0#self._tick_interval

        return opt