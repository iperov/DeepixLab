import colorsys

from core import ax, lx, mx, qt, qx

from .FGraph import FGraph
from .MxGraph import MxGraph


class QxGraph(qx.QVBox):
    def __init__(self, graph : MxGraph):
        super().__init__()
        self._graph = graph
        self._main_thread = ax.get_current_thread()
        self._fg = ax.FutureGroup().dispose_with(self)

        self._bottleneck_fg = ax.FutureGroup().dispose_with(self)

        self._data = None

        self._m = FGraph()

        self._q_canvas = _Canvas(self)

        self._q_slider = q_slider = qx.QHRangeDoubleSlider()

        self._mx_selected_names = mx.MultiChoice[str](availuator=lambda: self._m.names,
                                                      defer=lambda new_names, names, mc: self.apply_model(self._m.set_selected_names(new_names)),
                                                      ).dispose_with(self)

        (self.add(qx.QCheckBoxMxMultiChoice(self._mx_selected_names).v_compact(), align=qx.Align.CenterH)
             .add(self._q_canvas.v_expand())
             .add(qx.QHBox().v_compact()
                                .add(q_slider.set_decimals(4).set_range(0.0, 1.0))

                                .add(qx.QPushButton().h_compact().set_icon_size(qx.Size.S).set_icon(qx.IconDB.cut_sharp)
                                        .inline(lambda btn: btn.mx_clicked.listen(lambda: (self._graph.trim(*q_slider.mx_values.get()),
                                                                                        q_slider.mx_values.set((0.0,1.0)))  )))) )

        self._mx_settings.reflect(lambda settings, enter, bag=mx.Disposable().dispose_with(self):
                                  self.__ref_settings(settings, enter, bag))

        qx.QApplication.instance().mx_language.reflect(self._ref_lang).dispose_with(self)

        graph.mx_data.reflect(lambda data: self._update_data()).dispose_with(self)

    def __ref_settings(self, settings : qx.QSettings, enter : bool, bag : mx.Disposable):
        if enter:
            self._mx_selected_names.listen(lambda selected_names: settings.state.set('selected_names', selected_names) ).dispose_with(bag)
            if (selected_names := settings.state.get('selected_names', None)) is not None:
                self._mx_selected_names.set(selected_names)

            self._q_slider.mx_values.listen(lambda values: (settings.state.set('slider_values', values),
                                                            self.apply_model(self._m.set_data_view_range(values)) )
                                                        ).dispose_with(bag)
            if (slider_values := settings.state.get('slider_values', None)) is not None:
                self._q_slider.mx_values.set(slider_values)

        else:
            bag.dispose_items()

    def _ref_lang(self, lang : str):
        self._lang = lang
        self._q_canvas.update()

    @ax.task
    def _update_data(self):
        yield ax.switch_to(self._main_thread)
        yield ax.attach_to(self._fg)

        self._data = self._graph.mx_data.get()

        yield ax.attach_to(self._bottleneck_fg, max_tasks=1)

        while True:
            data = self._data

            m_data = FGraph.Data(array=data.buffer[:data.length],
                                 names=data.names,
                                 colors = [  colorsys.hsv_to_rgb( n*(1.0/len(data.names)), 0.5, 1.0 )
                                             for n in range(len(data.names)) ] )

            self.apply_model( self._m.set_data(m_data) )

            yield ax.sleep(1.0)

            if data is self._data:
                break

    @property
    def model(self) -> FGraph: return self._m

    def apply_model(self, new_m : FGraph):
        m, self._m = self._m, new_m
        if new_m is m:
            # Nothing was changed
            return


        changed_names = new_m.selected_names != m.selected_names

        upd =   new_m.cli_size != m.cli_size or \
                new_m.mouse_l_down_cli_pt != m.mouse_l_down_cli_pt or \
                new_m.mouse_cli_pt != m.mouse_cli_pt or \
                new_m.cli_image != m.cli_image

        upd_geo = False

        if new_m.names != m.names:
            self._mx_selected_names.reevaluate()
            self._mx_selected_names.set(new_m.names)

        if changed_names:
            self._mx_selected_names._set(new_m.selected_names)

        if upd:
            self.update()

        if upd_geo:
            self.update_geometry()
            self._q_canvas.update_geometry()






class _Canvas(qx.QWidget):
    def __init__(self, host : QxGraph):
        super().__init__()
        self._host = host
        self._qp = qt.QPainter()

        self.set_mouse_tracking(True)
        self.set_cursor(qt.Qt.CursorShape.BlankCursor)


    def _minimum_size_hint(self) -> qt.QSize: return qt.QSize(64, 64)

    def _leave_event(self, ev: qt.QEvent):
        super()._leave_event(ev)
        self._host.apply_model( self._host._m.mouse_leave())

    def _resize_event(self, ev: qt.QResizeEvent):
        super()._resize_event(ev)
        self._host.apply_model( self._host._m.set_cli_size(ev.size().width(), ev.size().height()))

    def _mouse_press_event(self, ev: qt.QMouseEvent):
        super()._mouse_press_event(ev)
        if ev.button() == qt.Qt.MouseButton.LeftButton:
            self._host.apply_model( self._host._m.mouse_lbtn_down(qt.QPoint_to_FVec2f(ev.pos()), ctrl_pressed=qt.QApplication.keyboardModifiers() == qt.Qt.KeyboardModifier.ControlModifier,
                                                                shift_pressed=qt.QApplication.keyboardModifiers() == qt.Qt.KeyboardModifier.ShiftModifier))

    def _mouse_move_event(self, ev: qt.QMouseEvent):
        super()._mouse_move_event(ev)
        self._host.apply_model( self._host._m.mouse_move( qt.QPoint_to_FVec2f(ev.pos())) )

    def _mouse_release_event(self, ev: qt.QMouseEvent):
        super()._mouse_release_event(ev)
        if ev.button() == qt.Qt.MouseButton.LeftButton:
            self._host.apply_model( self._host._m.mouse_lbtn_up(qt.QPoint_to_FVec2f(ev.pos())) )

    def _wheel_event(self, ev: qt.QWheelEvent):
        super()._wheel_event(ev)

        self._host.apply_model( self._host._m.mouse_wheel(qt.QPointF_to_FVec2f(ev.position()),
                                                          ev.angleDelta().y(),
                                                          ctrl_pressed=qt.QApplication.keyboardModifiers() == qt.Qt.KeyboardModifier.ControlModifier,
                                                          shift_pressed=qt.QApplication.keyboardModifiers() == qt.Qt.KeyboardModifier.ShiftModifier) )



    def _paint_event(self, ev : qt.QPaintEvent):
        rect = self.rect
        qp = self._qp
        qp.begin(self.q_widget)

        host = self._host
        lang = host._lang

        m = host._m
        W,H = m.cli_size

        qp.drawPixmap(0,0, qt.QPixmap_from_FImage(m.cli_image))

        if (mouse_cli_pt := m.mouse_cli_pt) is not None:

            pen_color = qt.QColor(qx.StyleColor.Text)

            if (cli_selection_ab := m.cli_selection_ab) is not None:
                s, e = cli_selection_ab
                pen_color.setAlpha(64)
                qp.setPen(pen_color)
                qp.fillRect( qt.QRect(s, 0, e-s+1, H), pen_color )


            pen_color.setAlpha(127)
            qp.setPen(pen_color)
            qp.drawLine(0, mouse_cli_pt.y, W, mouse_cli_pt.y)

            if (data := m.selected_view_data) is not None:

                array = data.array
                N, C = array.shape
                if N > 0:

                    data_N_start = data.N_start
                    data_N_end = data_N_start + N

                    font = qx.QFontDB.instance().fixed_width()
                    fm = qt.QFontMetrics(font)

                    text_lines = []
                    text_sizes = []
                    text_colors = []

                    text_lines.append(text :=  f"{lx.L('@(Average_for)', lang)} {data_N_start} - {data_N_end}")
                    text_sizes.append(fm.boundingRect(qt.QRect(), 0, text).size())
                    text_colors.append(qx.StyleColor.Text)

                    if C != 0:
                        sel_data = array.mean(0)
                        if sel_data.max() != 0:
                            for c in range(C):
                                v = sel_data[c]
                                if v != 0.0:
                                    text_lines.append(text := lx.L(f'{v:.4f} — {data.names[c]}', lang) )
                                    text_sizes.append(fm.boundingRect(qt.QRect(), 0, text).size())
                                    text_colors.append( qt.QColor.fromRgbF(*data.colors[c][::-1]) )

                    tooltip_width  = max(size.width() for size in text_sizes)
                    tooltip_height = sum(size.height() for size in text_sizes)

                    tooltip = qt.QPixmap(tooltip_width, tooltip_height)
                    tooltip.fill(qt.QColor(0,0,0,0))
                    tooltip_qp = qt.QPainter(tooltip)
                    tooltip_qp.setFont(font)

                    text_y = 0
                    for text, color, size in zip(text_lines, text_colors, text_sizes):
                        text_height = size.height()
                        tooltip_qp.setPen(color)
                        tooltip_qp.drawText(0, text_y, tooltip_width, text_height, qt.Qt.AlignmentFlag.AlignLeft, text )
                        text_y += text_height
                    tooltip_qp.end()

                    tooltip_pad = 4
                    tooltip_rect = tooltip.rect()
                    draw_at_left = mouse_cli_pt.x >= (tooltip_rect.width()+tooltip_pad*2)
                    draw_at_top  = mouse_cli_pt.y >= (tooltip_rect.height()+tooltip_pad*2)

                    qp.fillRect  (  tooltip_rect.translated(mouse_cli_pt.x+ ( (-tooltip_rect.width() -tooltip_pad*2) if draw_at_left else 1),
                                                            mouse_cli_pt.y+ ( (-tooltip_rect.height()-tooltip_pad*2) if draw_at_top else 1) )
                                                .adjusted(0,0,tooltip_pad*2,tooltip_pad*2),
                                    qt.QColor(0,0,0,150))

                    qp.drawPixmap(  tooltip_rect.translated(mouse_cli_pt.x+ ( (-tooltip_rect.width()-tooltip_pad)  if draw_at_left else 1+tooltip_pad),
                                                            mouse_cli_pt.y+ ( (-tooltip_rect.height()-tooltip_pad) if draw_at_top else 1+tooltip_pad) ),
                                    tooltip)


        qp.end()




