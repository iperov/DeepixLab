from __future__ import annotations

from .. import qt
from ._constants import Align, Align_to_AlignmentFlag
from ._helpers import q_init
from .QLayout import QLayout
from .QWidget import QWidget


class QGrid(QLayout):

    class Row:
        def __init__(self, grid : QGrid, row : int, col : int):
            self._grid = grid
            self._row = row
            self._col = col

        def grid(self) -> QGrid:
            return self._grid

        def add(self, widget : QWidget|None, row_span=1, col_span=1, align : Align = Align.CenterE) -> QGrid.Row:
            self._grid.add(widget, self._row, self._col, row_span=row_span, col_span=col_span, align=align)
            self._col += col_span
            return self

        def next_row(self, col : int = 0) -> QGrid.Row:
            self._row += 1
            self._col = col
            return self
        
        def next_col(self) -> QGrid.Row:
            self._col += 1
            return self

    class Col:
        def __init__(self, grid : QGrid, row : int, col : int):
            self._grid = grid
            self._row = row
            self._col = col

        def grid(self) -> QGrid:
            return self._grid

        def add(self, widget : QWidget|None, row_span=1, col_span=1, align : Align = Align.CenterE) -> QGrid.Col:
            self._grid.add(widget, self._row, self._col, row_span=row_span, col_span=col_span, align=align)
            self._row += row_span
            return self

        def next_col(self, row : int = 0) -> QGrid.Col:
            self._col += 1
            self._row = row
            return self

    def __init__(self, **kwargs):
        super().__init__(q_layout=q_init('q_grid_layout', qt.QGridLayout, **kwargs), **kwargs)

    @property
    def q_grid_layout(self) -> qt.QGridLayout: return self.q_layout

    def add(self, widget : QWidget|None, row : int, col : int, row_span=1, col_span=1, align : Align = Align.CenterE):
        """add widget"""
        if widget is not None:
            widget.set_parent(self)
            self.q_grid_layout.addWidget(widget.q_widget, row, col, row_span, col_span, alignment=Align_to_AlignmentFlag[align])
        return self

    def row(self, row : int, col : int = 0) -> QGrid.Row: return QGrid.Row(self, row, col)

    def col(self, col : int, row : int = 0) -> QGrid.Col: return QGrid.Col(self, row, col)

    def set_spacing(self, spacing : int):
        self.q_grid_layout.setSpacing(spacing)
        return self

    def set_column_stretch(self, column : int, stretch : int, *or_list):
        l = (column, stretch)
        if len(or_list) != 0:
            l = l + or_list
        for i in range(0, len(l), 2):
            column, stretch = l[i:i+2]
            self.q_grid_layout.setColumnStretch(column, stretch)
        return self

    def set_row_stretch(self, row : int, stretch : int, *or_list):
        l = (row, stretch)
        if len(or_list) != 0:
            l = l + or_list
        for i in range(0, len(l), 2):
            row, stretch = l[i:i+2]
            self.q_grid_layout.setRowStretch(row, stretch)
        return self
