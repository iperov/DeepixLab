from core import qt, qx

from .SSI import SSI


class QSSI:

    class Image(qx.QGrid):
        def __init__(self, ssi_image : SSI.Image):
            super().__init__()
            self.set_row_stretch(0,1,1,0)

            if (image := ssi_image.image) is not None:
                self.add( qx.QPixmapWidget().set_pixmap(qt.QPixmap_from_FImage(ssi_image.image)), 0, 0 )

            if ssi_image.caption is not None:
                caption = ssi_image.caption
                if image is not None:
                    caption = f'{caption}\n({image.shape[1]}x{image.shape[0]})'

                self.add( qx.QLabel().set_font(qx.FontDB.FixedWidth).set_align(qx.Align.CenterH).set_text(caption), 1, 0, align=qx.Align.CenterH)

    class Grid(qx.QGrid):
        def __init__(self, ssi_grid : SSI.Grid):
            super().__init__()

            for (row,col), item in ssi_grid.items.items():
                if isinstance(item, SSI.Image):
                    item_widget = QSSI.Image(item)
                else:
                    raise NotImplementedError()

                self.add(item_widget, row, col)

    class Sheet(qx.QVBox):
        def __init__(self, ssi_sheet : SSI.Sheet):
            super().__init__()

            tab_widget = qx.QTabWidget().set_tab_position(qx.QTabWidget.TabPosition.South)

            for name, section in ssi_sheet.sections.items():
                if isinstance(section, SSI.Grid):
                    section_widget = QSSI.Grid(section)
                else:
                    raise NotImplementedError()

                tab_widget.add_tab(lambda tab: tab.set_title(name).add(section_widget))

            self.add(tab_widget)
