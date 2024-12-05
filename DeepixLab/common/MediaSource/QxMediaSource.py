from __future__ import annotations

from core import mx, qt, qx

from ..Timeline import QxTimeline
from .MxMediaSource import MxMediaSource


class QxMediaSourceHead(qx.QHBox):
    def __init__(self, ms : MxMediaSource):
        super().__init__()
        self._ms = ms
        self._pix_hdr_vbox = qx.QVBox()
        self._media_path_hbox = qx.QHBox()

        qx.QPopupMxTextEmitter(ms.mx_error).set_parent(self)

        (self.v_compact().add(qx.QLabel().set_text('@(Source_type)').h_compact())
                            .add(qx.QComboBoxMxStateChoice(ms.mx_source_type))
                            .add(self._pix_hdr_vbox.hide())
                            .add(self._media_path_hbox) )

        ms.mx_source_type.reflect(lambda source_type, enter, bag=mx.Disposable().dispose_with(self):
                                  self._ref_source_type(source_type, enter, bag)).dispose_with(self)

    def _ref_source_type(self,  source_type : MxMediaSource.SourceType, enter : bool, bag : mx.Disposable):
        if enter:
            self._media_path_hbox.add(qx.QMxPathState(self._ms.mx_media_path).dispose_with(bag))
            self._ms.mx_media_path.mx_opened.reflect(lambda opened, enter, bag=mx.Disposable().dispose_with(self):
                                                    self._ref_state(source_type, opened, enter, bag)).dispose_with(bag)
        else:
            bag.dispose_items()

    def _ref_state(self, source_type : MxMediaSource.SourceType,
                         opened : bool, enter : bool, bag : mx.Disposable):
        if enter:
            if opened:
                if source_type == MxMediaSource.SourceType.VideoFile:
                    (self._pix_hdr_vbox
                        .add( qx.QHBox().dispose_with(bag)
                                .add_spacer(4)
                                .add( qx.QCheckBoxMxFlag(self._ms.mx_pix_hdr).set_text('HDR')) )
                        ).show()
                    mx.CallOnDispose(lambda: self._pix_hdr_vbox.hide() ).dispose_with(bag)
        else:
            bag.dispose_items()



class QxMediaSourceCentral(qx.QVBox):
    def __init__(self, ms : MxMediaSource):
        super().__init__()
        self._ms = ms
        ms.mx_source_type.reflect(lambda source_type, enter, bag=mx.Disposable().dispose_with(self):
                                  self._ref_source_type(source_type, enter, bag)).dispose_with(self)

    def _ref_source_type(self,  source_type : MxMediaSource.SourceType, enter : bool, bag : mx.Disposable):

        if enter:
            self._ms.mx_media_path.mx_opened.reflect(lambda opened, enter, bag=mx.Disposable().dispose_with(self):
                                                    self._ref_state(source_type, opened, enter, bag)).dispose_with(bag)
        else:
            bag.dispose_items()


    def _ref_state(self, source_type : MxMediaSource.SourceType,
                         opened : bool, enter : bool, bag : mx.Disposable):
        if enter:
            if opened:
                self.add(qx.QVBox().dispose_with(bag)
                            .add( preview_pixmap_widget := qx.QPixmapWidget().v_expand(min=64).h_expand() )
                            .add( QxTimeline(self._ms.mx_timeline) ) )

                self._preview_pixmap_widget = preview_pixmap_widget
                self._ms.mx_preview_image.reflect(lambda image: preview_pixmap_widget.set_pixmap(qt.QPixmap_from_FImage(image) if image is not None else None)
                                                  ).dispose_with(self)
        else:
            bag.dispose_items()
