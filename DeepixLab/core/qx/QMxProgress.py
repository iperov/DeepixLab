from .. import mx
from ._constants import Align
from .QBox import QVBox
from .QLabel import QLabel
from .QProgressBar import QProgressBar


class QMxProgress(QVBox):

    def __init__(self, progress : mx.IProgress_rv, hide_inactive = False):
        super().__init__()
        self._progress = progress
        self._hide_inactive = hide_inactive
        self._show_it_s = False
        
        self._mx_settings.reflect(lambda settings, enter, bag=mx.Disposable().dispose_with(self): 
                                   self.__ref_settings(settings, enter, bag))
        

    def set_show_it_s(self, show : bool):
        self._show_it_s = show
        return self
    
    def __ref_settings(self, settings, enter, bag : mx.Disposable):
        if enter:
            self._q_progress_bar = QProgressBar()
            self._q_caption = QLabel()

            self.add(QVBox().dispose_with(bag)
                        .add(QVBox()
                                .add(self._q_caption.hide(), align=Align.BottomE))
                        .add(self._q_progress_bar.set_minimum(0) )
                        .add(QVBox()))
            
            self._progress.mx_active.reflect(lambda active, enter, bag=mx.Disposable().dispose_with(bag): 
                                    self._ref_active(active, enter, bag)).dispose_with(bag)
        else:
            bag.dispose_items()

    def _ref_active(self, active : bool, enter : bool, bag : mx.Disposable):
        if enter:
            if active:
                if self._hide_inactive:
                    self.show()
                    
                self._progress.mx_model.reflect(self._ref_model).dispose_with(bag)
            else:
                if self._hide_inactive:
                    self.hide()
        else:
            bag.dispose_items()

    def _ref_model(self, model : mx.Progress.FModel, old_model : mx.Progress.FModel):
        if model is old_model:
            changed_caption = True
            changed_progress = True
        else:
            changed_caption = model.caption != old_model.caption

            changed_progress = model.is_infinity != old_model.is_infinity or \
                               ( not model.is_infinity and model.progress != old_model.progress )

        if changed_caption:
            if (caption := model.caption) is not None:
                self._q_caption.show().set_text(caption)
            else:
                self._q_caption.hide()

        if changed_progress:
            if model.is_infinity:
                self._q_progress_bar.set_format('')
                self._q_progress_bar.set_maximum(0).set_value(0)
            else:
                if self._show_it_s:
                    self._q_progress_bar.set_format(f'%v / %m ({model.it_s:.1f} @(it_s))')
                else:
                    self._q_progress_bar.set_format(f'%v / %m')

                self._q_progress_bar.set_maximum(model.progress_max).set_value(model.progress)

