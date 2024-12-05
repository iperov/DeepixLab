from .. import qt
from ._helpers import q_init
from .QWidget import QWidget
from ._constants import Align, Align_to_AlignmentFlag

class QTextBrowser(QWidget):
    def __init__(self, **kwargs):
        super().__init__(q_widget=q_init('q_text_browser', qt.QTextBrowser, **kwargs), **kwargs)
        
        q_text_browser = self.q_text_browser
        
    @property
    def q_text_browser(self) -> qt.QTextBrowser: return self.q_widget
    
    def set_align(self, align : Align):
        self.q_text_browser.setAlignment(Align_to_AlignmentFlag[align])
        return self
    
    def set_html(self, html : str):
        self.q_text_browser.setHtml(html)
        return self
    
    def set_open_external_links(self, b : bool):
        self.q_text_browser.setOpenExternalLinks(b)
        return self