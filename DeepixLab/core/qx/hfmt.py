"""
HTML format helpers
"""
from .. import qt
from .QShortcut import QShortcut
from .StyleColor import StyleColor

def colored_text(text : str, color : qt.QColor) -> str:
    return f"<span style='color: {color.name()};'>{text}</span>"


def colored_shortcut_keycomb(shortcut : QShortcut, color : qt.QColor = None) -> str:
    if color is None:
        color = StyleColor.Link
        
    return colored_text(f'[{qt.QKeyCombination_to_string(shortcut.key_comb)}]', color)
