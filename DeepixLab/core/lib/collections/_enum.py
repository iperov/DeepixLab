from enum import Enum
from typing import TypeVar

T = TypeVar('T')

def get_enum_id_by_name(enum_cls : Enum, str_id : str, default):
    """get enum_cls's id by str_id, otherwise return default"""
    for id in enum_cls:
        if id.name == str_id:
            return id
    return default
