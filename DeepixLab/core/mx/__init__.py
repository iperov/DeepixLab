"""
ModelX library. 

A set of mechanisms and base classes to operate the backend(model) applicable for view/view-models/view-controllers (such as graphical or console).

Designed and developed from scratch by github.com/iperov
"""

from .Chain import Chain, ChainLink, IChain_v
from .Disposable import CallOnDispose, Disposable
from .Event import (Event, Event0, Event1, Event2, Event3, IEvent0_rv,
                    IEvent0_v, IEvent1_rv, IEvent1_v, IEvent2_rv, IEvent2_v,
                    IEvent3_rv, IEvent3_v, IEvent_v, IReplayEvent1_rv,
                    IReplayEvent1_v, IReplayEvent_rv, IReplayEvent_v,
                    ReplayEvent, ReplayEvent1)
from .EventConnection import EventConnection
from .Flag import Flag, IFlag_rv, IFlag_v
from .FProperty import FProperty, IFProperty_rv, IFProperty_v
from .Menu import IMenu_v, Menu
from .MultiChoice import IMultiChoice_v, MultiChoice
from .Number import INumber_rv, INumber_v, Number
from .Path import IPath_v, Path
from .Progress import IProgress_rv, Progress
from .Property import (EvaluableProperty, GetSetProperty,
                       IEvaluableProperty_rv, IProperty_rv, IProperty_v,
                       Property)
from .State import IState_rv, IState_v, State
from .StateChoice import IStateChoice_v, StateChoice
from .Text import IText_v, Text
from .TextEmitter import ITextEmitter_v, TextEmitter
