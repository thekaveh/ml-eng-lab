from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict

from ..params.nn_iteration_data_point import NNIterationDataPoint
    
@dataclass(frozen=True, kw_only=True, slots=True)
class NNCheckpoint:
    idp         : NNIterationDataPoint
    model_state : OrderedDict
    optim_state : OrderedDict