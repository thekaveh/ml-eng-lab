from __future__ import annotations

from typing import Optional
from dataclasses import dataclass, replace

from .nn_evaluation_data_point import NNEvaluationDataPoint

@dataclass(frozen=True, kw_only=True, slots=True)
class NNIterationDataPoint:
    iter_idx        : int
    epoch_idx       : int
    batch_idx       : int  
    train_edp       : NNEvaluationDataPoint
    val_edp         : Optional[NNEvaluationDataPoint]   = None
    
    def with_val_edp(self, value: NNEvaluationDataPoint):
        return replace(self, val_edp=value)