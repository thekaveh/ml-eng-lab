from __future__ import annotations

from typing import Optional
from dataclasses import dataclass, replace

from .nn_evaluation_data_point import NNEvaluationDataPoint

@dataclass(frozen=True, kw_only=True, slots=True)
class NNIterationDataPoint:
    iter_idx    : int
    epoch_idx   : int
    batch_idx   : int  
    train_edp   : NNEvaluationDataPoint
    val_edp     : Optional[NNEvaluationDataPoint]   = None
    
    def with_val_edp(self, value: NNEvaluationDataPoint):
        return replace(self, val_edp=value)
    
    def to_dict(self) -> dict:
        return dict(
            iter_idx    = self.iter_idx
            , epoch_idx = self.epoch_idx
            , batch_idx = self.batch_idx
            , train_edp = self.train_edp.to_dict()
            , val_edp   = self.val_edp.to_dict() if self.val_edp is not None else None
        )
    
    @staticmethod
    def from_dict(rep: dict) -> NNIterationDataPoint:
        return NNIterationDataPoint(
            iter_idx    = rep['iter_idx']
            , epoch_idx = rep['epoch_idx']
            , batch_idx = rep['batch_idx']
            , train_edp = NNEvaluationDataPoint.from_dict(
                dict(
                    loss=rep['train_edp.loss']
                    , error=rep['train_edp.error']
                    , accuracy=rep['train_edp.accuracy']
                    , f1=rep['train_edp.f1']
                    , recall=rep['train_edp.recall']
                    , precision=rep['train_edp.precision']
                )
            )
            , val_edp = NNEvaluationDataPoint.from_dict(
                dict(
                    loss=rep['val_edp.loss']
                    , error=rep['val_edp.error']
                    , accuracy=rep['val_edp.accuracy']
                    , f1=rep['val_edp.f1']
                    , recall=rep['val_edp.recall']
                    , precision=rep['val_edp.precision']
                )
            )
        )