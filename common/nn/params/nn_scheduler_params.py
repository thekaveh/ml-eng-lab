from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True, kw_only=True, slots=True)
class NNSchedulerParams:
    patience    : int
    factor      : float
    threshold   : float
    
    def __str__(self) -> str:
        return f"[patience={self.patience}, factor={self.factor:1.0e}, threshold={self.threshold:1.0e}]"
    
    def to_dict(self) -> dict:
        return dict(
            factor      = self.factor
            , patience  = self.patience
            , threshold = self.threshold
        )
    
    @staticmethod
    def from_dict(rep: dict) -> NNSchedulerParams:
        return NNSchedulerParams(
            factor      = rep['factor']
            , patience  = rep['patience']
            , threshold = rep['threshold']
        )