from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True, kw_only=True, slots=True)
class NNSchedulerParams:
    patience    : int
    factor      : float
    threshold   : float
    
    def __str__(self):
        return f"[patience={self.patience}, factor={self.factor:1.0e}, threshold={self.threshold:1.0e}]"