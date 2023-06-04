from typing import Tuple, Union
from dataclasses import dataclass

@dataclass(frozen=True, kw_only=True, slots=True)
class NNOptimParams:
    lr_start    : float
    weight_decay: float
    momentum    : Union[float, Tuple[float, float]]
    
    def __str__(self):
        return f"[lr_start={self.lr_start:1.0e}, weight_decay={self.weight_decay:1.0e}, momentum={self.momentum}]"