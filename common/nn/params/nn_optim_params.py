from typing import Tuple, Union
from dataclasses import dataclass

from ..enum.optim import Optim

@dataclass(frozen=True, kw_only=True, slots=True)
class NNOptimParams:
    optim       : Optim
    lr_start    : float
    weight_decay: float
    momentum    : Union[float, Tuple[float, float]]
    
    def __str__(self):
        return f"[optim={self.optim}, lr_start={self.lr_start:1.0e}, weight_decay={self.weight_decay:1.0e}, momentum={self.momentum}]"
    
    def to_dict(self):
        return dict(
            lr_start        = self.lr_start
            , momentum      = str(self.momentum)
            , optim         = str(self.optim)
            , weight_decay  = self.weight_decay
        )
    
    def is_valid(self):
        if self.optim == Optim.SGD or self.optim == Optim.SGD_NESTEROV:
            return isinstance(self.momentum, float)
        elif self.optim == Optim.ADAM or self.optim == Optim.ADAM_AMSGRAD:
            return (
                isinstance(self.momentum, tuple)
                and len(self.momentum) == 2
                and all(isinstance(x, float) for x in self.momentum)
            )