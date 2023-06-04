from __future__ import annotations

from typing import Optional
from dataclasses import dataclass
from torch.utils.data import DataLoader

from ..enum.optim import Optim
from ..params.nn_optim_params import NNOptimParams
from ..params.nn_scheduler_params import NNSchedulerParams
    
@dataclass(frozen=True, kw_only=True, slots=True)
class NNTrainParams:
    n_epochs            : int
    optim               : Optim             = Optim.ADAM
    scheduler_params    : NNSchedulerParams = NNSchedulerParams(patience=5, factor=9e-1, threshold=1e-4)
    optim_params        : NNOptimParams     = NNOptimParams(lr_start=9e-1, momentum=(0.9, 0.999), weight_decay=5e-4)
    
    train_loader        : DataLoader
    val_loader          : Optional[DataLoader]  = None
    
    def __str__(self):
        return f"Train={{n_epochs={self.n_epochs}, optim={self.optim}, OptimParams={self.optim_params}, SchedulerParams={self.scheduler_params}}}"
    
    def is_valid(self):
        if self.optim == Optim.SGD or self.optim == Optim.SGD_NESTEROV:
            return isinstance(self.optim_params.momentum, float)
        elif self.optim == Optim.ADAM or self.optim == Optim.ADAM_AMSGRAD:
            return (
                isinstance(self.optim_params.momentum, tuple)
                and len(self.optim_params.momentum) == 2
                and all(isinstance(x, float) for x in self.optim_params.momentum)
            )