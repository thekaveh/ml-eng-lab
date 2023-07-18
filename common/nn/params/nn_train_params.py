from __future__ import annotations

from typing import Optional
from torch.utils.data import DataLoader
from dataclasses import dataclass, field, replace

from ..enum.optim import Optim
from ..params.nn_optim_params import NNOptimParams
from ..params.nn_scheduler_params import NNSchedulerParams
    
@dataclass(frozen=True, kw_only=True, slots=True)
class NNTrainParams:
    n_epochs        : int
    scheduler_params: NNSchedulerParams     = NNSchedulerParams(patience=5, factor=9e-1, threshold=1e-4)
    optim_params    : NNOptimParams         = NNOptimParams(optim=Optim.ADAM, lr_start=9e-1, momentum=(0.9, 0.999), weight_decay=5e-4)
    
    train_loader    : Optional[DataLoader]  = field(repr=False, default=None)
    val_loader      : Optional[DataLoader]  = field(repr=False, default=None)
    
    def with_train_loader(self, value: DataLoader) -> NNTrainParams:
        return replace(self, train_loader=value)
    
    def with_val_loader(self, value: DataLoader) -> NNTrainParams:
        return replace(self, val_loader=value)
    
    def __str__(self):
        return f"Train={{n_epochs={self.n_epochs}, OptimParams={self.optim_params}, SchedulerParams={self.scheduler_params}}}"
            
    def to_dict(self):
        return dict(
            n_epochs            = self.n_epochs
            , optim_params      = self.optim_params.to_dict()
            , scheduler_params  = self.scheduler_params.to_dict()
        )
        
    @staticmethod
    def from_dict(rep: dict) -> NNTrainParams:
        return NNTrainParams(
            n_epochs            = rep['n_epochs']
            , optim_params      = NNOptimParams.from_dict(rep['optim_params'])
            , scheduler_params  = NNSchedulerParams.from_dict(rep['scheduler_params'])
        )