from __future__ import annotations

import os
import torch

from typing import Optional
from dataclasses import dataclass
from collections import OrderedDict

from ..enum.checkpoint_type import CheckpointType
from ..params.nn_iteration_data_point import NNIterationDataPoint
    
@dataclass(frozen=True, kw_only=True, slots=True)
class NNCheckpoint:
    idp         : NNIterationDataPoint
    model_state : OrderedDict
    optim_state : OrderedDict
    
    def to_file(self, path: str) -> None:
        dir_path = os.path.dirname(path)
        
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        torch.save(self, path)
        
    def to_type(self, type: CheckpointType) -> None:
        self.to_file(path=f"./checkpoint/{str(type)}.pt")
        
    @staticmethod
    def from_file(path: str) -> Optional[NNCheckpoint]:
        if not os.path.exists(path):
            return None
        
        ret = torch.load(path)
        
        if not isinstance(ret, NNCheckpoint):
            return None
        
        return ret
    
    @staticmethod
    def from_type(type: CheckpointType) -> Optional[NNCheckpoint]:
        return NNCheckpoint.from_file(path=f"./checkpoint/{str(type)}.pt")