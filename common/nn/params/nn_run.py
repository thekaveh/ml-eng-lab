from __future__ import annotations

import os
import yaml
import torch
import hashlib

from typing import List, Optional
from collections import OrderedDict
from dataclasses import dataclass, field

from nn_params import NNParams
from nn_train_params import NNTrainParams
from nn_model_params import NNModelParams
from nn_iteration_data_point import NNIterationDataPoint

@dataclass(frozen=True, kw_only=True, slots=True)
class NNRun:
    net_params  : NNParams
    train_params: NNTrainParams
    model_params: NNModelParams 
    
    _id         : Optional[str]                         = field(repr=False, default=None)
    idps        : Optional[List[NNIterationDataPoint]]  = field(repr=False, default=None)
    
    @property
    def id(self) -> str:
        return self._id
    
    def __post_init__(self):
        id = hashlib.md5(
                str(self.net_params).encode('utf-8')
                + str(self.train_params).encode('utf-8')
                + str(self.model_params).encode('utf-8')
            ).hexdigest()
        
        object.__setattr__(self, '_id', id)
        
    # def save(self) -> None:
    #     path = f"./run/{self.id}.yaml"
        
    #     dir_path = os.path.dirname(path)
        
    #     if not os.path.exists(dir_path):
    #         os.makedirs(dir_path)
        
    #     with open(path, 'w') as f:
    #         yaml.dump(self, f)