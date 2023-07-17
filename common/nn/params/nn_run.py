from __future__ import annotations

import os
import yaml
import torch
import hashlib

import pandas as pd

from typing import List, Optional
from dataclasses import asdict, dataclass, field, replace

from ..enum.checkpoint_type import CheckpointType

from ..params.nn_params import NNParams
from ..params.nn_checkpoint import NNCheckpoint
from ..params.nn_train_params import NNTrainParams
from ..params.nn_model_params import NNModelParams
from ..params.nn_iteration_data_point import NNIterationDataPoint

@dataclass(frozen=True, kw_only=True, slots=True)
class NNRun:
    net_params  : NNParams
    train_params: NNTrainParams
    model_params: NNModelParams 
    
    _id         : Optional[str]                         = field(repr=False, default=None)
    _rep        : Optional[dict]                        = field(repr=False, default=None)
    idps        : Optional[List[NNIterationDataPoint]]  = field(repr=False, default=None)
    
    @property
    def id(self) -> str:
        return self._id
    
    @property
    def rep(self) -> str:
        return self._rep
    
    def __post_init__(self):
        rep = dict(
            model_params    = self.model_params.to_dict()
            , net_params    = self.net_params.to_dict()
            , train_params  = self.train_params.to_dict()
        )
        
        id = hashlib.md5(
            str(rep).encode('utf-8')
        ).hexdigest()
        
        object.__setattr__(self, '_id', id)
        object.__setattr__(self, '_rep', {"id": id, **rep})
    
    def with_idps(self, value: List[NNIterationDataPoint]) -> NNRun:
        return replace(self, idps=value)
        
    def save(self) -> NNRun:
        run_path = os.path.join(os.getcwd(), "runs", self.id)
        best_run_path = os.path.join(os.getcwd(), "runs", "best")
        
        csv_path = os.path.join(run_path, "idps.csv")
        yaml_path = os.path.join(run_path, "run.yaml")

        if not os.path.exists(run_path):
            os.makedirs(run_path)
        
        with open(yaml_path, 'w') as f:
            yaml.dump(self.rep, f)
            
        pd.json_normalize(
            data=[asdict(idp) for idp in self.idps]
        ).to_csv(csv_path)
            
        if not os.path.exists(best_run_path):
            os.symlink(src=run_path, dst=best_run_path)
        else:
            best_err = NNCheckpoint.load(run="best", type=CheckpointType.BEST).idp.val_edp.error
            curr_err = NNCheckpoint.load(run=self.id, type=CheckpointType.BEST).idp.val_edp.error
            
            if curr_err < best_err:
                os.remove(path=best_run_path)
                os.symlink(src=run_path, dst=best_run_path)
        
        return self