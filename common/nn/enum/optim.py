from __future__ import annotations

from enum import Enum
from torch import nn, optim

from ..params.nn_optim_params import NNOptimParams

class Optim(Enum):
    SGD             = "sgd"
    ADAM            = "adam"
    ADAM_AMSGRAD    = "adam_amsgrad"
    SGD_NESTEROV    = "sgd_nesterov"
    
    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)
    
    def __call__(self, net: nn.Module, params: NNOptimParams):
        assert net is not None and params is not None
        
        match self:
            case Optim.SGD:
                return optim.SGD(
                    lr=params.lr_start
                    , params=net.parameters()
                    , momentum=params.momentum
                    , weight_decay=params.weight_decay
                )
            case Optim.ADAM:
                return optim.Adam(
                    lr=params.lr_start
                    , betas=params.momentum
                    , params=net.parameters()
                    , weight_decay=params.weight_decay
                )
            case Optim.ADAM_AMSGRAD:
                return optim.Adam(
                    amsgrad=True
                    , lr=params.lr_start
                    , betas=params.momentum
                    , params=net.parameters()
                    , weight_decay=params.weight_decay
                )
            case Optim.SGD_NESTEROV:
                return optim.SGD(
                    nesterov=True
                    , lr=params.lr_start
                    , params=net.parameters()
                    , momentum=params.momentum
                    , weight_decay=params.weight_decay
                )