from __future__ import annotations

import torch.nn.functional as F

from enum import Enum

class NNActivationFn(Enum):
    ELU         = 'elu'
    SELU        = 'selu'
    TANH        = 'tanh'
    RELU        = 'relu'
    SOFTMAX     = 'softmax'
    SIGMOID     = 'sigmoid'
    SOFTPLUS    = 'softplus'
    LEAKY_RELU  = 'leaky_relu'

    def __str__(self):
        return self.value
    
    def __repr__(self):
        return str(self)

    def __call__(self):
        match self:
            case NNActivationFn.ELU         : return F.elu
            case NNActivationFn.SELU        : return F.selu
            case NNActivationFn.TANH        : return F.tanh
            case NNActivationFn.RELU        : return F.relu
            case NNActivationFn.SOFTMAX     : return F.softmax
            case NNActivationFn.SIGMOID     : return F.sigmoid
            case NNActivationFn.SOFTPLUS    : return F.softplus
            case NNActivationFn.LEAKY_RELU  : return F.leaky_relu